from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.agent.action import Action
from src.agent.event import Event
from src.agent.execution.action_result import ActionResult
from src.agent.memory.memory_background_worker import MemoryBackgroundWorker
from src.agent.memory.memory_candidate import MemoryCandidate
from src.agent.memory.memory_task import (
    build_action_memory_task,
    build_event_memory_task,
)
from src.agent.state import AgentState
from src.storage.long_term_memory_store import LongTermMemoryStore
from tests.fakes.fake_llm_service import FakeLLMService


class _CapturingPipeline:
    def __init__(self, *, block: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.event_calls: list[tuple[str, str, dict[str, object]]] = []
        self.action_calls: list[dict[str, object]] = []
        self.thread_name = ""
        if not block:
            self.release.set()

    def process_event(self, user_id, event, state, llm_service):
        del llm_service
        self.thread_name = threading.current_thread().name
        self.started.set()
        self.release.wait(timeout=2)
        self.event_calls.append((user_id, state.current_user_id, dict(event.payload)))
        return None

    def process_actions(self, user_id, actions, timestamp, **kwargs):
        del user_id, timestamp
        self.thread_name = threading.current_thread().name
        self.started.set()
        self.release.wait(timeout=2)
        self.action_calls.append(
            {
                "state_user_id": kwargs["state"].current_user_id,
                "action_payload": dict(actions[0].payload),
            }
        )
        return None


class _FailingPipeline:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.call_count = 0

    def process_event(self, user_id, event, state, llm_service):
        del user_id, event, state, llm_service
        self.call_count += 1
        if self.call_count <= self.failures_before_success:
            raise RuntimeError("scripted memory failure")
        return None

    def process_actions(self, user_id, actions, timestamp, **kwargs):
        del user_id, actions, timestamp, kwargs
        return None


def _event(timestamp: int, text: str | None = None) -> Event:
    return Event(
        type="user_text_input",
        timestamp=timestamp,
        payload={"text": text or f"remember preference {timestamp}", "source": "test"},
    )


class MemoryBackgroundWorkerTestCase(unittest.TestCase):
    def test_event_worker_runs_in_background_with_snapshot_and_metrics(self) -> None:
        pipeline = _CapturingPipeline(block=True)
        worker = MemoryBackgroundWorker(pipeline, FakeLLMService())  # type: ignore[arg-type]
        state = AgentState(current_user_id="u1")
        event = _event(1, "from now on I prefer one hour focus")

        result = worker.submit_event_memory(user_id="u1", event=event, state=state)
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "enqueued")
        self.assertIsNotNone(result.task_id)
        self.assertTrue(pipeline.started.wait(timeout=1))

        state.current_user_id = "changed"
        event.payload["text"] = "changed"
        pipeline.release.set()
        self.assertTrue(worker.wait_for_idle(timeout=2))
        shutdown_result = worker.shutdown()

        self.assertTrue(pipeline.thread_name.startswith("memory-worker"))
        self.assertEqual(pipeline.event_calls[0][0:2], ("u1", "u1"))
        self.assertEqual(
            pipeline.event_calls[0][2]["text"],
            "from now on I prefer one hour focus",
        )
        metrics = worker.get_metrics()
        self.assertEqual(metrics["submitted_count"], 1)
        self.assertEqual(metrics["enqueued_count"], 1)
        self.assertEqual(metrics["processed_count"], 1)
        self.assertEqual(metrics["failed_count"], 0)
        self.assertGreaterEqual(metrics["average_process_time_ms"], 0.0)
        self.assertFalse(shutdown_result["timed_out"])

    def test_action_worker_uses_snapshot(self) -> None:
        pipeline = _CapturingPipeline(block=True)
        worker = MemoryBackgroundWorker(pipeline, FakeLLMService())  # type: ignore[arg-type]
        state = AgentState(current_user_id="u1")
        action = Action(type="start_timer", payload={"duration_sec": 3600})

        result = worker.submit_action_memory(
            user_id="u1",
            actions=[action],
            timestamp=1,
            action_results=[ActionResult(action_type="start_timer", success=True, timestamp=1)],
            source_event=_event(1, "start one hour focus"),
            state=state,
        )
        self.assertTrue(result.accepted)
        self.assertTrue(pipeline.started.wait(timeout=1))
        state.current_user_id = "changed"
        action.payload["duration_sec"] = 1
        pipeline.release.set()
        self.assertTrue(worker.wait_for_idle(timeout=2))
        worker.shutdown()

        self.assertEqual(pipeline.action_calls[0]["state_user_id"], "u1")
        self.assertEqual(
            pipeline.action_calls[0]["action_payload"],
            {"duration_sec": 3600},
        )

    def test_stable_task_id_ignores_state_snapshot_and_separates_task_types(self) -> None:
        event = _event(1, "remember this")
        first_state = AgentState(current_user_id="u1")
        second_state = AgentState(current_user_id="changed")
        first = build_event_memory_task(user_id="u1", event=event, state=first_state)
        second = build_event_memory_task(user_id="u1", event=event, state=second_state)
        action = build_action_memory_task(
            user_id="u1",
            actions=[Action(type="speak", payload={"text": "ok"})],
            timestamp=1,
            action_results=[ActionResult(action_type="speak", success=True, timestamp=1)],
            source_event=event,
            state=first_state,
        )

        self.assertEqual(first.task_id, second.task_id)
        self.assertNotEqual(first.task_id, action.task_id)
        self.assertEqual(first.priority, 100)
        self.assertEqual(action.priority, 30)

    def test_duplicate_task_is_rejected_and_executed_once(self) -> None:
        pipeline = _CapturingPipeline(block=True)
        worker = MemoryBackgroundWorker(pipeline, FakeLLMService())  # type: ignore[arg-type]
        event = _event(1)
        state = AgentState(current_user_id="u1")

        first = worker.submit_event_memory(user_id="u1", event=event, state=state)
        self.assertTrue(pipeline.started.wait(timeout=1))
        second = worker.submit_event_memory(user_id="u1", event=event, state=state)
        pipeline.release.set()
        self.assertTrue(worker.wait_for_idle(timeout=2))
        third = worker.submit_event_memory(user_id="u1", event=event, state=state)
        worker.shutdown()

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, "duplicate_task")
        self.assertFalse(third.accepted)
        self.assertEqual(third.reason, "duplicate_task")
        self.assertEqual(len(pipeline.event_calls), 1)
        self.assertEqual(worker.get_metrics()["duplicate_count"], 2)

    def test_full_queue_rejects_new_task_and_records_drop(self) -> None:
        pipeline = _CapturingPipeline(block=True)
        worker = MemoryBackgroundWorker(
            pipeline,
            FakeLLMService(),  # type: ignore[arg-type]
            max_queue_size=1,
        )
        state = AgentState(current_user_id="u1")

        first = worker.submit_event_memory(user_id="u1", event=_event(1), state=state)
        self.assertTrue(first.accepted)
        self.assertTrue(pipeline.started.wait(timeout=1))
        second = worker.submit_event_memory(user_id="u1", event=_event(2), state=state)
        third = worker.submit_event_memory(user_id="u1", event=_event(3), state=state)

        self.assertTrue(second.accepted)
        self.assertFalse(third.accepted)
        self.assertEqual(third.reason, "queue_full")
        self.assertEqual(worker.get_metrics()["dropped_count"], 1)

        pipeline.release.set()
        self.assertTrue(worker.wait_for_idle(timeout=2))
        worker.shutdown()
        self.assertEqual(
            [payload["text"] for _, _, payload in pipeline.event_calls],
            ["remember preference 1", "remember preference 2"],
        )

    def test_single_worker_preserves_fifo_order(self) -> None:
        pipeline = _CapturingPipeline()
        worker = MemoryBackgroundWorker(pipeline, FakeLLMService())  # type: ignore[arg-type]
        state = AgentState(current_user_id="u1")

        results = [
            worker.submit_event_memory(user_id="u1", event=_event(index), state=state)
            for index in range(1, 5)
        ]
        self.assertTrue(all(result.accepted for result in results))
        self.assertTrue(worker.wait_for_idle(timeout=2))
        worker.shutdown()

        self.assertEqual(
            [payload["text"] for _, _, payload in pipeline.event_calls],
            [f"remember preference {index}" for index in range(1, 5)],
        )

    def test_transient_failure_retries_then_succeeds(self) -> None:
        pipeline = _FailingPipeline(failures_before_success=1)
        worker = MemoryBackgroundWorker(
            pipeline,
            FakeLLMService(),  # type: ignore[arg-type]
            max_retries=2,
            retry_backoff_sec=(0.0,),
        )

        result = worker.submit_event_memory(
            user_id="u1",
            event=_event(1),
            state=AgentState(current_user_id="u1"),
        )
        self.assertTrue(result.accepted)
        self.assertTrue(worker.wait_for_idle(timeout=2))
        worker.shutdown()

        metrics = worker.get_metrics()
        self.assertEqual(pipeline.call_count, 2)
        self.assertEqual(metrics["retried_count"], 1)
        self.assertEqual(metrics["processed_count"], 1)
        self.assertEqual(metrics["failed_count"], 0)
        self.assertEqual(metrics["dead_letter_count"], 0)

    def test_permanent_failure_moves_task_to_dead_letter(self) -> None:
        pipeline = _FailingPipeline(failures_before_success=10)
        worker = MemoryBackgroundWorker(
            pipeline,
            FakeLLMService(),  # type: ignore[arg-type]
            max_retries=2,
            retry_backoff_sec=(0.0,),
        )

        result = worker.submit_event_memory(
            user_id="u1",
            event=_event(1, "private preference text"),
            state=AgentState(current_user_id="u1"),
        )
        self.assertTrue(result.accepted)
        self.assertTrue(worker.wait_for_idle(timeout=2))
        worker.shutdown()

        metrics = worker.get_metrics()
        dead_letters = worker.get_dead_letters()
        self.assertEqual(pipeline.call_count, 3)
        self.assertEqual(metrics["retried_count"], 2)
        self.assertEqual(metrics["failed_count"], 1)
        self.assertEqual(metrics["dead_letter_count"], 1)
        self.assertEqual(dead_letters[0]["task_id"], result.task_id)
        self.assertEqual(dead_letters[0]["retry_count"], 2)
        self.assertNotIn("private preference text", str(dead_letters[0]))
        self.assertEqual(dead_letters[0]["payload_summary"]["text_length"], 23)

    def test_shutdown_drains_and_rejects_later_submissions(self) -> None:
        pipeline = _CapturingPipeline()
        worker = MemoryBackgroundWorker(pipeline, FakeLLMService())  # type: ignore[arg-type]
        state = AgentState(current_user_id="u1")
        accepted = worker.submit_event_memory(user_id="u1", event=_event(1), state=state)

        shutdown_result = worker.shutdown(timeout=1)
        rejected = worker.submit_event_memory(user_id="u1", event=_event(2), state=state)

        self.assertTrue(accepted.accepted)
        self.assertFalse(shutdown_result["timed_out"])
        self.assertEqual(shutdown_result["remaining_queue_size"], 0)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "worker_shutdown")

    def test_shutdown_timeout_reports_remaining_work(self) -> None:
        pipeline = _CapturingPipeline(block=True)
        worker = MemoryBackgroundWorker(pipeline, FakeLLMService())  # type: ignore[arg-type]
        worker.submit_event_memory(
            user_id="u1",
            event=_event(1),
            state=AgentState(current_user_id="u1"),
        )
        self.assertTrue(pipeline.started.wait(timeout=1))

        shutdown_result = worker.shutdown(timeout=0.01)
        self.assertTrue(shutdown_result["timed_out"])
        self.assertEqual(shutdown_result["remaining_queue_size"], 1)

        pipeline.release.set()
        self.assertTrue(worker.wait_for_idle(timeout=2))
        worker.shutdown(timeout=1)

    def test_long_term_store_serializes_concurrent_reads_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermMemoryStore(Path(temp_dir) / "memory.json")

            def write_and_read(index: int) -> None:
                store.upsert_candidate(
                    "u1",
                    MemoryCandidate(
                        memory_type="behavior_pattern",
                        content=f"pattern-{index}",
                        confidence=0.7,
                        evidence=[{"event_type": "test", "timestamp": index}],
                    ),
                    timestamp=index,
                )
                store.list("u1")

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(write_and_read, index) for index in range(20)]
                for future in futures:
                    future.result(timeout=2)

            self.assertEqual(len(store.list("u1")), 20)


if __name__ == "__main__":
    unittest.main()
