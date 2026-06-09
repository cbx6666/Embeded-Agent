from __future__ import annotations

import threading
import time
import unittest

from src.agent.core.event_ingress import AgentEventIngress
from src.agent.event.event_model import Event


class EventIngressTest(unittest.TestCase):
    def test_coalesce_high_frequency_perception(self) -> None:
        processed: list[str] = []
        done = threading.Event()

        def processor(event: Event) -> tuple[list, list]:
            processed.append(str(event.type))
            if len(processed) >= 1:
                done.set()
            return [], []

        ingress = AgentEventIngress(processor)
        ingress.start()
        try:
            for i in range(20):
                ingress.submit(
                    Event(
                        type="user_fatigue_updated",
                        timestamp=i,
                        payload={"fatigue_level": "high", "confidence": 0.9},
                    )
                )
            self.assertTrue(done.wait(timeout=2.0))
            self.assertEqual(len(processed), 1)
        finally:
            ingress.stop()

    def test_speech_processed_on_worker(self) -> None:
        worker_ids: list[int] = []
        done = threading.Event()

        def processor(event: Event) -> tuple[list, list]:
            worker_ids.append(threading.get_ident())
            done.set()
            return [], []

        ingress = AgentEventIngress(processor)
        ingress.start()
        try:
            main_id = threading.get_ident()
            ingress.submit(Event(type="speech_recognized", timestamp=1, payload={"text": "hi"}))
            self.assertTrue(done.wait(timeout=2.0))
            self.assertEqual(len(worker_ids), 1)
            self.assertNotEqual(worker_ids[0], main_id)
        finally:
            ingress.stop()


if __name__ == "__main__":
    unittest.main()
