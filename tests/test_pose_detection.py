from __future__ import annotations

import unittest

from src.adapters.pose import PoseDetectionAdapter, YOLOPoseDetector
from src.agent.event import Event
from src.agent.reducer import reduce_state
from src.agent.state import AgentState


class PoseReducerTestCase(unittest.TestCase):
    def test_posture_update_changes_state(self) -> None:
        state = AgentState()
        reduce_state(
            state,
            Event(
                type="user_posture_updated",
                timestamp=1000,
                payload={"posture": "sitting", "confidence": 0.9, "source": "test"},
            ),
        )
        self.assertEqual(state.user.posture, "sitting")
        self.assertEqual(state.user.posture_confidence, 0.9)

    def test_activity_update_changes_state(self) -> None:
        state = AgentState()
        reduce_state(
            state,
            Event(
                type="user_activity_updated",
                timestamp=1000,
                payload={"activity": "studying", "confidence": 0.85, "source": "test"},
            ),
        )
        self.assertEqual(state.user.current_activity, "studying")


class PoseAdapterTestCase(unittest.TestCase):
    def test_detector_placeholder_returns_result(self) -> None:
        detector = YOLOPoseDetector(model_path="yolov8n-pose.pt", device="cpu")
        self.assertTrue(detector.load_model())
        result = detector.detect()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.posture, "sitting")
        self.assertEqual(result.activity, "studying")

    def test_adapter_emits_events_on_change(self) -> None:
        events: list[Event] = []
        detector = YOLOPoseDetector(model_path="yolov8n-pose.pt", device="cpu")
        adapter = PoseDetectionAdapter(
            detector=detector,
            event_callback=events.append,
            detection_interval=0.01,
        )
        adapter.start()
        adapter._running = False
        if adapter._thread is not None:
            adapter._thread.join(timeout=2.0)
        self.assertTrue(any(event.type == "user_posture_updated" for event in events))
        self.assertTrue(any(event.type == "user_activity_updated" for event in events))


if __name__ == "__main__":
    unittest.main()
