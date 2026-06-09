from __future__ import annotations

import unittest

import numpy as np

from src.adapters.behavior.pose_inference import infer_posture_and_activity
from src.adapters.vision_common.yolo_ultralytics_ops import PosePerson
from src.agent.event import Event
from src.agent.state.reducer import reduce_state
from src.agent.state import AgentState


class PoseReducerTestCase(unittest.TestCase):
    def test_posture_update_changes_state(self) -> None:
        state = AgentState()
        reduce_state(
            state,
            Event(
                type="user_posture_updated",
                timestamp=1000,
                payload={"posture": "sitting", "confidence": 0.9, "source": "yolo26_pose_om_v1"},
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
                payload={"activity": "studying", "confidence": 0.85, "source": "yolo26_pose_om_v1"},
            ),
        )
        self.assertEqual(state.user.current_activity, "studying")


class PoseInferenceTestCase(unittest.TestCase):
    def _desk_person(self) -> PosePerson:
        # 肩在上、髋在下、膝再下 → sitting
        xy = np.zeros((17, 2), dtype=np.float32)
        conf = np.ones(17, dtype=np.float32)
        xy[5] = (200, 100)
        xy[6] = (240, 100)
        xy[11] = (210, 200)
        xy[12] = (250, 200)
        xy[13] = (210, 280)
        xy[14] = (250, 280)
        xy[0] = (225, 80)
        return PosePerson(keypoints_xy=xy, keypoints_conf=conf, box_conf=0.9)

    def test_sitting_working_when_person_visible(self) -> None:
        posture, activity, conf = infer_posture_and_activity(
            person=self._desk_person(),
            person_visible=True,
            presence_phase="present",
            phone_in_hand=False,
            looking_down=False,
        )
        self.assertEqual(posture, "sitting")
        self.assertEqual(activity, "working")
        self.assertGreater(conf, 0.5)

    def test_phone_use_activity(self) -> None:
        posture, activity, _ = infer_posture_and_activity(
            person=self._desk_person(),
            person_visible=True,
            presence_phase="present",
            phone_in_hand=True,
            looking_down=False,
        )
        self.assertEqual(activity, "phone_use")

    def test_unknown_when_left(self) -> None:
        posture, activity, conf = infer_posture_and_activity(
            person=None,
            person_visible=False,
            presence_phase="left",
            phone_in_hand=False,
            looking_down=False,
        )
        self.assertEqual(posture, "unknown")
        self.assertEqual(activity, "unknown")
        self.assertEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main()
