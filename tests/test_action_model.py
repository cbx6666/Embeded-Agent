import unittest

from src.agent.action.action_builders import _build_action, set_tts_volume, speak


class ActionModelTests(unittest.TestCase):
    def test_action_factories_create_registered_action_types(self) -> None:
        """Action 工厂只能产出已注册动作类型。"""
        self.assertEqual(speak("你好").type, "speak")
        self.assertEqual(set_tts_volume(30).type, "set_tts_volume")

    def test_build_action_rejects_unknown_action_type(self) -> None:
        """未知动作类型应在工厂入口被拦截，避免绕过 ActionType 闭集。"""
        with self.assertRaises(ValueError):
            _build_action("unknown_action")


if __name__ == "__main__":
    unittest.main()
