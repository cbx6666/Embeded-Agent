from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.core import AgentCore
from src.agent.event import Event
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import DEFAULT_USER_ID, UserProfileService
from src.storage.json_store import JsonStore
from src.storage.profile_store import ProfileStore


class StubLLMService(LLMService):
    def __init__(self) -> None:
        pass

    def generate_reply(self, text: str, state) -> str:  # type: ignore[override]
        return "测试回复。"


class UserProfileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_user_is_created(self) -> None:
        service = UserProfileService(ProfileStore(self.root / "profiles.json"))

        user_ids = {profile.info.user_id for profile in service.list_users()}

        self.assertIn(DEFAULT_USER_ID, user_ids)
        self.assertTrue((self.root / "profiles.json").exists())

    def test_create_and_switch_user_updates_current_user(self) -> None:
        core = self._make_core()

        message = core.switch_user("xiaoli", display_name="小李", timestamp=1000)

        self.assertEqual(core.state.current_user_id, "xiaoli")
        self.assertIn("小李", message)
        self.assertEqual(core.profile_service.get_user("xiaoli").info.display_name, "小李")

    def test_set_user_preference(self) -> None:
        core = self._make_core()
        core.switch_user("xiaoli", display_name="小李", timestamp=1000)

        core.set_user_preference("favorite_content_types", "相声,脱口秀", timestamp=1001)
        core.set_user_preference("reminder_style", "温和", timestamp=1002)

        preference = core.profile_service.get_user("xiaoli").preference
        self.assertEqual(preference.favorite_content_types, ["相声", "脱口秀"])
        self.assertEqual(preference.reminder_style, "温和")

    def test_set_user_info(self) -> None:
        core = self._make_core()
        core.switch_user("xiaoli", display_name="小李", timestamp=1000)

        core.set_user_info("age", "12", timestamp=1001)
        core.set_user_info("gender", "女", timestamp=1002)
        core.set_user_info("identity", "小学生", timestamp=1003)
        core.set_user_info("hobbies", "画画,足球", timestamp=1004)

        info = core.profile_service.get_user("xiaoli").info
        self.assertEqual(info.age, 12)
        self.assertEqual(info.gender, "女")
        self.assertEqual(info.identity, "小学生")
        self.assertEqual(info.hobbies, ["画画", "足球"])

    def test_preferences_persist_after_reload(self) -> None:
        profile_path = self.root / "profiles.json"
        service = UserProfileService(ProfileStore(profile_path))
        service.switch_user("xiaoli", display_name="小李", timestamp=1000)
        service.update_preference("xiaoli", "favorite_music_styles", "轻音乐,古风", timestamp=1001)

        reloaded = UserProfileService(ProfileStore(profile_path))
        preference = reloaded.get_user("xiaoli").preference

        self.assertEqual(preference.favorite_music_styles, ["轻音乐", "古风"])

    def test_user_info_persists_after_reload(self) -> None:
        profile_path = self.root / "profiles.json"
        service = UserProfileService(ProfileStore(profile_path))
        service.switch_user("xiaoli", display_name="小李", timestamp=1000)
        service.update_info("xiaoli", "age", "12", timestamp=1001)
        service.update_info("xiaoli", "identity", "小学生", timestamp=1002)
        service.update_info("xiaoli", "hobbies", "画画,足球", timestamp=1003)

        reloaded = UserProfileService(ProfileStore(profile_path))
        info = reloaded.get_user("xiaoli").info

        self.assertEqual(info.age, 12)
        self.assertEqual(info.identity, "小学生")
        self.assertEqual(info.hobbies, ["画画", "足球"])

    def test_render_profile_shows_user_and_preferences(self) -> None:
        core = self._make_core()
        core.switch_user("xiaoli", display_name="小李", timestamp=1000)
        core.set_user_info("age", "12", timestamp=1001)
        core.set_user_info("hobbies", "画画,足球", timestamp=1002)
        core.set_user_preference("favorite_content_types", "音乐", timestamp=1001)
        core.set_user_preference("favorite_music_styles", "轻音乐", timestamp=1002)

        rendered = core.render_profile()

        self.assertIn("小李", rendered)
        self.assertIn("age: 12", rendered)
        self.assertIn("hobbies: 画画, 足球", rendered)
        self.assertIn("favorite_content_types: 音乐", rendered)
        self.assertIn("favorite_music_styles: 轻音乐", rendered)

    def test_rest_suggestion_uses_current_user_preferences(self) -> None:
        core = self._make_core()
        core.switch_user("xiaoli", display_name="小李", timestamp=1000)
        core.set_user_preference("favorite_content_types", "相声", timestamp=1001)
        core.set_user_preference("reminder_style", "温和", timestamp=1002)
        core.state.focus.active = True
        core.state.focus.start_ts = 0
        core.state.focus.elapsed_sec = 600
        core.state.focus.remaining_sec = 900
        core.state.interaction.mode = "focus"
        core.state.user.presence = "present"
        core.state.user.fatigue_level = "high"

        actions, _ = core.handle_event_with_results(
            Event(
                type="system_triggered",
                timestamp=1600,
                payload={"trigger": "focus_health_check", "source": "test"},
            )
        )
        texts = [str(action.payload.get("text", "")) for action in actions]

        self.assertTrue(any("小李" in text for text in texts))
        self.assertTrue(any("相声" in text for text in texts))
        self.assertTrue(any("有点累啦" in text for text in texts))

    def test_music_content_type_can_use_music_style(self) -> None:
        core = self._make_core()
        core.switch_user("xiaoli", display_name="小李", timestamp=1000)
        core.set_user_preference("favorite_content_types", "音乐", timestamp=1001)
        core.set_user_preference("favorite_music_styles", "轻音乐", timestamp=1002)
        core.state.focus.active = True
        core.state.focus.start_ts = 0
        core.state.focus.elapsed_sec = 600
        core.state.focus.remaining_sec = 900
        core.state.interaction.mode = "focus"
        core.state.user.presence = "present"
        core.state.user.fatigue_level = "high"

        actions, _ = core.handle_event_with_results(
            Event(
                type="system_triggered",
                timestamp=1600,
                payload={"trigger": "focus_health_check", "source": "test"},
            )
        )
        texts = [str(action.payload.get("text", "")) for action in actions]

        self.assertTrue(any("轻音乐" in text for text in texts))

    def test_preferences_are_isolated_by_user(self) -> None:
        service = UserProfileService(ProfileStore(self.root / "profiles.json"))
        service.switch_user("xiaoli", display_name="小李", timestamp=1000)
        service.update_preference("xiaoli", "favorite_content_types", "相声", timestamp=1001)
        service.switch_user("xiaowang", display_name="小王", timestamp=1002)
        service.update_preference("xiaowang", "favorite_content_types", "脱口秀", timestamp=1003)

        self.assertEqual(service.get_user("xiaoli").preference.favorite_content_types, ["相声"])
        self.assertEqual(service.get_user("xiaowang").preference.favorite_content_types, ["脱口秀"])

    def _make_core(self) -> AgentCore:
        return AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            memory_service=MemoryService(),
            llm_service=StubLLMService(),
            store=JsonStore(self.root / "runtime.json"),
            profile_service=UserProfileService(ProfileStore(self.root / "profiles.json")),
        )


if __name__ == "__main__":
    unittest.main()
