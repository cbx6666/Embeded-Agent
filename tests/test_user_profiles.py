from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.adapters.console_output import ConsoleOutput
from src.agent.user.personal_context_builder import PersonalContextBuilder
from src.agent.core import AgentCore
from src.services.runtime_history_service import RuntimeHistoryService
from src.services.timer_service import TimerService
from src.services.user_profile_service import DEFAULT_USER_ID, UserProfileService
from src.storage.json_store import JsonStore
from src.storage.long_term_memory_store import LongTermMemoryStore
from src.storage.user_profile_store import UserProfileStore
from tests.fakes.fake_llm_service import FakeLLMService


class UserProfileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_user_is_created(self) -> None:
        service = UserProfileService(UserProfileStore(self.root / "profiles.json"))

        user_ids = {profile.info.user_id for profile in service.list_users()}

        self.assertIn(DEFAULT_USER_ID, user_ids)
        self.assertTrue((self.root / "profiles.json").exists())

    def test_create_and_switch_user_updates_current_user(self) -> None:
        core = self._make_core()

        message = core.switch_user("alice", display_name="Alice", timestamp=1000)

        self.assertEqual(core.state.current_user_id, "alice")
        self.assertIn("Alice", message)
        self.assertEqual(core.personal_context_builder.user_profile_service.get_user("alice").info.display_name, "Alice")

    def test_set_user_preference(self) -> None:
        core = self._make_core()
        core.switch_user("alice", display_name="Alice", timestamp=1000)

        core.set_user_preference("favorite_content_types", "music,podcast", timestamp=1001)
        core.set_user_preference("reminder_style", "gentle", timestamp=1002)

        preference = core.personal_context_builder.user_profile_service.get_user("alice").preference
        self.assertEqual(preference.favorite_content_types, ["music", "podcast"])
        self.assertEqual(preference.reminder_style, "gentle")

    def test_set_user_info(self) -> None:
        core = self._make_core()
        core.switch_user("alice", display_name="Alice", timestamp=1000)

        core.set_user_info("age", "12", timestamp=1001)
        core.set_user_info("gender", "female", timestamp=1002)
        core.set_user_info("identity", "student", timestamp=1003)
        core.set_user_info("hobbies", "drawing,football", timestamp=1004)

        info = core.personal_context_builder.user_profile_service.get_user("alice").info
        self.assertEqual(info.age, 12)
        self.assertEqual(info.gender, "female")
        self.assertEqual(info.identity, "student")
        self.assertEqual(info.hobbies, ["drawing", "football"])

    def test_preferences_persist_after_reload(self) -> None:
        profile_path = self.root / "profiles.json"
        service = UserProfileService(UserProfileStore(profile_path))
        service.switch_user("alice", display_name="Alice", timestamp=1000)
        service.update_preference("alice", "favorite_music_styles", "lofi,classical", timestamp=1001)

        reloaded = UserProfileService(UserProfileStore(profile_path))
        preference = reloaded.get_user("alice").preference

        self.assertEqual(preference.favorite_music_styles, ["lofi", "classical"])

    def test_user_info_persists_after_reload(self) -> None:
        profile_path = self.root / "profiles.json"
        service = UserProfileService(UserProfileStore(profile_path))
        service.switch_user("alice", display_name="Alice", timestamp=1000)
        service.update_info("alice", "age", "12", timestamp=1001)
        service.update_info("alice", "identity", "student", timestamp=1002)
        service.update_info("alice", "hobbies", "drawing,football", timestamp=1003)

        reloaded = UserProfileService(UserProfileStore(profile_path))
        info = reloaded.get_user("alice").info

        self.assertEqual(info.age, 12)
        self.assertEqual(info.identity, "student")
        self.assertEqual(info.hobbies, ["drawing", "football"])

    def test_render_profile_shows_user_and_preferences(self) -> None:
        core = self._make_core()
        core.switch_user("alice", display_name="Alice", timestamp=1000)
        core.set_user_info("age", "12", timestamp=1001)
        core.set_user_info("hobbies", "drawing,football", timestamp=1002)
        core.set_user_preference("favorite_content_types", "music", timestamp=1001)
        core.set_user_preference("favorite_music_styles", "lofi", timestamp=1002)

        rendered = core.render_profile()

        self.assertIn("Alice", rendered)
        self.assertIn("age: 12", rendered)
        self.assertIn("hobbies: drawing, football", rendered)
        self.assertIn("favorite_content_types: music", rendered)
        self.assertIn("favorite_music_styles: lofi", rendered)

    def test_preferences_are_isolated_by_user(self) -> None:
        service = UserProfileService(UserProfileStore(self.root / "profiles.json"))
        service.switch_user("alice", display_name="Alice", timestamp=1000)
        service.update_preference("alice", "favorite_content_types", "music", timestamp=1001)
        service.switch_user("bob", display_name="Bob", timestamp=1002)
        service.update_preference("bob", "favorite_content_types", "podcast", timestamp=1003)

        self.assertEqual(service.get_user("alice").preference.favorite_content_types, ["music"])
        self.assertEqual(service.get_user("bob").preference.favorite_content_types, ["podcast"])

    def _make_core(self) -> AgentCore:
        return AgentCore(
            output=ConsoleOutput(silent=True),
            timer_service=TimerService(background=False),
            runtime_history_service=RuntimeHistoryService(),
            llm_service=FakeLLMService(reply_text="test reply"),
            store=JsonStore(self.root / "runtime.json"),
            personal_context_builder=PersonalContextBuilder(
                long_term_memory_store=LongTermMemoryStore(self.root / "long_term_memory.json"),
                user_profile_service=UserProfileService(UserProfileStore(self.root / "profiles.json")),
            ),
        )


if __name__ == "__main__":
    unittest.main()
