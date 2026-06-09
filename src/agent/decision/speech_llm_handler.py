from __future__ import annotations

"""speech_recognized 单次语音 LLM 处理器。

媒体：仅 LLM 根据完整曲库选 track_id；用户点播须先 TTS 播报再播放；关怀建议须 pending 批准。
"""

import logging

from src.agent.action import action_builders as action_build
from src.agent.action.realizer import ActionRealizer
from src.agent.core.models import DecisionResult, Event, Intent
from src.agent.llm.client import LLMClient
from src.agent.llm.prompt_builder import build_speech_prompt
from src.agent.llm.reply_validator import normalize_reply, validate_tts_reply
from src.agent.media.intent_parser import parse_llm_media_control, parse_media_control
from src.agent.media.media_models import MediaRequest, MediaSource
from src.agent.policy_config import LLMRoutingPolicy
from src.agent.state.agent_state import AgentState

logger = logging.getLogger(__name__)


_ALLOWED_INTENTS = {
    "answer_user",
    "start_focus",
    "stop_focus",
    "set_tts_volume",
    "no_op",
    "media_control",
}


class SpeechLLMHandler:
    def __init__(
        self,
        *,
        realizer: ActionRealizer | None = None,
        policy: LLMRoutingPolicy | None = None,
        media_controller: object | None = None,
    ) -> None:
        self.realizer = realizer or ActionRealizer()
        self.policy = policy or LLMRoutingPolicy()
        self.media_controller = media_controller

    def decide(
        self,
        *,
        state: AgentState,
        event: Event,
        llm_client: LLMClient,
        user_context: dict[str, object] | None = None,
    ) -> DecisionResult:
        user_text = str(event.payload.get("text", "")).strip()
        if not user_text:
            return DecisionResult(
                intents=[Intent("no_op", "empty speech text")],
                source="speech_llm",
                reason="empty speech text",
            )

        mc = self.media_controller
        is_playing = mc.is_playing() if mc is not None else False
        media_context = self._build_media_context(mc) if mc is not None else None

        prompt = build_speech_prompt(
            state=state,
            user_context=user_context or {},
            user_text=user_text,
            media_context=media_context,
        )

        used_llm = True
        try:
            data = llm_client.complete_json(
                self.policy.speech_prompt, prompt, temperature=self.policy.reply_temperature
            )
            llm_media = parse_llm_media_control(data)
            if llm_media is not None and mc is not None:
                if self._is_pending_rejection(user_text, media_context):
                    return self._reject_pending(mc, data, user_context)
                return self._handle_media_control(
                    llm_media,
                    state,
                    user_context,
                    used_llm=True,
                    pending=self._pending_suggestion(mc),
                )

            intent, reply = self._parse(data)
            reply = normalize_reply(reply)
            if intent.type == "answer_user":
                valid, invalid_reason = validate_tts_reply(reply)
                if not valid:
                    if mc is not None:
                        mc.on_user_command_finished()
                    return DecisionResult(
                        intents=[Intent("no_op", f"invalid llm reply: {invalid_reason}")],
                        actions=[],
                        used_llm=used_llm,
                        source="speech_llm",
                        reason=f"invalid_llm_reply:{invalid_reason}",
                        log_fields={
                            "invalid_llm_reply": True,
                            "invalid_llm_reply_reason": invalid_reason,
                        },
                    )
            elif intent.type != "no_op" and reply:
                valid, _invalid_reason = validate_tts_reply(reply)
                if not valid:
                    reply = ""
            if mc is not None and self._is_pending_rejection(user_text, media_context):
                return self._reject_pending(mc, data, user_context, reply=reply)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[语音LLM] 决策失败，尝试播放中控制指令兜底：%s", exc)
            if mc is not None and is_playing:
                fallback = parse_media_control(user_text, is_playing=True)
                if fallback is not None and fallback.action != "play_media":
                    return self._handle_media_control(
                        fallback,
                        state,
                        user_context,
                        used_llm=False,
                    )
            if mc is not None:
                mc.on_user_command_finished()
            return DecisionResult(
                intents=[Intent("no_op", f"llm_failed:{exc}")],
                actions=[],
                used_llm=False,
                source="speech_llm",
                reason=f"llm_failed:{exc}",
                log_fields={"llm_failed": True, "fallback_suppressed": True},
            )

        actions = self.realizer.realize([intent], reply_text=reply)
        if mc is not None:
            mc.on_user_command_finished()
        return DecisionResult(
            intents=[intent],
            actions=actions,
            used_llm=used_llm,
            source="speech_llm",
            reason=intent.reason,
            reply_text=reply,
        )

    @staticmethod
    def _build_media_context(mc: object) -> dict[str, object]:
        playback = mc.get_playback_state()
        pending = playback.pending_suggestion
        current_title = None
        track_id = playback.current_track_id
        selector = getattr(mc, "selector", None)
        if track_id and selector is not None:
            for track in selector.index.tracks:
                if track.id == track_id:
                    current_title = track.title
                    break
        from src.agent.media.media_library import build_library_catalog

        library: dict[str, object] = {"tracks": [], "folders": [], "total": 0}
        if selector is not None:
            library = build_library_catalog(selector.index)
        return {
            "is_playing": bool(playback.is_playing),
            "current_track_id": track_id,
            "current_track_title": current_title,
            "current_media_type": playback.current_media_type,
            "current_category": playback.current_category,
            "recent_played_ids": list(playback.recent_played_ids),
            "pending_suggestion": pending,
            "library": library,
        }

    @staticmethod
    def _pending_suggestion(mc: object) -> dict | None:
        pending = mc.get_playback_state().pending_suggestion
        return dict(pending) if isinstance(pending, dict) else None

    @staticmethod
    def _is_pending_rejection(user_text: str, media_context: dict | None) -> bool:
        if not media_context or not media_context.get("pending_suggestion"):
            return False
        from src.agent.media.media_intent_parser import is_media_rejection

        return is_media_rejection(user_text)

    def _reject_pending(
        self,
        mc: object,
        data: dict,
        user_context: dict,
        *,
        reply: str = "",
    ) -> DecisionResult:
        reply = reply or str(data.get("reply", "")).strip() or "好的，那先不放。"
        mc.reject_pending_suggestion()
        mc.on_user_command_finished()
        return DecisionResult(
            intents=[Intent("answer_user", "reject_media_suggestion")],
            actions=self.realizer.realize(
                [Intent("answer_user", "reject_media_suggestion")],
                reply_text=reply,
            ),
            used_llm=True,
            source="speech_llm",
            reason="reject_media_suggestion",
            reply_text=reply,
        )

    def _handle_media_control(
        self,
        media_intent,
        state,
        user_context,
        *,
        used_llm: bool,
        pending: dict | None = None,
    ) -> DecisionResult:
        mc = self.media_controller
        assert mc is not None

        action = media_intent.action
        reply = media_intent.reply or ""
        ctx = mc.build_selection_context(agent_state=state, user_context=user_context)
        pending = pending or self._pending_suggestion(mc)

        if action == "play_media" and not pending:
            if not media_intent.track_id:
                reply = reply or "我还没在曲库里找到合适的歌，你可以再说具体一点。"
                return self._answer_only(reply, "media_play_no_track_id", used_llm=used_llm)
            track = mc.get_track_by_id(media_intent.track_id)
            if track is None:
                reply = reply or "这首在本地曲库里找不到，换一首试试？"
                return self._answer_only(reply, "media_play_invalid_track_id", used_llm=used_llm)
            return self._play_track_result(
                track,
                reply,
                MediaSource.USER_EXPLICIT,
                "media_control:play_media",
                used_llm=used_llm,
            )

        if action == "play_media" and pending:
            if not media_intent.track_id:
                reply = reply or "好的，不过暂时没找到合适的音频。"
                return self._answer_only(reply, "confirm_media_no_track", used_llm=used_llm)
            track = mc.get_track_by_id(media_intent.track_id)
            if track is None:
                reply = reply or "好的，不过暂时没找到合适的音频。"
                return self._answer_only(reply, "confirm_media_no_track", used_llm=used_llm)
            mc.get_playback_state().pending_suggestion = None
            return self._play_track_result(
                track,
                reply,
                MediaSource.AGENT_SUGGESTION,
                "confirm_media_suggestion",
                used_llm=used_llm,
            )

        if action in {"stop_media", "pause_media", "resume_media"}:
            intent_type = action
            actions = self.realizer.realize([Intent(intent_type, "media_control")])
            if reply:
                actions.extend(self.realizer.realize([Intent("answer_user", "ack")], reply_text=reply))
            mc.on_user_command_finished()
            return DecisionResult(
                intents=[Intent(intent_type, "media_control")],
                actions=actions,
                used_llm=used_llm,
                source="speech_llm",
                reason=f"media_control:{action}",
                reply_text=reply,
            )

        if action == "next_media":
            if not media_intent.track_id:
                reply = reply or "暂时没有别的可以换了。"
                return self._answer_only(reply, "next_media_empty", used_llm=used_llm)
            current_id = mc.get_playback_state().current_track_id
            if media_intent.track_id == current_id:
                reply = reply or "已经是这首了，要不要换别的风格？"
                return self._answer_only(reply, "next_media_same_track", used_llm=used_llm)
            track = mc.get_track_by_id(media_intent.track_id)
            if track is None:
                reply = reply or "暂时没有别的可以换了。"
                return self._answer_only(reply, "next_media_empty", used_llm=used_llm)
            return self._play_track_result(
                track,
                reply,
                MediaSource.USER_EXPLICIT,
                "media_control:next_media",
                used_llm=used_llm,
            )

        mc.on_user_command_finished()
        return DecisionResult(
            intents=[Intent("no_op", "unknown media action")],
            used_llm=used_llm,
            source="speech_llm",
            reason="unknown_media_action",
        )

    def _play_track_result(
        self,
        track,
        reply,
        source,
        reason,
        *,
        used_llm: bool,
    ) -> DecisionResult:
        mc = self.media_controller
        spoken = (reply or "").strip() or f"好的，这就给你放《{track.title}》。"
        play_payload = {
            "track_id": track.id,
            "path": track.path,
            "title": track.title,
            "media_type": track.media_type,
            "category": track.category,
            "source": source.value if hasattr(source, "value") else str(source),
            "defer_after_speak": True,
        }
        intents = [Intent("play_media", "media_control", payload=play_payload)]
        actions = [
            action_build.speak(spoken, kind="agent_reply", reason="media_play_ack"),
            action_build.display(spoken),
            action_build.play_media(
                track_id=track.id,
                path=track.path,
                title=track.title,
                media_type=track.media_type,
                category=track.category,
                source=play_payload["source"],
                defer_after_speak=True,
            ),
        ]
        if mc is not None:
            mc.on_user_command_finished()
        return DecisionResult(
            intents=intents,
            actions=actions,
            used_llm=used_llm,
            source="speech_llm",
            reason=reason,
            reply_text=spoken,
        )

    def _answer_only(self, reply: str, reason: str, *, used_llm: bool) -> DecisionResult:
        mc = self.media_controller
        if mc is not None:
            mc.on_user_command_finished()
        return DecisionResult(
            intents=[Intent("answer_user", reason)],
            actions=self.realizer.realize([Intent("answer_user", reason)], reply_text=reply),
            used_llm=used_llm,
            source="speech_llm",
            reason=reason,
            reply_text=reply,
        )

    @staticmethod
    def _parse(data: dict) -> tuple[Intent, str]:
        intent_type = str(data.get("intent", "")).strip()
        if intent_type not in _ALLOWED_INTENTS:
            intent_type = "answer_user"
        reply = normalize_reply(data.get("reply", ""))
        payload: dict[str, object] = {}
        if intent_type == "start_focus" and data.get("duration_sec") is not None:
            payload["duration_sec"] = data.get("duration_sec")
        if intent_type == "set_tts_volume" and data.get("volume") is not None:
            payload["volume"] = data.get("volume")
        return Intent(intent_type, "speech_recognized llm decision", payload), reply
