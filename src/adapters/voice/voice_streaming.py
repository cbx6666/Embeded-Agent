from __future__ import annotations

"""云端流式：LLM token 流 → 按句切分 → 逐句 TTS。"""

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol


class VoiceStreamSink(Protocol):
    """LLM 流式输出时逐句播报的回调。"""

    def on_sentence(self, sentence: str) -> None: ...

    @property
    def spoke_any(self) -> bool: ...


@dataclass
class SentenceChunker:
    """把 token 流切成可播报的句子。"""

    min_chars: int = 2
    _buffer: str = ""
    _sentences: list[str] = field(default_factory=list)

    _SPLIT_RE = re.compile(r"(?<=[。！？!?；;…\n])")

    def feed(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._buffer += delta
        parts = self._SPLIT_RE.split(self._buffer)
        if not parts:
            return []
        if not self._buffer.endswith(tuple("。！？!?；;…\n")):
            self._buffer = parts.pop()
        else:
            self._buffer = ""
        ready: list[str] = []
        for part in parts:
            sentence = part.strip()
            if len(sentence) >= self.min_chars:
                ready.append(sentence)
                self._sentences.append(sentence)
        return ready

    def flush(self) -> str:
        tail = self._buffer.strip()
        self._buffer = ""
        return tail


@dataclass
class CallbackVoiceStreamSink:
    speak_fn: Callable[[str], None]
    spoke_any: bool = False

    def on_sentence(self, sentence: str) -> None:
        text = sentence.strip()
        if not text:
            return
        self.speak_fn(text)
        self.spoke_any = True


def stream_text_to_sentences(text_stream, *, chunker: SentenceChunker | None = None):
    """把字符流 yield 成完整句子。"""
    chunker = chunker or SentenceChunker()
    for delta in text_stream:
        for sentence in chunker.feed(delta):
            yield sentence
    tail = chunker.flush()
    if tail:
        yield tail
