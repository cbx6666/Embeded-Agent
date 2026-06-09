"""Sherpa-ONNX 关键词唤醒（KWS）工具与检测器。

模型下载（无需注册）：
  https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models

或运行：python scripts/setup_sherpa_kws.py --keyword 小助
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_SHERPA_KWS_DIR = Path(
    "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
)
DEFAULT_KEYWORDS_FILE = Path("models/sherpa-kws-keywords.txt")
DEFAULT_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2"
)
TRANSDUCER_BASENAME = "epoch-12-avg-2-chunk-16-left-64"


def resolve_sherpa_kws_dir(model_dir: str | Path | None = None) -> Path:
    """解析 Sherpa KWS 模型目录（需含 tokens.txt）。"""
    raw = model_dir or os.environ.get("EMBED_SHERPA_KWS_DIR") or DEFAULT_SHERPA_KWS_DIR
    path = Path(raw).expanduser()
    if path.is_file() and path.name == "tokens.txt":
        return path.parent
    if (path / "tokens.txt").is_file():
        return path.resolve()
    raise FileNotFoundError(
        f"Sherpa KWS 模型目录无效：{path}\n"
        f"请先运行：python scripts/setup_sherpa_kws.py --keyword 小助"
    )


def resolve_transducer_paths(model_dir: Path, *, use_int8: bool = True) -> dict[str, Path]:
    """返回 encoder/decoder/joiner/tokens 路径。"""
    suffix = ".int8.onnx" if use_int8 else ".onnx"
    tokens = model_dir / "tokens.txt"
    encoder = model_dir / f"encoder-{TRANSDUCER_BASENAME}{suffix}"
    decoder = model_dir / f"decoder-{TRANSDUCER_BASENAME}{suffix}"
    joiner = model_dir / f"joiner-{TRANSDUCER_BASENAME}{suffix}"
    missing = [p for p in (tokens, encoder, decoder, joiner) if not p.is_file()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(f"Sherpa KWS 模型文件缺失：{names}（目录 {model_dir}）")
    return {
        "tokens": tokens,
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
    }


def _sherpa_cli() -> list[str]:
    """保留给外部脚本可选使用。"""
    shared = Path(sys.executable).resolve().parent / "sherpa-onnx-cli"
    if shared.is_file():
        return [str(shared)]
    return ["sherpa-onnx-cli"]


def build_keywords_file(
    *,
    model_dir: Path,
    phrases: list[str],
    keywords_file: Path,
    keywords_threshold: float = 0.25,
    keywords_score: float = 2.0,
) -> Path:
    """把中文短语转成 Sherpa keywords.txt（ppinyin）。"""
    cleaned = [p.strip() for p in phrases if p.strip()]
    if not cleaned:
        raise ValueError("至少需要一个唤醒短语")

    keywords_file.parent.mkdir(parents=True, exist_ok=True)
    texts = []
    extras: list[list[str]] = []
    for phrase in cleaned:
        texts.append(phrase)
        extras.append([f":{keywords_score}", f"#{keywords_threshold}", f"@{phrase}"])

    try:
        from sherpa_onnx.utils import text2token
    except ImportError as exc:
        raise ImportError(
            "请先安装：pip install sherpa-onnx pypinyin sentencepiece"
        ) from exc

    tokens_path = str(model_dir / "tokens.txt")
    encoded = text2token(texts, tokens=tokens_path, tokens_type="ppinyin")
    with keywords_file.open("w", encoding="utf-8") as handle:
        for tokens, extra in zip(encoded, extras):
            handle.write(" ".join([*tokens, *extra]) + "\n")
    return keywords_file.resolve()


def _keywords_file_matches(
    path: Path,
    *,
    phrases: list[str],
    keywords_threshold: float,
    keywords_score: float,
) -> bool:
    """检查 keywords.txt 是否与当前 CLI 阈值/短语一致（避免沿用旧 #0.25 导致唤不醒）。"""
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return False
    cleaned = [p.strip() for p in phrases if p.strip()]
    if len(lines) != len(cleaned):
        return False
    for line, phrase in zip(lines, cleaned):
        if f"@{phrase}" not in line:
            return False
        m_th = re.search(r"#([\d.]+)", line)
        m_sc = re.search(r":([\d.]+)", line)
        if not m_th or not m_sc:
            return False
        if abs(float(m_th.group(1)) - float(keywords_threshold)) > 1e-6:
            return False
        if abs(float(m_sc.group(1)) - float(keywords_score)) > 1e-6:
            return False
    return True


def ensure_keywords_file(
    *,
    model_dir: str | Path,
    phrases: list[str],
    keywords_file: str | Path | None = None,
    keywords_threshold: float = 0.25,
    keywords_score: float = 2.0,
    force: bool = False,
) -> Path:
    """若 keywords 不存在、配置过期或 force，则从短语重新生成。"""
    resolved_dir = resolve_sherpa_kws_dir(model_dir)
    out = Path(keywords_file or DEFAULT_KEYWORDS_FILE).expanduser()
    if (
        out.is_file()
        and not force
        and _keywords_file_matches(
            out,
            phrases=phrases,
            keywords_threshold=keywords_threshold,
            keywords_score=keywords_score,
        )
    ):
        return out.resolve()
    return build_keywords_file(
        model_dir=resolved_dir,
        phrases=phrases,
        keywords_file=out,
        keywords_threshold=keywords_threshold,
        keywords_score=keywords_score,
    )


def create_keyword_spotter(
    *,
    model_dir: Path,
    keywords_file: Path,
    keywords_threshold: float = 0.25,
    keywords_score: float = 2.0,
    num_threads: int = 2,
    use_int8: bool = True,
) -> Any:
    """构造 sherpa_onnx.KeywordSpotter。"""
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise ImportError(
            "请先安装 Sherpa-ONNX：pip install sherpa-onnx pypinyin"
        ) from exc

    paths = resolve_transducer_paths(model_dir, use_int8=use_int8)
    if not keywords_file.is_file():
        raise FileNotFoundError(f"keywords 文件不存在：{keywords_file}")

    return sherpa_onnx.KeywordSpotter(
        tokens=str(paths["tokens"]),
        encoder=str(paths["encoder"]),
        decoder=str(paths["decoder"]),
        joiner=str(paths["joiner"]),
        keywords_file=str(keywords_file),
        num_threads=num_threads,
        keywords_score=keywords_score,
        keywords_threshold=keywords_threshold,
        provider="cpu",
    )
