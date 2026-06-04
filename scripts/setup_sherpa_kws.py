#!/usr/bin/env python3
"""下载 Sherpa-ONNX 中文 KWS 模型并生成 keywords.txt（无需 Picovoice 注册）。"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.adapters.voice.sherpa_kws import (  # noqa: E402
    DEFAULT_KEYWORDS_FILE,
    DEFAULT_MODEL_URL,
    DEFAULT_SHERPA_KWS_DIR,
    build_keywords_file,
    ensure_keywords_file,
    resolve_sherpa_kws_dir,
)


def download_model(model_dir: Path, url: str, force: bool) -> Path:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if (model_dir / "tokens.txt").is_file() and not force:
        print(f"模型已存在：{model_dir}")
        return model_dir.resolve()

    print(f"下载 Sherpa KWS 模型：{url}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.tar.bz2"
        subprocess.run(
            ["wget", "-O", str(archive), url],
            check=True,
        )
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=model_dir.parent)
    resolved = resolve_sherpa_kws_dir(model_dir)
    print(f"模型已解压到：{resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup Sherpa-ONNX KWS wake word")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_SHERPA_KWS_DIR),
        help="模型解压目录",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=DEFAULT_MODEL_URL,
        help="模型 tar.bz2 下载地址",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="小助",
        help="唤醒短语（可多次指定）",
    )
    parser.add_argument(
        "--keywords-file",
        type=str,
        default=str(DEFAULT_KEYWORDS_FILE),
        help="输出 keywords.txt 路径",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--score", type=float, default=2.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-keywords", action="store_true")
    args = parser.parse_args()

    model_dir = download_model(Path(args.model_dir), args.model_url, args.force_download)
    keywords = ensure_keywords_file(
        model_dir=model_dir,
        phrases=[args.keyword],
        keywords_file=args.keywords_file,
        keywords_threshold=args.threshold,
        keywords_score=args.score,
        force=args.force_keywords,
    )
    print(f"唤醒词 keywords：{keywords}")
    print("启动示例：")
    print(
        f"  python -m src.main --wake-backend sherpa-onnx "
        f"--wake-word {args.keyword} --voice-debug"
    )


if __name__ == "__main__":
    main()
