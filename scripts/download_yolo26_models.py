#!/usr/bin/env python3
"""从 Ultralytics 官方 GitHub Assets 下载 YOLO26 权重。

官方发布页: https://github.com/ultralytics/assets/releases
默认 release 与 ultralytics 8.4+ 一致: v8.4.0

用法（项目根目录）:
  python scripts/download_yolo26_models.py
  python scripts/download_yolo26_models.py --models yolo26n.pt yolo26n-pose.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models" / "yolo26"
# 与 ultralytics.utils.downloads.attempt_download_asset 默认 release 一致
OFFICIAL_RELEASE = "v8.4.0"
ASSETS_BASE = f"https://github.com/ultralytics/assets/releases/download/{OFFICIAL_RELEASE}"


def _download_one(name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name
    if dest.is_file() and dest.stat().st_size > 100_000:
        print(f"已存在，跳过: {dest} ({dest.stat().st_size // 1024} KB)")
        return dest

    url = f"{ASSETS_BASE}/{name}"
    print(f"下载: {url}")
    try:
        from ultralytics.utils.downloads import safe_download

        safe_download(url=url, file=dest, progress=True)
    except ImportError:
        import urllib.request

        urllib.request.urlretrieve(url, dest)
    if not dest.is_file() or dest.stat().st_size < 100_000:
        raise RuntimeError(f"下载失败或文件过小: {dest}")
    print(f"完成: {dest} ({dest.stat().st_size // 1024} KB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Ultralytics YOLO26 官方权重")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"保存目录（默认 {DEFAULT_OUT.relative_to(ROOT)}）",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo26n.pt", "yolo26n-pose.pt"],
        help="要下载的 .pt 文件名",
    )
    parser.add_argument(
        "--via-yolo",
        action="store_true",
        help="用 ultralytics YOLO() 触发自动下载（会缓存到用户目录，并复制到 --out-dir）",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir.resolve()

    if args.via_yolo:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("缺少 ultralytics，请: pip install 'ultralytics>=8.3.0'", file=sys.stderr)
            sys.exit(1)
        for name in args.models:
            print(f"通过 YOLO() 拉取: {name}")
            model = YOLO(name)
            src = Path(getattr(model, "ckpt_path", None) or model.model.pt_path)
            if not src.is_file():
                src = Path(name)
            dest = out_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dest.resolve():
                import shutil

                shutil.copy2(src, dest)
            print(f"已保存到: {dest}")
        return

    for name in args.models:
        _download_one(name, out_dir)
    print(f"\n全部完成。模型目录: {out_dir}")


if __name__ == "__main__":
    main()
