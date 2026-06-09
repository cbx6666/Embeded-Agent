from __future__ import annotations

"""扫描本地 data/music 目录，建立媒体库索引。"""

import logging
import re
from pathlib import Path

from src.agent.media.media_models import MediaLibraryIndex, MediaTrack

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"})

# 标准顶层类型目录（英文）
_TOP_LEVEL_TYPES = frozenset({"music", "xiangsheng", "opera"})

# 目录名（中/英）-> 规范化 (media_type, category)。
# 支持用户直接把「轻音乐/相声/摇滚」放在 data/music 下，不必套 music/light 两层。
_CANONICAL_FOLDERS: dict[str, tuple[str, str]] = {
    "music": ("music", "general"),
    "xiangsheng": ("xiangsheng", "general"),
    "opera": ("opera", "general"),
    "轻音乐": ("music", "light"),
    "light": ("music", "light"),
    "学习音乐": ("music", "study"),
    "study": ("music", "study"),
    "放松": ("music", "relaxing"),
    "relaxing": ("music", "relaxing"),
    "流行": ("music", "pop"),
    "pop": ("music", "pop"),
    "摇滚": ("music", "rock"),
    "rock": ("music", "rock"),
    "抒情": ("music", "relaxing"),
    "情歌": ("music", "pop"),
    "欢快": ("music", "pop"),
    "外语": ("music", "foreign"),
    "foreign": ("music", "foreign"),
    "相声": ("xiangsheng", "short"),
    "short": ("xiangsheng", "short"),
    "京剧": ("opera", "jingju"),
    "jingju": ("opera", "jingju"),
}


def scan_media_library(root: str | Path = "data/music") -> MediaLibraryIndex:
    """扫描媒体目录；目录为空或不存在时优雅返回空索引。"""

    root_path = Path(root)
    if not root_path.is_dir():
        logger.info("[媒体库] 目录不存在或不可读：%s，返回空索引", root_path)
        return MediaLibraryIndex(root=str(root_path))

    tracks: list[MediaTrack] = []
    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue
        track = _infer_track(file_path, root_path)
        if track is not None:
            tracks.append(track)

    media_types = sorted({t.media_type for t in tracks})
    categories_by_type: dict[str, set[str]] = {}
    for track in tracks:
        categories_by_type.setdefault(track.media_type, set()).add(track.category)

    index = MediaLibraryIndex(
        root=str(root_path.resolve()),
        tracks=tracks,
        media_types=media_types,
        categories_by_type={k: sorted(v) for k, v in categories_by_type.items()},
    )
    logger.info("[媒体库] 扫描完成：%d 个文件，类型=%s", index.count, media_types)
    return index


def _infer_track(file_path: Path, root: Path) -> MediaTrack | None:
    """从目录结构推断 media_type 与 category。

    示例：
    - data/music/music/light/a.mp3 -> music / light
    - data/music/xiangsheng/short/b.mp3 -> xiangsheng / short
    - data/music/opera/jingju/c.mp3 -> opera / jingju
    """

    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return None

    raw_parts = list(rel.parts[:-1])
    parts_lower = [p.lower() for p in raw_parts]
    if not raw_parts:
        media_type = "unknown"
        category = "general"
    elif parts_lower[0] in _TOP_LEVEL_TYPES:
        media_type = parts_lower[0]
        sub = parts_lower[1] if len(parts_lower) > 1 else "general"
        _, category = _CANONICAL_FOLDERS.get(sub, (media_type, sub))
    else:
        folder_key = raw_parts[0]
        media_type, category = _CANONICAL_FOLDERS.get(
            folder_key,
            _CANONICAL_FOLDERS.get(parts_lower[0], (parts_lower[0], "general")),
        )

    track_id = _make_track_id(rel)
    title = file_path.stem
    tags = list(raw_parts)
    return MediaTrack(
        id=track_id,
        title=title,
        path=str(file_path.resolve()),
        media_type=media_type,
        category=category,
        tags=tags,
    )


def _make_track_id(rel: Path) -> str:
    raw = str(rel).replace("\\", "/")
    slug = re.sub(r"[^a-zA-Z0-9/_\-.]", "_", raw)
    return slug.lower()


def build_library_catalog(index: MediaLibraryIndex) -> dict[str, object]:
    """把本地曲库整理成 LLM 可读的曲目清单（id / 标题 / 文件夹 / 类型）。"""

    tracks: list[dict[str, str]] = []
    folders: set[str] = set()
    for track in index.tracks:
        folder = track.tags[0] if track.tags else track.category
        folders.add(folder)
        tracks.append(
            {
                "id": track.id,
                "title": track.title,
                "folder": folder,
                "media_type": track.media_type,
            }
        )
    return {
        "tracks": tracks,
        "folders": sorted(folders),
        "total": len(tracks),
    }
