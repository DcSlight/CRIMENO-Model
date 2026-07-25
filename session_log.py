# session_log.py
# Shared "current session" contract used by video_broadcaster.py, the vlm/groq/tracker
# workers, and eval/score_logs.py. Lets per-video logs land in their own
# logs/<business>/<worker>/ folder (versioned per replay) instead of one flat file that
# grows forever and mixes videos together.
#
# Every public function here is defensive: workers call these from hot paths that must
# never crash just because a log directory couldn't be created. On any failure, fall
# back to None (or the legacy flat-file path the caller already has).
import json
import re
import time
from pathlib import Path
from typing import Optional

LOGS_ROOT = Path(__file__).resolve().parent / "logs"
SESSION_FILE = LOGS_ROOT / "current_session.json"

_VERSION_RE = re.compile(r"_v(\d+)\.")


def slugify(name: str) -> str:
    """Sanitize a video path/URL into a safe folder name. Not guaranteed pretty for
    URLs (Path(...).stem on a URL just takes the last path segment minus extension) —
    only guaranteed non-empty and filesystem-safe."""
    stem = Path(name).stem
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "video"


def begin_session(video_path: str) -> dict:
    """Called by the broadcaster whenever a video starts (auto-play or 'play' command).
    Picks the next version number for this business by scanning existing worker output
    folders for '_v<N>.' filenames, then persists the pointer file workers read from.
    Never raises — a broken logs/ folder must not block video playback."""
    business = slugify(video_path)
    business_dir = LOGS_ROOT / business

    version = 1
    try:
        max_seen = 0
        for worker in ("vlm", "groq", "tracker"):
            worker_dir = business_dir / worker
            worker_dir.mkdir(parents=True, exist_ok=True)
            for p in worker_dir.iterdir():
                m = _VERSION_RE.search(p.name)
                if m:
                    max_seen = max(max_seen, int(m.group(1)))
        version = max_seen + 1
    except Exception:
        # Treat as "no existing versions found" — version stays 1.
        version = 1

    session = {
        "business": business,
        "version": version,
        "started_at_unix_ms": int(time.time() * 1000),
    }

    try:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(session), encoding="utf-8")
    except Exception:
        pass

    return session


def read_session() -> Optional[dict]:
    """Read the current session pointer. Returns None if it doesn't exist yet (no video
    has ever been played) or is corrupt in any way — never raises."""
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_log_path(worker: str) -> Optional[Path]:
    """worker is one of "vlm", "groq", "tracker". Returns the versioned log path for
    the current session, or None if there's no session yet (caller should fall back to
    its legacy flat-file path)."""
    session = read_session()
    if session is None:
        return None
    try:
        business = session["business"]
        version = session["version"]
        path = LOGS_ROOT / business / worker / f"{worker}_v{version}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def latest_groq_log() -> Optional[Path]:
    """Used by eval/score_logs.py as the default --logs path. Finds the most recently
    written groq_v*.jsonl across all businesses (excludes groq_context_v*.txt — jsonl
    only). Returns None if logs/ doesn't exist or nothing matches."""
    try:
        candidates = list(LOGS_ROOT.glob("*/groq/groq_v*.jsonl"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None
