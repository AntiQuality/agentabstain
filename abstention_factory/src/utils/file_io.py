from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def write_yaml(path: str | Path, payload: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    # Atomic write: serialize to a sibling temp file, fsync, then
    # os.replace onto the target. Without this, two concurrent writers
    # (e.g. overlapping eval runs hitting the same eval.json) can each
    # truncate and write at different lengths, leaving the longer
    # writer's tail bytes appended to the shorter writer's payload --
    # producing files that json.loads rejects with `Extra data`.
    target = Path(path)
    ensure_dir(target.parent)
    text = json.dumps(payload, indent=4, sort_keys=False, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
