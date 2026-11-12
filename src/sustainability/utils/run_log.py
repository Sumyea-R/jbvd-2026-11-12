import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_manifest_prefix(filename_prefix: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", filename_prefix).strip("._-")
    return prefix or "run"


def write_run_manifest(logs_dir: Path, filename_prefix: str, payload: Dict[str, Any]) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_path = logs_dir / f"{safe_manifest_prefix(filename_prefix)}_{utc_timestamp()}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return file_path
