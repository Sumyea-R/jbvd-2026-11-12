"""Regenerate public aggregate outputs; never invoke inference or private data."""
from pathlib import Path
import json
import subprocess
import sys

from verify_results import check_results

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = check_results()
    output = ROOT / "build"
    output.mkdir(exist_ok=True)
    (output / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    for script in ("generate_tables.py", "generate_figures.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)
    print("Regenerated outputs in build/. No private records or model calls were used.")


if __name__ == "__main__":
    main()
