"""Copy a finished run to demo/run so the app can replay it without any LLM call.

Usage: python scripts/save_demo.py <run_id>
"""
import shutil
import sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("usage: python scripts/save_demo.py <run_id>")
src = Path("outputs") / sys.argv[1]
if not (src / "replay.json").exists():
    sys.exit(f"{src}/replay.json not found: run a generation from the app first")
dst = Path("demo") / "run"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)
print(f"Saved {src} -> {dst}. Commit demo/ so the replay works on any machine.")
