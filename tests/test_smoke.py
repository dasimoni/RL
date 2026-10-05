"""Smoke tests: run every chapter script in --quick mode and check it exits cleanly.

Run with `pytest tests/` from the repository root. Each script must finish its
--quick run within TIMEOUT_S seconds and must not write figures in that mode.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 180


def _scripts():
    for path in sorted((ROOT / "code").glob("ch*/*.py")):
        if "__main__" in path.read_text(encoding="utf-8"):
            yield path


SCRIPTS = list(_scripts())


@pytest.mark.parametrize("script", SCRIPTS, ids=[str(p.relative_to(ROOT)) for p in SCRIPTS])
def test_quick_run(script):
    figures = script.parent / "figures"
    before = {p: p.stat().st_mtime_ns for p in figures.glob("*")} if figures.exists() else {}

    result = subprocess.run(
        [sys.executable, str(script.relative_to(ROOT)), "--quick"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, f"{script.name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"

    after = {p: p.stat().st_mtime_ns for p in figures.glob("*")} if figures.exists() else {}
    assert after == before, f"{script.name} wrote figures in --quick mode"
