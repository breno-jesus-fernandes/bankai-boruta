"""RSS retention guard in an isolated process, complementing Rust owner counters."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_repeated_arrow_fits_and_errors_do_not_retain_value_buffers():
    pytest.importorskip("polars")
    pytest.importorskip("psutil")
    script = Path(__file__).parents[1] / "benchmarks" / "repeated_fit_memory.py"
    result = subprocess.run([sys.executable, str(script)], check=True, text=True,
                            capture_output=True, timeout=120)
    report = json.loads(result.stdout)
    assert report["growth_bytes"] < report["allowed_growth_bytes"], report
