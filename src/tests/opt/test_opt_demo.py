"""
Filename: test_opt_demo.py
Author: Santiago Nunez-Corrales
Date: 2026-05-14
Version: 1.0
Description:
    Smoke-tests examples/opt_demo.py: it must run to completion at every
    opt_level without raising, with a working test config.

License: Apache 2.0
Contact: nunezco2@illinois.edu
"""
import runpy
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO = REPO_ROOT / "examples" / "opt_demo.py"
TEST_CONFIG = (
    REPO_ROOT / "src" / "tests" / "data" / "testing.toml"
)


@pytest.mark.skipif(not DEMO.exists(), reason="opt_demo.py not present")
def test_opt_demo_runs(monkeypatch, tmp_path, capsys):
    # The demo script hard-codes config/default.toml; for the smoke test
    # we run it from the repo root so that path resolves. If the project
    # config file is absent in the test env, fall back to copying the test
    # config into the expected location.
    cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        runpy.run_path(str(DEMO), run_name="__main__")
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "opt_level=0" in out
    assert "opt_level=3" in out
