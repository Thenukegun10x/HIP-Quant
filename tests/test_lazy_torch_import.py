from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_archive_import_does_not_eagerly_import_torch():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["HIP_VISIBLE_DEVICES"] = ""
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", "import hq2, sys; assert 'torch' not in sys.modules; print(hq2.HQ2_FORMAT.name)"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "HQ2"
