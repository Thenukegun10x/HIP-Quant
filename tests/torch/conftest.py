"""Pytest cleanup for ROCm torch extension tests.

On Windows ROCm, PyTorch/HIP runtime state can keep the Python process alive
after pytest has already printed the final summary.  Force-exit after pytest has
finished so commands like `python -m pytest tests\torch\test_fp8.py -q` return
to the shell with the correct exit code.
"""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_torch_api_extension_cache():
    """Undo CPU mock injection from tests/test_pipeline.py before GPU tests."""
    try:
        import hip_quant.torch_api as torch_api

        torch_api._C = None
    except Exception:
        pass
    yield


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("HIP_QUANT_PYTEST_NO_FORCE_EXIT", "").lower() in {"1", "true", "yes", "on"}:
        return

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exitstatus))
