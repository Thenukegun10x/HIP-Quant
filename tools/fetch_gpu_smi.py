"""Fetch / update the bundled ``gpu-smi`` binary from its GitHub Releases.

``gpu-smi`` is a separate project (https://github.com/Thenukegun10x/GPU-SMI),
not part of this source tree.  This script downloads the platform release asset
into ``tools/`` so it can be bundled in the wheel, and is a no-op when the
bundled binary already matches the target release.

Called automatically by ``setup.py`` (wheel build) and ``build.ps1``.  Every
build therefore checks for updates, but only downloads when the asset differs.

Environment:
    HIP_QUANT_SKIP_GPU_SMI=1     skip entirely (offline builds)
    HIP_QUANT_GPU_SMI_VERSION=v1.3.0   pin a release tag (default: latest)
    HIP_QUANT_GPU_SMI_REPO=owner/repo  override the source repository
    HIP_QUANT_GPU_SMI_TOKEN=...        GitHub token for private/rate-limited API

Exit codes: 0 = bundled/up-to-date/skipped, 1 = fetch failed (callers may warn).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "Thenukegun10x/GPU-SMI"
_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
_MARKER = _TOOLS / ".gpu-smi-version"
_TIMEOUT = 60


def _log(msg: str) -> None:
    print(f"[gpu-smi] {msg}")


def _asset_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "gpu-smi.exe"
    return "gpu-smi"


def _select_asset(assets: list[dict]) -> dict | None:
    system = platform.system().lower()
    exts = (".exe",) if system == "windows" else ("",)
    def matches(a):
        n = a.get("name", "").lower()
        if system == "windows":
            return n.endswith(".exe") and ("windows" in n or "win" in n)
        return ("linux" in n) and not n.endswith((".exe", ".sha256", ".txt"))
    for a in assets:
        if matches(a):
            return a
    # Permissive fallback: first executable-looking asset for this platform.
    for a in assets:
        if a.get("name", "").lower().endswith(exts):
            return a
    return None


def _api(url: str) -> dict:
    headers = {"User-Agent": "hip-quant-build", "Accept": "application/vnd.github+json"}
    token = os.environ.get("HIP_QUANT_GPU_SMI_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if os.environ.get("HIP_QUANT_SKIP_GPU_SMI", "").lower() in ("1", "true", "yes", "on"):
        _log("skipped (HIP_QUANT_SKIP_GPU_SMI set)")
        return 0

    repo = os.environ.get("HIP_QUANT_GPU_SMI_REPO") or DEFAULT_REPO
    version = os.environ.get("HIP_QUANT_GPU_SMI_VERSION")
    target = _TOOLS / _asset_name()

    try:
        if version:
            release = _api(f"https://api.github.com/repos/{repo}/releases/tags/{version}")
        else:
            release = _api(f"https://api.github.com/repos/{repo}/releases/latest")
    except urllib.error.HTTPError as exc:
        _log(f"release lookup failed for {repo} ({exc.code}); leaving existing binary")
        return 1 if not target.exists() else 0
    except Exception as exc:  # offline, DNS, rate limit, ...
        _log(f"release lookup failed ({exc}); leaving existing binary")
        return 1 if not target.exists() else 0

    tag = release.get("tag_name", "?")
    asset = _select_asset(release.get("assets", []))
    if asset is None:
        _log(f"release {tag} has no asset for this platform; leaving existing binary")
        return 1 if not target.exists() else 0

    expected = ""
    digest = asset.get("digest") or ""
    if digest.startswith("sha256:"):
        expected = digest.split(":", 1)[1].lower()

    if target.exists() and expected and _sha256(target) == expected:
        _log(f"up to date ({tag}, sha256 {expected[:12]}...)")
        _MARKER.parent.mkdir(parents=True, exist_ok=True)
        _MARKER.write_text(tag + "\n", encoding="utf-8")
        return 0

    _TOOLS.mkdir(parents=True, exist_ok=True)
    _log(f"downloading {asset['name']} ({tag}) ...")
    fd, tmp_name = tempfile.mkstemp(suffix=".exe" if platform.system() == "Windows" else "", dir=str(_TOOLS))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(
            asset["browser_download_url"], headers={"User-Agent": "hip-quant-build"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if expected:
            got = _sha256(tmp)
            if got != expected:
                _log(f"checksum mismatch (expected {expected}, got {got}); discarding")
                return 1
        shutil.move(str(tmp), str(target))
    except Exception as exc:
        _log(f"download failed ({exc}); leaving existing binary")
        return 1
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    _MARKER.parent.mkdir(parents=True, exist_ok=True)
    _MARKER.write_text(tag + "\n", encoding="utf-8")
    _log(f"bundled {target.name} from {tag} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
