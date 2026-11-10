"""Install a Hub environment into its own virtual environment, and pick the verifiers version it was written for."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rewardaudit.errors import AuditError, short_error

CACHE = Path(os.environ.get("REWARDAUDIT_HOME", Path.home() / ".cache" / "rewardaudit"))
HUB = "https://hub.primeintellect.ai"
V0_PIN = "verifiers>=0.1.2,<0.2"  # last line of the Environment/Rubric/Parser API
# verifiers 0.1.15.dev152 to .dev189 export a taskset/harness API at the top level (`vf.Taskset[vf.TasksetConfig]`,
# `load_environment(config: vf.EnvConfig)` returning `vf.Env(taskset, harness)`); 0.1.14 has none of it and
# later releases moved it into verifiers.v1 with another signature, so these packages need that window.
TASKSET_PIN = "verifiers>=0.1.15.dev152,<0.1.15.dev190"

_DETECT = r"""
import importlib, importlib.util, importlib.metadata as m, json, re, sys, pathlib
name = sys.argv[1]
out = {"verifiers": m.version("verifiers"), "api": "unknown"}
variants = [name, name.replace("-", "_"), name.replace("_", "-"), name.lower(), name.lower().replace("-", "_")]
cands = list(dict.fromkeys(variants))
for dn in dict.fromkeys(variants):  # a distribution whose module is named differently
    try:
        dist = m.distribution(dn)
    except m.PackageNotFoundError:
        continue
    mods = sorted({str(f).split("/")[0].removesuffix(".py") for f in dist.files or []
                   if str(f).endswith(".py") and ".dist-info" not in str(f) and "__pycache__" not in str(f)})
    out.update(dist=dist.metadata["Name"], verifiers_req=[r for r in dist.requires or [] if re.match(r"verifiers\b", r)])
    cands += [x for x in mods if x not in cands]
    break
for cand in cands:
    spec = importlib.util.find_spec(cand)
    if spec is None or not spec.origin:
        continue
    out["module"] = cand
    src = pathlib.Path(spec.origin).read_text(encoding="utf-8", errors="replace")
    if "verifiers.v1" not in src and re.search(r"\bvf\.(?:Taskset|EnvConfig|load_harness)\b", src):
        out["api"] = "v0_taskset"  # top-level taskset/harness API of the verifiers 0.1.15 pre-releases
    elif re.search(r"^\s*def load_environment\b", src, flags=re.M) or "SingleTurnEnv" in src or "vf.Rubric" in src:
        out["api"] = "v0"
    elif "verifiers.v1" in src or "Taskset" in src:
        out["api"] = "v1"
    break
print(json.dumps(out))
"""


def env_dir(env: str) -> Path:
    """A venv is reused once installed, so a version asked for gets one of its own: `owner/name@1.2.0` installed into the
    venv of `owner/name` would audit whatever that one holds."""
    owner, _, rest = env.partition("/")
    name, _, version = rest.partition("@")
    return CACHE / "envs" / (f"{owner}__{name}" + (f"@{version}" if version and version != "latest" else ""))


def detect_api(py: Path, name: str, timeout_s: float = 300.0) -> dict:
    try:
        proc = subprocess.run([str(py), "-c", _DETECT, name], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"api": "unknown", "error": f"API detection timed out after {timeout_s:.0f}s"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"api": "unknown", "error": proc.stderr[-500:]}


def _version_key(v: str) -> tuple:
    """Order PEP 440 versions well enough to compare verifiers bounds (release, pre, post, dev)."""
    m = re.match(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?(?:\.dev(\d+))?", v)
    if not m:
        return ((-1,), (-2, 0), -1, -1)
    rel = tuple(int(x) for x in m.group(1).split("."))
    rel += (0,) * (6 - len(rel))
    if m.group(2):
        pre = ({"a": 0, "b": 1, "rc": 2}[m.group(2)], int(m.group(3)))
    else:
        pre = (-1, 0) if m.group(5) and not m.group(4) else (3, 0)  # 1.0.dev1 < 1.0a1 < 1.0
    return rel, pre, int(m.group(4) or -1), int(m.group(5)) if m.group(5) else float("inf")


def _names_own_verifiers(info: dict) -> bool:
    return any(" @ " in r for r in info.get("verifiers_req") or [])


def declared_min_verifiers(reqs: list[str]) -> str | None:
    """The lower bound a package puts on verifiers (`verifiers[openenv]>=0.3.1` -> '0.3.1', `verifiers==0.3.0` -> '0.3.0'),
    ignoring requirements behind an extra."""
    for r in reqs:
        if "extra ==" in r:
            continue
        m = re.search(r"(?:>=|==|~=)\s*([0-9][\w.]*)", r.split(";")[0])
        if m:
            return m.group(1)
    return None


def _in_taskset_window(version: str) -> bool:
    m = re.fullmatch(r"0\.1\.15\.dev(\d+)", version or "")
    return bool(m) and 152 <= int(m.group(1)) < 190


def _uv() -> str:
    uv = shutil.which("uv") or shutil.which("uv", path=str(Path(sys.executable).parent))
    if not uv:
        raise AuditError("Installing a Hub environment needs uv: pip install uv")
    return uv


def _run(cmd: list[str], what: str, timeout_s: float | None) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise AuditError(f"{what} timed out after {timeout_s:.0f}s") from None
    if proc.returncode != 0:
        raise AuditError(f"{what} failed: {short_error(proc.stderr)}")


def _pin_for(info: dict) -> str | None:
    """The verifiers this package's API needs, when what it got does not have that API."""
    floor = declared_min_verifiers(info.get("verifiers_req") or [])
    if _names_own_verifiers(info) or (info.get("api") == "v0" and floor and _version_key(floor) >= _version_key("0.2")):
        return None
    if info.get("api") == "v0" and not info.get("verifiers", "0").startswith("0.1."):
        return V0_PIN
    if info.get("api") == "v0_taskset" and not _in_taskset_window(info.get("verifiers", "")):
        return TASKSET_PIN
    return None


def install_env(env: str, *, verifiers: str | None = None, python_version: str = "3.12", log=None,
                timeout_s: float | None = 1800.0) -> tuple[Path, dict]:
    """The interpreter of a venv holding `env` (`owner/name[@version]`) and verifiers, and what detection learned: api,
    module, verifiers version. `verifiers` is a version constraint such as '<0.1.13' that overrides the choice made here."""
    log = log or (lambda s: None)
    owner, _, rest = env.partition("/")
    name, _, version = rest.partition("@")
    if not (owner and name):
        raise AuditError(f"Expected a Hub environment as owner/name, got {env!r}")
    uv, d = _uv(), env_dir(env)
    py = d / "bin" / "python"
    if not py.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        log(f"installing {env} into {d}")
        try:
            _run([uv, "venv", "--python", python_version, str(d)], f"Creating a venv for {env}", timeout_s)
            # `datasets` is what most tasksets load their rows with, yet not every package declares it
            _run([uv, "pip", "install", "--python", str(py), "verifiers", "datasets",
                  f"{name}=={version}" if version and version != "latest" else name,
                  "--extra-index-url", f"{HUB}/{owner}/{name}/install/simple/"], f"Installing {env}", timeout_s)
        except AuditError:
            shutil.rmtree(d, ignore_errors=True)
            raise
    info = detect_api(py, name)
    pin = f"verifiers{verifiers}" if verifiers and verifiers[0] in "<>=!~" else verifiers or _pin_for(info)
    if pin:
        log(f"installing {pin} for {env}")
        _run([uv, "pip", "install", "--python", str(py), pin], f"Installing {pin} for {env}", timeout_s)
        info["verifiers"] = pin
    return py, info
