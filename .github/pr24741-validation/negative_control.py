"""Rebuild the converter without the fix, require its predicted failure, restore."""

import hashlib
import json
from pathlib import Path
import subprocess


source = Path("source").resolve()
build = Path("build").resolve()
out = Path("evidence/control").resolve()
out.mkdir(parents=True, exist_ok=True)
path = (
    source
    / "compiler/src/iree/compiler/DispatchCreation/ConvertDispatchRegionsToWorkgroups.cpp"
)
original = path.read_bytes()
results = []


def run(name, command, check=True):
    command = [str(arg) for arg in command]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    (out / f"{name}.log").write_text(proc.stdout + proc.stderr)
    results.append({"name": name, "command": command, "returncode": proc.returncode})
    (out / "commands.json").write_text(json.dumps(results, indent=2) + "\n")
    if check and proc.returncode:
        raise RuntimeError(f"{name} failed with exit {proc.returncode}")
    return proc


try:
    baseline = run(
        "baseline-source",
        [
            "git",
            "-C",
            source,
            "show",
            "265aef8e4e1d50d582de92ba065a2991c16651db:compiler/src/iree/compiler/DispatchCreation/ConvertDispatchRegionsToWorkgroups.cpp",
        ],
    )
    path.write_text(baseline.stdout)
    run("build-control", ["ninja", "-C", build, "-j2", "iree-compile"])
    (out / "compiler-sha256.txt").write_text(
        hashlib.file_digest(
            open(build / "lib/libIREECompiler.so", "rb"), "sha256"
        ).hexdigest()
        + "\n"
    )
    proc = run(
        "compile-regression",
        [
            build / "tools/iree-compile",
            source / "tests/e2e/regression/argsort_dynamic.mlir",
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
            "-o",
            out / "regression.vmfb",
        ],
        check=False,
    )
    if proc.returncode == 0 or "stack allocation" not in proc.stderr.lower():
        raise RuntimeError(
            "control did not fail for the expected stack allocation reason"
        )
finally:
    path.write_bytes(original)
    run(
        "restore-build",
        ["ninja", "-C", build, "-j2", "iree-compile", "iree-opt", "iree-run-mlir"],
    )
    status = run(
        "restored-source-status", ["git", "-C", source, "status", "--porcelain"]
    )
    if status.stdout:
        raise RuntimeError("source not clean after restoration")
