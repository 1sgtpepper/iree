"""Build three wrong variants, require their predicted failures, then restore."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("source"))
parser.add_argument("--build", type=Path, default=Path("build"))
parser.add_argument("--out", type=Path, default=Path("evidence/control"))
args = parser.parse_args()
source, build, out = args.source.resolve(), args.build.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
converter = (
    source
    / "compiler/src/iree/compiler/DispatchCreation/ConvertDispatchRegionsToWorkgroups.cpp"
)
tiling = (
    source / "compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp"
)
original = {path: path.read_bytes() for path in [converter, tiling]}
report = {"status": "running", "commands": [], "compiler_hashes": {}}
base = "265aef8e4e1d50d582de92ba065a2991c16651db"


def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def run(name, command, check=True):
    command = [str(arg) for arg in command]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    (out / f"{name}.log").write_text(proc.stdout + proc.stderr)
    report["commands"].append(
        {"name": name, "command": command, "returncode": proc.returncode}
    )
    save()
    if check and proc.returncode:
        raise RuntimeError(f"{name} failed with exit {proc.returncode}")
    return proc


def rebuild(name):
    run(
        name, ["ninja", "-C", build, "-j2", "iree-compile", "iree-opt", "iree-run-mlir"]
    )
    report["compiler_hashes"][name] = hashlib.file_digest(
        open(build / "lib/libIREECompiler.so", "rb"), "sha256"
    ).hexdigest()
    save()


def baseline(path):
    return run(
        "base-" + path.stem,
        ["git", "-C", source, "show", f"{base}:{path.relative_to(source)}"],
    ).stdout


flags = ["--iree-hal-target-backends=llvm-cpu", "--iree-llvmcpu-target-cpu=generic"]
try:
    report["source"] = run(
        "source", ["git", "-C", source, "rev-parse", "HEAD"]
    ).stdout.strip()
    if run("source-status", ["git", "-C", source, "status", "--porcelain"]).stdout:
        raise RuntimeError("source must be clean before controls")
    converter.write_text(baseline(converter))
    rebuild("build-no-workspaces")
    proc = run(
        "compile-no-workspaces",
        [
            build / "tools/iree-compile",
            source / "tests/e2e/regression/argsort_dynamic.mlir",
            *flags,
            "-o",
            out / "no-workspaces.vmfb",
        ],
        check=False,
    )
    if proc.returncode == 0 or "stack allocation" not in proc.stderr.lower():
        raise RuntimeError(
            "baseline did not fail for the expected stack allocation reason"
        )
    # Retain matching-compiler static baseline artifacts for the local cost probe.
    for size in [4, 64, 256]:
        fixture = out.parent / "cost-probes" / f"argsort_static_{size}.mlir"
        if fixture.exists():
            run(
                f"compile-static-control-{size}",
                [
                    build / "tools/iree-compile",
                    fixture,
                    *flags,
                    "-o",
                    fixture.with_name(f"control-{size}.vmfb"),
                ],
            )

    converter.write_bytes(original[converter])
    tiling.write_text(baseline(tiling))
    rebuild("build-no-tiling-guard")
    artifact = out / "partial-sort.vmfb"
    run(
        "compile-partial-sort",
        [
            build / "tools/iree-compile",
            source / "tests/e2e/regression/argsort_internal_key_use.mlir",
            *flags,
            "-o",
            artifact,
        ],
    )
    paths = [out / f"partial-sort-{i}.npy" for i in range(3)]
    run(
        "execute-partial-sort",
        [
            build / "tools/iree-run-module",
            f"--module={artifact}",
            "--device=local-sync",
            "--function=argsort_internal_key_use",
            "--input=3xi64=[3 1 2]",
            *[f"--output=@{path}" for path in paths],
        ],
    )
    actual = [np.load(path, allow_pickle=False) for path in paths]
    predicted = [
        np.asarray([1, 0, 2], dtype=np.int64),
        np.asarray(13, dtype=np.int64),
        np.asarray([3, 1, 2], dtype=np.int64),
    ]
    if any(
        a.dtype != p.dtype or a.shape != p.shape or not np.array_equal(a, p)
        for a, p in zip(actual, predicted)
    ):
        raise RuntimeError(
            "unguarded tiling did not produce the predicted partial-sort result"
        )
    report["partial_sort_actual"] = [array.tolist() for array in actual]
    save()

    tiling.write_bytes(original[tiling])
    text = original[converter].decode()
    old = "for (auto sortOp : llvm::reverse(sortOps))"
    if text.count(old) != 1:
        raise RuntimeError("forward-order control no longer matches the candidate")
    converter.write_text(text.replace(old, "for (auto sortOp : sortOps)"))
    rebuild("build-forward-order")
    proc = run(
        "compile-forward-order",
        [
            build / "tools/iree-compile",
            source / "tests/e2e/regression/sort_workspace_versions.mlir",
            *flags,
            "-o",
            out / "forward-order.vmfb",
        ],
        check=False,
    )
    if (
        proc.returncode == 0
        or "dynamic dispatch dimensions not properly captured" not in proc.stderr
    ):
        raise RuntimeError(
            "forward-order control did not fail for disconnected workspace bindings"
        )
    report["status"] = "pass"
except BaseException as exc:
    report["status"] = "fail"
    report["error"] = repr(exc)
    raise
finally:
    for path, content in original.items():
        path.write_bytes(content)
    try:
        rebuild("restore-build")
        if run(
            "restored-source-status", ["git", "-C", source, "status", "--porcelain"]
        ).stdout:
            raise RuntimeError("source not clean after restoration")
    except BaseException as exc:
        report["status"] = "fail"
        report["restore_error"] = repr(exc)
        raise
    finally:
        save()
print(
    json.dumps({"status": report["status"], "source": report["source"], "controls": 3})
)
