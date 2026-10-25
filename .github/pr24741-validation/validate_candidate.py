"""Candidate-only validation with independent runtime expectations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()
source, build, out = args.source.resolve(), args.build.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
report = {"status": "running", "commands": [], "runtime_cases": 0}
environment = dict(os.environ)
environment["PATH"] = f"{build}/tools:{build}/llvm-project/bin:" + environment["PATH"]
environment["TEST_TMPDIR"] = str(out / "lit-temp")


def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def run(name, command, timeout=600):
    command = [str(v) for v in command]
    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=timeout
    )
    (out / f"{name}.log").write_text(result.stdout + result.stderr)
    report["commands"].append(
        {"name": name, "command": command, "returncode": result.returncode}
    )
    save()
    if result.returncode:
        raise RuntimeError(
            f"{name}: exit {result.returncode}; see {out / (name + '.log')}"
        )
    return result.stdout


def order(values):
    return np.asarray(
        sorted(range(len(values)), key=lambda i: int(values[i])), dtype=np.int64
    )


def execute(name, artifact, function, inputs, expected, device):
    command = [
        build / "tools/iree-run-module",
        f"--module={artifact}",
        f"--device={device}",
        f"--function={function}",
    ]
    if device == "local-task":
        command.append("--task_topology_group_count=2")
    for i, array in enumerate(inputs):
        path = out / f"{name}-input-{i}.npy"
        np.save(path, array)
        command.append(f"--input=@{path}")
    paths = [out / f"{name}-output-{i}.npy" for i in range(len(expected))]
    for path in paths:
        path.unlink(missing_ok=True)
        command.append(f"--output=@{path}")
    run(name, command, timeout=120)
    for path, oracle in zip(paths, expected):
        actual = np.load(path, allow_pickle=False)
        if (
            actual.dtype != oracle.dtype
            or actual.shape != oracle.shape
            or not np.array_equal(actual, oracle)
        ):
            raise RuntimeError(f"{name}: {actual!r} != {oracle!r}")
    report["runtime_cases"] += 1
    save()


try:
    report["source"] = run("source", ["git", "-C", source, "rev-parse", "HEAD"]).strip()
    report["source_status"] = run(
        "source-status", ["git", "-C", source, "status", "--porcelain"]
    )
    if report["source_status"]:
        raise RuntimeError("candidate source is dirty")
    report["source_tree"] = run(
        "source-tree", ["git", "-C", source, "rev-parse", "HEAD^{tree}"]
    ).strip()
    report["llvm"] = run(
        "llvm-source",
        ["git", "-C", source / "third_party/llvm-project", "rev-parse", "HEAD"],
    ).strip()
    configured_source = f"CMAKE_HOME_DIRECTORY:INTERNAL={source}"
    if configured_source not in (build / "CMakeCache.txt").read_text().splitlines():
        raise RuntimeError("build is configured for a different source directory")
    run(
        "build-freshness",
        [
            "ninja",
            "-C",
            build,
            "iree-compile",
            "iree-opt",
            "iree-run-module",
            "iree-run-mlir",
            "FileCheck",
        ],
        timeout=1800,
    )
    report["compiler_sha256"] = hashlib.file_digest(
        open(build / "lib/libIREECompiler.so", "rb"), "sha256"
    ).hexdigest()
    report["compiler_version"] = run(
        "compiler-version", [build / "tools/iree-compile", "--version"]
    )
    fixtures = [
        "argsort_dynamic",
        "argsort_preserve_input_dynamic",
        "argsort_comparator_lanes_dynamic",
        "argsort_views_dynamic",
        "argsort_internal_key_use",
        "sort_workspace_versions",
    ]
    lit = source / "third_party/llvm-project/llvm/utils/lit/lit.py"
    compiler_tests = [
        source / "compiler/src/iree/compiler/DispatchCreation/test",
        source / "compiler/src/iree/compiler/Dialect/Flow/IR/test",
        source / "compiler/src/iree/compiler/Dialect/LinalgExt/IR/test",
        source / "compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test",
        source
        / "compiler/src/iree/compiler/Codegen/Common/test/iree_comprehensive_bufferize.mlir",
    ]
    run("compiler-lit", ["python3", lit, "-v", "-j2", *compiler_tests])
    run(
        "runtime-lit",
        [
            "python3",
            lit,
            "-v",
            "-j2",
            *[source / f"tests/e2e/regression/{name}.mlir" for name in fixtures],
        ],
    )

    cases = [
        np.asarray(x, dtype=np.int64)
        for x in (
            [3, 1, 2],
            [],
            [7],
            [2, 1, 2],
            [5, 4, 3, 2, 1],
            [0, -1, -(2**63), 2**63 - 1, 0],
            [((i * 37) % 23) - 11 for i in range(65)],
        )
    ]
    backends = [
        (
            "cpu",
            [
                "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=generic",
            ],
            ["local-sync", "local-task"],
        )
    ]
    if args.gpu:
        backends.append(
            (
                "cuda",
                ["--iree-hal-target-backends=cuda", "--iree-cuda-target=sm_120"],
                ["cuda"],
            )
        )
    for backend, flags, devices in backends:
        artifacts = {}
        for fixture in fixtures:
            artifact = out / f"{backend}-{fixture}.vmfb"
            run(
                f"{backend}-{fixture}-compile",
                [
                    build / "tools/iree-compile",
                    source / f"tests/e2e/regression/{fixture}.mlir",
                    *flags,
                    "--mlir-disable-threading",
                    "--mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize",
                    "-o",
                    artifact,
                ],
            )
            text = (out / f"{backend}-{fixture}-compile.log").read_text()
            if (
                "IR Dump After IREEComprehensiveBufferizePass" not in text
                or "memref.alloca" in text
                or "memref.alloc(" in text
            ):
                raise RuntimeError(
                    f"{backend}-{fixture}: bufferization evidence missing or private allocation present"
                )
            artifacts[fixture] = artifact
        for device in devices:
            for i, values in enumerate(cases):
                indices = order(values)
                prefix = f"{device}-{i}"
                execute(
                    prefix + "-argsort",
                    artifacts["argsort_dynamic"],
                    "argsort",
                    [values],
                    [indices],
                    device,
                )
                execute(
                    prefix + "-preserve",
                    artifacts["argsort_preserve_input_dynamic"],
                    "argsort_preserve_input",
                    [values],
                    [indices, values, values.reshape(1, -1)],
                    device,
                )
                execute(
                    prefix + "-shared",
                    artifacts["argsort_comparator_lanes_dynamic"],
                    "argsort_shared_keys",
                    [values],
                    [indices, values],
                    device,
                )
                execute(
                    prefix + "-both",
                    artifacts["argsort_comparator_lanes_dynamic"],
                    "sort_and_original",
                    [values],
                    [values[indices], values],
                    device,
                )
                weighted = sum(
                    (j + 1) * int(v) for j, v in enumerate(values[indices])
                ) % (2**64)
                weighted = np.asarray(weighted, dtype=np.uint64).view(np.int64)
                execute(
                    prefix + "-internal",
                    artifacts["argsort_internal_key_use"],
                    "argsort_internal_key_use",
                    [values],
                    [indices, weighted, values],
                    device,
                )
                execute(
                    prefix + "-consumer",
                    artifacts["argsort_internal_key_use"],
                    "argsort_internal_consumer",
                    [values],
                    [indices + values, values],
                    device,
                )
                execute(
                    prefix + "-view",
                    artifacts["argsort_views_dynamic"],
                    "argsort_collapsed_keys",
                    [values.reshape(1, -1)],
                    [indices, values.reshape(1, -1)],
                    device,
                )
            keys = np.asarray([2, 1], dtype=np.int64)
            labels = np.asarray([7, -3], dtype=np.int64)
            execute(
                device + "-labels",
                artifacts["argsort_comparator_lanes_dynamic"],
                "sort_carried_labels",
                [keys, labels],
                [np.asarray([-3, 7], dtype=np.int64), keys, labels],
                device,
            )
            floats = np.asarray([3.5, 1.5, 2.5], dtype=np.float32)
            execute(
                device + "-float",
                artifacts["argsort_comparator_lanes_dynamic"],
                "argsort_trailing_float_keys",
                [floats],
                [np.asarray([1, 2, 0], dtype=np.int64)],
                device,
            )
            for i, rows in enumerate(
                [
                    np.asarray([[3, 1, 2], [2, 1, 2]], dtype=np.int64),
                    np.empty((2, 0), dtype=np.int64),
                    np.empty((0, 3), dtype=np.int64),
                    np.asarray([[7], [-3]], dtype=np.int64),
                ]
            ):
                indices = np.asarray(
                    [order(row) for row in rows], dtype=np.int64
                ).reshape(rows.shape)
                execute(
                    f"{device}-rows-{i}",
                    artifacts["argsort_views_dynamic"],
                    "argsort_rows",
                    [rows],
                    [indices, rows],
                    device,
                )
            keys = np.asarray([4, 1, 3, 2], dtype=np.int64)
            labels = np.asarray([7, -3, 9, 11], dtype=np.int64)
            execute(
                device + "-versions",
                artifacts["sort_workspace_versions"],
                "dependent_observed_dynamic",
                [keys, labels],
                [
                    np.asarray([7, 9, 11, -3], dtype=np.int64),
                    np.asarray([1, 2, 3, 4], dtype=np.int64),
                    keys,
                    labels,
                ],
                device,
            )
    report["status"] = "pass"
    save()
    print(
        json.dumps(
            {
                "status": "pass",
                "source": report["source"],
                "runtime_cases": report["runtime_cases"],
            }
        )
    )
except BaseException as exc:
    report["status"] = "fail"
    report["error"] = repr(exc)
    save()
    raise
