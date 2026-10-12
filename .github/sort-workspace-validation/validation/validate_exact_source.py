# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fail-closed CPU validation of the exact source identity and its compiled tools."""
import argparse
import atexit
import hashlib
import json
import os
import uuid
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=root.parent / "source")
parser.add_argument("--build", type=Path, default=root / "build")
parser.add_argument("--out", type=Path, default=root / "evidence/exact-source")
parser.add_argument("--smoke-only", action="store_true")
args = parser.parse_args()
source = args.source.resolve()
build = args.build.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
report = {
    "validation_id": os.environ.get("SORT_VALIDATION_RUN_ID", str(uuid.uuid4())),
    "verdict": "running",
    "runs": [],
    "source": str(source),
    "build": str(build),
}
expected = json.loads((root / "validation/source-identity.json").read_text())


def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def unfinished():
    if report["verdict"] == "running":
        report["verdict"] = "fail"
        report["failure"] = "Validation terminated before completing required checks"
        save()


atexit.register(unfinished)


def require(ok, message):
    if not ok:
        report["verdict"] = "fail"
        report["failure"] = message
        save()
        raise SystemExit(message)


def git(*values):
    return subprocess.check_output(
        ["git", "-C", str(source), *values], text=True
    ).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(name, values, phase, expected_status=0):
    values = [str(v) for v in values]
    try:
        result = subprocess.run(values, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        require(False, name + ": timeout is inconclusive, not semantic evidence")
    log = out / (name + ".log")
    log.write_text(result.stdout + result.stderr)
    row = {
        "name": name,
        "command": values,
        "phase": phase,
        "exit_status": result.returncode,
        "expected_status": expected_status,
        "log": str(log),
    }
    report["runs"].append(row)
    save()
    require(
        result.returncode == expected_status,
        name + ": " + phase + " failed; see " + str(log),
    )
    return result


require(
    git("rev-parse", "HEAD") == expected["head"],
    "source HEAD differs from admitted identity",
)
require(
    git("rev-parse", "HEAD^{tree}") == expected["tree"],
    "source tree differs from admitted identity",
)
require(not git("status", "--porcelain"), "source checkout is dirty")
for name, revision in expected["submodules"].items():
    actual = subprocess.check_output(
        ["git", "-C", str(source / name), "rev-parse", "HEAD"], text=True
    ).strip()
    require(actual == revision, name + ": submodule identity differs")
    require(
        not subprocess.check_output(
            ["git", "-C", str(source / name), "status", "--porcelain"], text=True
        ).strip(),
        name + ": submodule checkout is dirty",
    )
report["identity"] = expected
compiler = build / "tools/iree-compile"
optimizer = build / "tools/iree-opt"
runtime = build / "tools/iree-run-module"
lit_runtime = build / "tools/iree-run-mlir"
for tool in (compiler, optimizer, runtime, lit_runtime):
    require(tool.is_file(), "missing built tool: " + str(tool))
    report.setdefault("tool_sha256", {})[str(tool)] = sha(tool)
report["compiler_library_sha256"] = sha(build / "lib/libIREECompiler.so")
report["compiler_version"] = command(
    "compiler-version", [compiler, "--version"], "identity"
).stdout
require(
    expected["head"][:12] in report["compiler_version"]
    or expected["head"] in report["compiler_version"],
    "compiler build metadata does not identify the admitted source HEAD",
)


def compile_fixture(name, relative, extra=()):
    input_path = source / relative
    artifact = out / (name + ".vmfb")
    result = command(
        name + "-compile",
        [
            compiler,
            input_path,
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
            "--mlir-disable-threading",
            "--mlir-print-ir-before=iree-dispatch-creation-convert-dispatch-regions-to-workgroups",
            "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-codegen-iree-comprehensive-bufferize",
            "--dump-compilation-phases-to=" + str(out / (name + "-phases")),
            *extra,
            "-o",
            artifact,
        ],
        "compile",
    )
    require(
        "IR Dump After IREEComprehensiveBufferizePass" in result.stderr,
        name + ": missing allocation evidence",
    )
    require(
        "memref.alloca" not in result.stderr and "memref.alloc(" not in result.stderr,
        name + ": kernel allocation violates the workspace contract",
    )
    report.setdefault("fixtures", {})[relative] = sha(input_path)
    report["runs"][-1]["allocation_check"] = "pass: no private kernel alloc/alloca"
    save()
    return artifact


def execute(name, artifact, function, inputs, expected_outputs, device="local-sync"):
    values = [
        runtime,
        "--module=" + str(artifact),
        "--device=" + device,
        "--function=" + function,
    ]
    if device == "local-task":
        values.append("--task_topology_group_count=2")
    for i, array in enumerate(inputs):
        path = out / f"{name}-input-{i}.npy"
        np.save(path, array)
        values.append("--input=@" + str(path))
    output_paths = []
    for i in range(len(expected_outputs)):
        path = out / f"{name}-output-{i}.npy"
        path.unlink(missing_ok=True)
        output_paths.append(path)
        values.append("--output=@" + str(path))
    command(name, values, "runtime")
    for i, (path, expected_array) in enumerate(zip(output_paths, expected_outputs)):
        require(path.is_file(), name + ": output missing")
        actual = np.load(path, allow_pickle=False)
        require(
            actual.shape == expected_array.shape
            and actual.dtype == expected_array.dtype
            and np.array_equal(actual, expected_array),
            f"{name}: incorrect output {i}: {actual.tolist()} != {expected_array.tolist()}",
        )
    report["runs"][-1][
        "oracle"
    ] = "pass: independent Python ordering, exact dtype/shape/value"
    save()


def indices(values):
    return np.array(sorted(range(values.size), key=values.__getitem__), dtype=np.int64)


# Original reproducer must compile and execute before any expanded test work.
original = compile_fixture("original", "tests/e2e/regression/argsort_dynamic.mlir")
first = np.array([3, 1, 2], dtype=np.int64)
execute("original-reproducer", original, "argsort", [first], [indices(first)])
if args.smoke_only:
    report["verdict"] = "pass"
    report["scope"] = "original reproducer and allocation check only"
    save()
    print(
        "PASS: exact-source original reproducer, independent output oracle, no kernel allocations"
    )
    sys.exit(0)

cases = [
    first,
    np.array([], dtype=np.int64),
    np.array([7], dtype=np.int64),
    np.array([2, 1, 2], dtype=np.int64),
    np.array([5, 4, 3, 2, 1], dtype=np.int64),
    np.array([0, -1, -(2**63), 2**63 - 1, 0], dtype=np.int64),
    np.array([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int64),
    np.array([((i * 37) % 23) - 11 for i in range(65)], dtype=np.int64),
]
for i, values in enumerate(cases[1:], 1):
    execute(f"original-{i}", original, "argsort", [values], [indices(values)])
# Challenge the very same binary with an incorrect expected result; it must reject it.
wrong = command(
    "wrong-result-control",
    [
        runtime,
        "--module=" + str(original),
        "--device=local-sync",
        "--function=argsort",
        "--input=3xi64=[3 1 2]",
        "--expected_output=3xi64=[0 1 2]",
    ],
    "oracle",
    expected_status=1,
)
require(
    "expected" in (wrong.stdout + wrong.stderr).lower()
    and "1 2 0" in (wrong.stdout + wrong.stderr),
    "wrong-result control failed for an unrelated reason",
)

preserve = compile_fixture(
    "preserve", "tests/e2e/regression/argsort_preserve_input_dynamic.mlir"
)
for i, values in enumerate(cases):
    execute(
        f"preserve-{i}",
        preserve,
        "argsort_preserve_input",
        [values],
        [indices(values), values, values.reshape(1, -1)],
    )
lanes = compile_fixture(
    "lanes", "tests/e2e/regression/argsort_comparator_lanes_dynamic.mlir"
)
for i, values in enumerate(cases):
    execute(
        f"shared-keys-{i}",
        lanes,
        "argsort_shared_keys",
        [values],
        [indices(values), values],
    )
for i, values in enumerate(
    [np.array([3.5, 1.5, 2.5], dtype=np.float32), np.array([2, 1, 2], dtype=np.float32)]
):
    execute(
        f"float-keys-{i}",
        lanes,
        "argsort_trailing_float_keys",
        [values],
        [indices(values)],
    )
for i, labels in enumerate(
    [np.array([1, 0], dtype=np.int64), np.array([7, -3], dtype=np.int64)]
):
    keys = np.array([2, 1], dtype=np.int64)
    execute(
        f"carried-labels-{i}",
        lanes,
        "sort_carried_labels",
        [keys, labels],
        [labels[indices(keys)], keys, labels],
    )
for i, values in enumerate(cases):
    execute(
        f"sorted-and-original-{i}",
        lanes,
        "sort_and_original",
        [values],
        [values[indices(values)], values],
    )
views = compile_fixture("views", "tests/e2e/regression/argsort_views_dynamic.mlir")
for i, values in enumerate(cases):
    execute(
        f"collapsed-{i}",
        views,
        "argsort_collapsed_keys",
        [values.reshape(1, -1)],
        [indices(values), values.reshape(1, -1)],
    )
rows = np.array([[3, 1, 2], [2, 1, 2]], dtype=np.int64)
row_indices = np.stack([indices(row) for row in rows])
execute("rows", views, "argsort_rows", [rows], [row_indices, rows])
distributed = compile_fixture(
    "distributed",
    "tests/e2e/regression/argsort_views_dynamic.mlir",
    ["--iree-llvmcpu-distribution-size=1"],
)
execute(
    "distributed-rows",
    distributed,
    "argsort_rows",
    [rows],
    [row_indices, rows],
    device="local-task",
)
for i, rows in enumerate(
    [
        np.empty((0, 3), dtype=np.int64),
        np.empty((2, 0), dtype=np.int64),
        np.array([[7], [-3]], dtype=np.int64),
        np.stack([cases[-1], cases[-1][::-1]]),
    ]
):
    row_indices = np.array([indices(row) for row in rows], dtype=np.int64).reshape(
        rows.shape
    )
    execute(f"row-boundary-{i}", views, "argsort_rows", [rows], [row_indices, rows])
    execute(
        f"distributed-row-boundary-{i}",
        distributed,
        "argsort_rows",
        [rows],
        [row_indices, rows],
        device="local-task",
    )

internal = compile_fixture(
    "internal", "tests/e2e/regression/argsort_internal_key_use.mlir"
)
for i, values in enumerate(cases):
    total = np.array(
        sum(int(value) for value in values) % (2**64), dtype=np.uint64
    ).view(np.int64)
    execute(
        f"internal-reduction-{i}",
        internal,
        "argsort_internal_key_use",
        [values],
        [indices(values), total, values],
    )
    execute(
        f"internal-consumer-{i}",
        internal,
        "argsort_internal_consumer",
        [values],
        [indices(values) + values, values],
    )
local = compile_fixture("local-keys", "tests/e2e/regression/argsort_local_keys.mlir")
for i, labels in enumerate(cases):
    execute(f"local-keys-{i}", local, "argsort_local_keys", [labels], [labels[::-1]])
    execute(
        f"local-shared-keys-{i}",
        local,
        "argsort_local_shared_keys",
        [labels],
        [labels[::-1], -np.arange(labels.size, dtype=np.int64)],
    )
    execute(
        f"local-aux-keys-{i}",
        local,
        "argsort_local_aux_keys",
        [labels],
        [labels[::-1], np.arange(labels.size, dtype=np.int64) + 100],
    )
two_sorts = compile_fixture(
    "two-sorts", "tests/e2e/regression/two_sorts_original_version.mlir"
)
for i, values in enumerate(cases):
    descending = np.array(
        sorted(range(values.size), key=values.__getitem__, reverse=True), dtype=np.int64
    )
    execute(
        f"two-sorts-{i}",
        two_sorts,
        "two_sorts_original_version",
        [values],
        [indices(values), descending + 100, values],
    )

# Existing RUN lines define compiler fixture pipelines; check both sides of each pipe.
filecheck = build / "llvm-project/bin/FileCheck"
require(filecheck.is_file(), "FileCheck was not built")
compiler_fixtures = [
    "compiler/src/iree/compiler/DispatchCreation/test/convert_region_to_workgroups.mlir",
    "compiler/src/iree/compiler/DispatchCreation/test/dispatch_linalg_on_tensors.mlir",
    "compiler/src/iree/compiler/DispatchCreation/test/form_scalar_dispatches.mlir",
    "compiler/src/iree/compiler/Dialect/LinalgExt/IR/test/canonicalize.mlir",
    "compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/tiling.mlir",
    "compiler/src/iree/compiler/Dialect/Flow/IR/test/dispatch_folding.mlir",
    "compiler/src/iree/compiler/Codegen/Common/test/iree_comprehensive_bufferize.mlir",
]
for i, relative in enumerate(compiler_fixtures):
    path = source / relative
    lines = [
        line.removeprefix("// RUN: ")
        for line in path.read_text().splitlines()
        if line.startswith("// RUN: ")
    ]
    require(bool(lines), relative + ": no RUN lines")
    for j, line in enumerate(lines):
        tokens = shlex.split(line)
        split = tokens.index("|")
        run, check = tokens[:split], tokens[split + 1 :]
        require(
            run[0] == "iree-opt" and check[0] == "FileCheck",
            relative + ": unsupported test command",
        )
        result = command(
            f"compiler-{i}-{j}",
            [optimizer] + [str(path) if arg == "%s" else arg for arg in run[1:]],
            "compiler test",
        )
        generated = out / f"compiler-{i}-{j}.mlir"
        generated.write_text(result.stdout)
        command(
            f"compiler-{i}-{j}-oracle",
            [filecheck]
            + [str(path) if arg == "%s" else arg for arg in check[1:]]
            + ["--input-file=" + str(generated)],
            "oracle",
        )
# Execute every literal runtime RUN line through the existing standard tool.
# The independent NumPy oracles above remain the semantic authority.
for i, relative in enumerate(report["fixtures"]):
    path = source / relative
    for j, line in enumerate(
        line for line in path.read_text().splitlines() if line.startswith("// RUN: ")
    ):
        tokens = shlex.split(line.removeprefix("// RUN: "))
        split = tokens.index("|")
        run, check = tokens[:split], tokens[split + 1 :]
        require(
            run[0] == "iree-run-mlir" and check[0] == "FileCheck",
            relative + ": unsupported runtime RUN line",
        )
        result = command(
            f"lit-runtime-{i}-{j}",
            [lit_runtime, "--task_topology_group_count=2"]
            + [str(path) if arg == "%s" else arg for arg in run[1:]],
            "runtime fixture",
        )
        generated = out / f"lit-runtime-{i}-{j}.txt"
        generated.write_text(result.stdout)
        command(
            f"lit-runtime-{i}-{j}-oracle",
            [filecheck]
            + [str(path) if arg == "%s" else arg for arg in check[1:]]
            + ["--input-file=" + str(generated)],
            "oracle",
        )

# A raw region with two captured lanes requires two Copy dispatches and Sort. The
# public pass statistic must count the initialization dispatch as well.
converter_test = source / compiler_fixtures[0]
text = converter_test.read_text()
start = text.index("util.func public @sort_unused_comparator_workspace(")
end = text.index("// -----", start)
statistics_input = out / "statistics-input.mlir"
statistics_input.write_text(
    "util.global private @device : !hal.device\n" + text[start:end]
)
statistics = command(
    "dispatch-statistics",
    [
        optimizer,
        statistics_input,
        "--pass-pipeline=builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups))",
        "--mlir-pass-statistics",
        "--mlir-pass-statistics-display=list",
        "-o",
        out / "statistics-output.mlir",
    ],
    "compiler test",
)
require(
    "3 num-dispatches" in statistics.stderr,
    "dispatch statistic failed to count all three created dispatches",
)

report["verdict"] = "pass"
report["scope"] = (
    "exact-source CPU semantic fixtures, allocation checks, compiler tests and incorrect-output control"
)
require(
    git("rev-parse", "HEAD") == expected["head"] and not git("status", "--porcelain"),
    "source changed during validation",
)
save()
print(
    json.dumps(
        {
            "verdict": "pass",
            "head": expected["head"],
            "commands": len(report["runs"]),
            "results": str(out / "results.json"),
        }
    )
)
