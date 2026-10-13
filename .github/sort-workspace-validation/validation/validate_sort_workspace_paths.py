# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate terminal workspace versions and buffer tiling at the exact source."""

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=root.parent / "source")
parser.add_argument("--out", type=Path, default=root / "evidence/sort-workspace-paths")
parser.add_argument(
    "--positive-report", type=Path, default=root / "evidence/exact-source/results.json"
)
args = parser.parse_args()
source = args.source.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
build = root / "build"
identity = json.loads((root / "validation/source-identity.json").read_text())
record = {
    "validation_id": os.environ["SORT_VALIDATION_RUN_ID"],
    "verdict": "running",
    "identity": identity,
    "commands": [],
    "cases": [],
}


def save():
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n")


def require(condition, message):
    if not condition:
        record.update(verdict="fail", failure=message)
        save()
        raise RuntimeError(message)


def unfinished():
    if record["verdict"] == "running":
        record.update(
            verdict="fail", failure="Required workspace checks did not complete"
        )
        save()


atexit.register(unfinished)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(name, arguments, expected=0, diagnostic=None):
    argv = [str(arg) for arg in arguments]
    try:
        process = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        require(False, name + ": required command timed out")
    (out / (name + ".log")).write_text(process.stdout + process.stderr)
    record["commands"].append(
        {
            "name": name,
            "argv": argv,
            "exit_status": process.returncode,
            "expected_status": expected,
            "required_diagnostic": diagnostic,
        }
    )
    save()
    require(process.returncode == expected, name + ": unexpected command status")
    require(
        diagnostic is None or diagnostic in process.stdout + process.stderr,
        name + ": expected failure reason absent",
    )
    return process


positive = json.loads(args.positive_report.read_text())
require(
    positive["verdict"] == "pass"
    and positive["identity"] == identity
    and positive["validation_id"] == record["validation_id"],
    "positive source admission failed",
)
require(
    subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    == identity["head"],
    "source HEAD changed",
)
require(
    not subprocess.check_output(
        ["git", "-C", source, "status", "--porcelain"], text=True
    ).strip(),
    "source worktree is dirty",
)
record["compiler_library_sha256"] = sha(build / "lib/libIREECompiler.so")
require(
    record["compiler_library_sha256"] == positive["compiler_library_sha256"],
    "workspace cases use a different compiler",
)
compiler = build / "tools/iree-compile"
optimizer = build / "tools/iree-opt"
runtime = build / "tools/iree-run-module"
filecheck = build / "llvm-project/bin/FileCheck"
version = command("compiler-version", [compiler, "--version"])
require(
    identity["head"] in version.stdout, "compiler release identity differs from source"
)
record["tool_sha256"] = {
    str(path): sha(path) for path in (compiler, optimizer, runtime, filecheck)
}
flags = [
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu=generic",
    "--mlir-disable-threading",
]
chains = source / "tests/e2e/regression/two_sorts_original_version.mlir"
earlier = source / "tests/e2e/regression/sort_preserve_intermediate_version.mlir"
record["fixtures"] = {
    str(path.relative_to(source)): sha(path) for path in (chains, earlier)
}


def compile_module(name, path):
    result = command(
        name + "-compile",
        [
            compiler,
            path,
            *flags,
            "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-codegen-iree-comprehensive-bufferize",
            "-o",
            out / (name + ".vmfb"),
        ],
    )
    require(
        "IR Dump After IREEComprehensiveBufferizePass" in result.stderr,
        name + ": missing allocation IR",
    )
    return result.stderr


chain_ir = compile_module("chains", chains)
require(
    "memref.alloca" not in chain_ir and "memref.alloc(" not in chain_ir,
    "terminal working role still needs a private allocation",
)
earlier_ir = compile_module("earlier", earlier)
allocation_lines = [
    line for line in earlier_ir.splitlines() if "= memref.alloca" in line
]
require(
    len(allocation_lines) == 1
    and "memref.alloca()" in allocation_lines[0]
    and "memref<4xi64" in allocation_lines[0]
    and "memref.alloc(" not in earlier_ir,
    "earlier-version preservation copy is not the expected bounded allocation",
)
record["allocation_checks"] = [
    "chains: no private allocations",
    "earlier version: one bounded four-element copy",
]


def execute(name, module, function, inputs, expected, device):
    input_paths = [out / f"{name}-input-{i}.npy" for i in range(len(inputs))]
    output_paths = [out / f"{name}-output-{i}.npy" for i in range(len(expected))]
    for path, values in zip(input_paths, inputs):
        np.save(path, values)
    for path in output_paths:
        path.unlink(missing_ok=True)
    command(
        name,
        [
            runtime,
            "--module=" + str(module),
            "--function=" + function,
            "--device=" + device,
            "--task_topology_group_count=2",
            *["--input=@" + str(path) for path in input_paths],
            *["--output=@" + str(path) for path in output_paths],
        ],
    )
    for index, (path, wanted) in enumerate(zip(output_paths, expected)):
        require(path.is_file(), name + ": runtime output is missing")
        actual = np.load(path, allow_pickle=False)
        require(
            actual.dtype == wanted.dtype
            and actual.shape == wanted.shape
            and np.array_equal(actual, wanted),
            name + f": incorrect output {index}",
        )
    record["cases"].append(
        {"name": name, "function": function, "device": device, "verdict": "pass"}
    )
    save()


keys_cases = (
    [3, 1, 2],
    [],
    [7],
    [2, 1, 2, 0],
    [1, 2, 3],
    [5, 4, 3, 2, 1],
    [-(2**63), 2**63 - 1, 0],
    list(range(17, -1, -1)),
)
for device in ("local-sync", "local-task"):
    for number, keys in enumerate(keys_cases):
        labels = [11 + i * 7 if i % 2 == 0 else -13 - i * 5 for i in range(len(keys))]
        inputs = [np.array(keys, dtype=np.int64), np.array(labels, dtype=np.int64)]
        order = sorted(range(len(keys)), key=lambda i: keys[i], reverse=True)
        expected = [np.array([labels[i] for i in order], dtype=np.int64), *inputs]
        execute(
            f"dependent-{device}-{number}",
            out / "chains.vmfb",
            "two_sorts_dependent",
            inputs,
            expected,
            device,
        )
        expected = [inputs[1][::-1], np.full(len(keys), 7, dtype=np.int64), inputs[1]]
        execute(
            f"local-update-{device}-{number}",
            out / "chains.vmfb",
            "two_sorts_local_update",
            [inputs[1]],
            expected,
            device,
        )
    for number, keys in enumerate(
        ([4, 1, 3, 2], [2, 1, 2, 0], [-(2**63), 2**63 - 1, 0, -1], [3, 3, 3, 3])
    ):
        labels = [7, -3, 9, 11]
        order = sorted(range(4), key=lambda i: keys[i], reverse=True)
        weighted = sum((i + 1) * key for i, key in enumerate(sorted(keys)))
        inputs = [np.array(keys, dtype=np.int64), np.array(labels, dtype=np.int64)]
        expected = [
            np.array([labels[i] for i in order], dtype=np.int64),
            np.array(weighted % (2**64), dtype=np.uint64).view(np.int64),
            *inputs,
        ]
        execute(
            f"earlier-{device}-{number}",
            out / "earlier.vmfb",
            "two_sorts_keep_internal_version",
            inputs,
            expected,
            device,
        )

# A real buffer-form operation must reject the partial axis before codegen.
partial = out / "buffer-partial.mlir"
partial.write_text(
    """func.func @sort_partial_axis(%arg0: memref<4xi32>) {
  iree_linalg_ext.sort dimension(0) outs(%arg0 : memref<4xi32>) {
  ^bb0(%lhs: i32, %rhs: i32):
    %cmp = arith.cmpi sle, %lhs, %rhs : i32
    iree_linalg_ext.yield %cmp : i1
  }
  return
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %sort = transform.structured.match ops{["iree_linalg_ext.sort"]} in %module_op : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %sort tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
"""
)
result = command(
    "buffer-partial-rejection",
    [optimizer, partial, "--iree-transform-dialect-interpreter"],
    expected=1,
    diagnostic="failed to tile operation",
)
require(
    "failed to generate tiling loops" in result.stderr,
    "buffer tile failed for an unrelated reason",
)

# Replay the new bounded earlier-version fixture's literal RUN oracle.
result = command(
    "earlier-literal",
    [
        build / "tools/iree-run-mlir",
        "--task_topology_group_count=2",
        "--Xcompiler,iree-hal-target-backends=llvm-cpu",
        "--Xcompiler,iree-llvmcpu-target-cpu=generic",
        earlier,
        "--input=4xi64=[4 1 3 2]",
        "--input=4xi64=[7 -3 9 11]",
    ],
)
text = out / "earlier-literal.txt"
text.write_text(result.stdout)
command(
    "earlier-literal-oracle",
    [filecheck, earlier, "--check-prefix=EARLIER", "--input-file=" + str(text)],
)
require(
    len(record["cases"]) == 40 and len(record["commands"]) == 46,
    "required workspace phases or runtime cases are incomplete",
)
record["verdict"] = "pass"
save()
print(
    json.dumps(
        {
            "verdict": "pass",
            "runtime_cases": len(record["cases"]),
            "commands": len(record["commands"]),
        }
    )
)
