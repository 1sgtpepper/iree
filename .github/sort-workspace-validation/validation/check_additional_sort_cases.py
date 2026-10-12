# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bounded runtime challenges using the already validated exact-source compiler."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import uuid

import numpy as np


root = Path(__file__).resolve().parents[1]
source = root.parent / "source"
build = root / "build"
out = root / "evidence/correctness-recheck/additional"
out.mkdir(parents=True, exist_ok=True)
identity = json.loads((root / "validation/source-identity.json").read_text())
record = {
    "validation_id": os.environ.get("SORT_VALIDATION_RUN_ID", str(uuid.uuid4())),
    "verdict": "running",
    "identity": identity,
    "scope": "bounded CPU challenges; existing exact-source binaries, no rebuild",
    "commands": [],
    "cases": [],
}


def save():
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n")


def require(condition, message):
    if not condition:
        record["verdict"] = "fail"
        record["failure"] = message
        save()
        raise SystemExit(message)


def command(name, args):
    try:
        result = subprocess.run(
            [str(arg) for arg in args], capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        require(False, name + ": timeout; inconclusive execution")
    log = out / (name + ".log")
    log.write_text(result.stdout + result.stderr)
    record["commands"].append(
        {
            "name": name,
            "argv": [str(arg) for arg in args],
            "status": result.returncode,
            "log": str(log),
        }
    )
    save()
    require(result.returncode == 0, name + ": command failed; see " + str(log))
    return result


save()
require(
    subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    == identity["head"],
    "source identity changed",
)
require(
    not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]),
    "source is dirty",
)
library = build / "lib/libIREECompiler.so"
record["compiler_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
previous = json.loads((root / "evidence/exact-source/results.json").read_text())
require(
    previous.get("verdict") == "pass"
    and previous.get("identity") == identity
    and previous.get("validation_id") == record["validation_id"],
    "fresh exact-source positive validation is required; run validation/validate.sh",
)
require(
    record["compiler_library_sha256"] == previous["compiler_library_sha256"],
    "compiler library differs from exact-source validation",
)


def axis_function(rank, axis):
    typ = "tensor<" + "?x" * rank + "i64>"
    names = ", ".join(f"d{i}" for i in range(rank))
    dims = "\n".join(
        f"  %c{i} = arith.constant {i} : index\n"
        f"  %n{i} = tensor.dim %input, %c{i} : {typ}"
        for i in range(rank)
    )
    return f"""
func.func @axis{rank}_{axis}(%input: {typ}) -> ({typ}, {typ}) {{
{dims}
  %empty = tensor.empty({', '.join(f'%n{i}' for i in range(rank))}) : {typ}
  %indices = linalg.generic {{indexing_maps = [affine_map<({names}) -> ({names})>],
      iterator_types = [{', '.join('"parallel"' for _ in range(rank))}]}}
      outs(%empty : {typ}) {{
  ^bb0(%out: i64):
    %i = linalg.index {axis} : index
    %v = arith.index_cast %i : index to i64
    linalg.yield %v : i64
  }} -> {typ}
  %sorted:2 = iree_linalg_ext.sort dimension({axis}) outs(%input, %indices : {typ}, {typ}) {{
  ^bb0(%a: i64, %b: i64, %ai: i64, %bi: i64):
    %cmp = arith.cmpi sle, %a, %b : i64
    iree_linalg_ext.yield %cmp : i1
  }} -> {typ}, {typ}
  return %sorted#1, %input : {typ}, {typ}
}}
"""


mlir = """
func.func @lexsort(%p: tensor<?xi32>, %s: tensor<?xi16>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi32>, tensor<?xi16>, tensor<?xi64>) {
  %sorted:3 = iree_linalg_ext.sort dimension(0)
      outs(%p, %s, %labels : tensor<?xi32>, tensor<?xi16>, tensor<?xi64>) {
  ^bb0(%ap: i32, %bp: i32, %as: i16, %bs: i16, %al: i64, %bl: i64):
    %lt = arith.cmpi slt, %ap, %bp : i32
    %eq = arith.cmpi eq, %ap, %bp : i32
    %ge = arith.cmpi uge, %as, %bs : i16
    %tie = arith.andi %eq, %ge : i1
    %cmp = arith.ori %lt, %tie : i1
    iree_linalg_ext.yield %cmp : i1
  } -> tensor<?xi32>, tensor<?xi16>, tensor<?xi64>
  return %sorted#2, %p, %s, %labels
      : tensor<?xi64>, tensor<?xi32>, tensor<?xi16>, tensor<?xi64>
}

func.func @local_from_input(%input: tensor<?xi32>, %labels: tensor<?xi64>)
    -> (tensor<?xi64>, tensor<?xi32>, tensor<?xi32>, tensor<?xi64>) {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %input, %c0 : tensor<?xi32>
  %results:2 = flow.dispatch.region -> (tensor<?xi64>{%n}, tensor<?xi32>{%n}) {
    %empty = tensor.empty(%n) : tensor<?xi32>
    %generated:2 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%input : tensor<?xi32>) outs(%empty, %empty : tensor<?xi32>, tensor<?xi32>) {
    ^bb0(%in: i32, %out_key: i32, %out_aux: i32):
      %mask = arith.constant -1 : i32
      %five = arith.constant 5 : i32
      %key = arith.xori %in, %mask : i32
      %aux = arith.addi %in, %five : i32
      linalg.yield %key, %aux : i32, i32
    } -> (tensor<?xi32>, tensor<?xi32>)
    %sorted:2 = iree_linalg_ext.sort dimension(0)
        outs(%generated#0, %labels : tensor<?xi32>, tensor<?xi64>) {
    ^bb0(%a: i32, %b: i32, %al: i64, %bl: i64):
      %cmp = arith.cmpi sle, %a, %b : i32
      iree_linalg_ext.yield %cmp : i1
    } -> tensor<?xi32>, tensor<?xi64>
    flow.return %sorted#1, %generated#1 : tensor<?xi64>, tensor<?xi32>
  }
  return %results#0, %results#1, %input, %labels
      : tensor<?xi64>, tensor<?xi32>, tensor<?xi32>, tensor<?xi64>
}
"""
axes = [(2, 0), (3, 0), (3, 1), (3, 2)]
mlir += "".join(axis_function(rank, axis) for rank, axis in axes)
fixture = out / "additional.mlir"
fixture.write_text(mlir)
record["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()


def execute(name, artifact, function, inputs, expected, device):
    args = [
        build / "tools/iree-run-module",
        "--module=" + str(artifact),
        "--device=" + device,
        "--function=" + function,
    ]
    if device == "local-task":
        args.append("--task_topology_group_count=2")
    for i, array in enumerate(inputs):
        path = out / f"{name}-input{i}.npy"
        np.save(path, array)
        args.append("--input=@" + str(path))
    paths = []
    for i in range(len(expected)):
        path = out / f"{name}-output{i}.npy"
        path.unlink(missing_ok=True)
        paths.append(path)
        args.append("--output=@" + str(path))
    command(name, args)
    failures = []
    for i, (path, oracle) in enumerate(zip(paths, expected)):
        require(path.is_file(), name + ": missing output")
        actual = np.load(path, allow_pickle=False)
        if not (
            actual.dtype == oracle.dtype
            and actual.shape == oracle.shape
            and np.array_equal(actual, oracle)
        ):
            failures.append(f"output {i}: {actual.tolist()} != {oracle.tolist()}")
    record["cases"].append(
        {
            "name": name,
            "function": function,
            "device": device,
            "verdict": "fail" if failures else "pass",
            "failures": failures,
            "oracle": "Python ordering and exact dtype/shape/value",
        }
    )
    save()


for configuration, extra, device in [
    ("default", [], "local-sync"),
    ("distributed", ["--iree-llvmcpu-distribution-size=1"], "local-task"),
]:
    artifact = out / (configuration + ".vmfb")
    compiled = command(
        configuration + "-compile",
        [
            build / "tools/iree-compile",
            fixture,
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
            "--mlir-disable-threading",
            "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-codegen-iree-comprehensive-bufferize",
            *extra,
            "-o",
            artifact,
        ],
    )
    require(
        "IR Dump After IREEComprehensiveBufferizePass" in compiled.stderr,
        configuration + ": allocation evidence absent",
    )
    require(
        "memref.alloca" not in compiled.stderr
        and "memref.alloc(" not in compiled.stderr,
        configuration + ": unexpected private kernel allocation",
    )
    for i, primary in enumerate(itertools.product([-1, 0, 1], repeat=3)):
        p = np.array(primary, dtype=np.int32)
        s = np.array([0, -1, -32768] if i % 2 else [-1, 32767, 0], dtype=np.int16)
        labels = np.array([17, -53, 2**62], dtype=np.int64)
        unsigned = s.view(np.uint16)
        order = sorted(range(3), key=lambda j: (int(p[j]), -int(unsigned[j])))
        execute(
            f"{configuration}-lex-{i}",
            artifact,
            "lexsort",
            [p, s, labels],
            [labels[order], p, s, labels],
            device,
        )
    for i, primary in enumerate([[], [7], [2**31 - 1, -(2**31), 0, -(2**31)]]):
        p = np.array(primary, dtype=np.int32)
        s = (np.arange(p.size, dtype=np.int64) * 30000).astype(np.int16)
        labels = np.arange(p.size, dtype=np.int64) * 137 - 19
        unsigned = s.view(np.uint16)
        order = sorted(range(p.size), key=lambda j: (int(p[j]), -int(unsigned[j])))
        execute(
            f"{configuration}-lex-boundary-{i}",
            artifact,
            "lexsort",
            [p, s, labels],
            [labels[order], p, s, labels],
            device,
        )
    for rank, axis in axes:
        shapes = (
            [(3, 5), (0, 4), (3, 0), (1, 3), (17, 7)]
            if rank == 2
            else [(2, 3, 5), (0, 3, 4), (2, 0, 4), (2, 3, 0), (1, 1, 1), (3, 17, 5)]
        )
        for i, shape in enumerate(shapes):
            values = (
                ((np.arange(np.prod(shape), dtype=np.int64) * 37) % 11) - 5
            ).reshape(shape)
            moved = np.moveaxis(values, axis, -1)
            n = moved.shape[-1]
            slices = moved.reshape(int(np.prod(moved.shape[:-1])), n)
            expected = np.empty_like(slices)
            for j, row in enumerate(slices):
                expected[j] = sorted(range(n), key=row.__getitem__)
            expected = np.moveaxis(expected.reshape(moved.shape), -1, axis)
            execute(
                f"{configuration}-axis{rank}_{axis}-{i}",
                artifact,
                f"axis{rank}_{axis}",
                [values],
                [expected, values],
                device,
            )
    for i, primary in enumerate(
        [[], [7], [3, 1, 2], [2, 1, 2], [2**31 - 1, -(2**31), 0, -(2**31)]]
    ):
        values = np.array(primary, dtype=np.int32)
        labels = np.arange(values.size, dtype=np.int64) * 137 - 19
        order = sorted(range(values.size), key=lambda j: ~int(values[j]))
        aux = (values.astype(np.int64) + 5).astype(np.int32)
        execute(
            f"{configuration}-local-input-{i}",
            artifact,
            "local_from_input",
            [values, labels],
            [labels[order], aux, values, labels],
            device,
        )

require(
    not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]),
    "source changed during checks",
)
require(
    all(case["verdict"] == "pass" for case in record["cases"]),
    "runtime oracle rejected one or more additional cases; see results.json",
)
require(
    len(record["cases"]) == 116
    and {case["function"] for case in record["cases"]}
    == {"lexsort", "local_from_input", "axis2_0", "axis3_0", "axis3_1", "axis3_2"},
    "required additional case families are incomplete",
)
record["verdict"] = "pass"
save()
print(
    json.dumps(
        {
            "verdict": record["verdict"],
            "cases": len(record["cases"]),
            "head": identity["head"],
            "results": str(out / "results.json"),
        }
    )
)
