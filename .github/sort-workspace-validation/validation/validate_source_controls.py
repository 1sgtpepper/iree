# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build actual C++ controls, distinguish phases, and restore exact final source."""

import atexit
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
source = root.parent / "source"
build = root / "build"
out = root / "evidence/source-controls"
out.mkdir(exist_ok=True)
identity = json.loads((root / "validation/source-identity.json").read_text())
positive = json.loads((root / "evidence/exact-source/results.json").read_text())
if (
    positive["verdict"] != "pass"
    or positive["identity"] != identity
    or positive.get("validation_id") != os.environ.get("SORT_VALIDATION_RUN_ID")
    or not positive.get("scope", "").startswith("exact-source CPU semantic")
):
    raise SystemExit(
        "An exact-identity positive execution is required before source controls."
    )
relative = (
    "compiler/src/iree/compiler/DispatchCreation/ConvertDispatchRegionsToWorkgroups.cpp"
)
tiling_relative = (
    "compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp"
)
cow_relative = (
    "compiler/src/iree/compiler/Dialect/Stream/Transforms/MaterializeCopyOnWrite.cpp"
)
cpp = source / relative
final_files = {
    name: (source / name).read_bytes()
    for name in (relative, tiling_relative, cow_relative)
}
compiler = build / "tools/iree-compile"
runtime = build / "tools/iree-run-module"
report = {
    "validation_id": os.environ["SORT_VALIDATION_RUN_ID"],
    "verdict": "running",
    "identity": identity,
    "runs": [],
}


def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def unfinished():
    if report["verdict"] == "running":
        report["verdict"] = "fail"
        report["failure"] = "Required controls did not complete"
        save()


atexit.register(unfinished)


def require(ok, message):
    if not ok:
        report["verdict"] = "fail"
        report["failure"] = message
        save()
        raise RuntimeError(message)


def command(name, args, phase, status=0):
    args = [str(v) for v in args]
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    (out / (name + ".log")).write_text(result.stdout + result.stderr)
    report["runs"].append(
        {
            "name": name,
            "command": args,
            "phase": phase,
            "exit_status": result.returncode,
            "production_sha256": {
                path: hashlib.sha256((source / path).read_bytes()).hexdigest()
                for path in final_files
            },
        }
    )
    if phase == "build":
        report["runs"][-1]["compiler_library_sha256"] = hashlib.sha256(
            (build / "lib/libIREECompiler.so").read_bytes()
        ).hexdigest()
    save()
    require(result.returncode == status, name + ": unexpected " + phase + " status")
    return result


def restore():
    for name, contents in final_files.items():
        (source / name).write_bytes(contents)


def reject_ordering(name, actual_path, input_values, incorrect):
    actual = np.load(actual_path, allow_pickle=False)
    require(
        actual.dtype == np.int64
        and np.array_equal(actual, np.array(incorrect, dtype=np.int64)),
        name + ": control failed for an unexpected reason",
    )
    input_path = out / (name + "-input.npy")
    np.save(input_path, input_values)
    oracle = (
        "import numpy as np,sys; values=np.load(sys.argv[1],allow_pickle=False); "
        "actual=np.load(sys.argv[2],allow_pickle=False); "
        "expected=np.array(sorted(range(values.size),key=values.__getitem__),dtype=np.int64); "
        "ok=actual.dtype==expected.dtype and actual.shape==expected.shape and np.array_equal(actual,expected); "
        'print("correct output" if ok else "incorrect ordering: " + repr(actual.tolist()) + " != " + repr(expected.tolist())); '
        "sys.exit(0 if ok else 1)"
    )
    result = command(
        name + "-oracle",
        [sys.executable, "-c", oracle, input_path, actual_path],
        "oracle",
        status=1,
    )
    require(
        "incorrect ordering:" in result.stdout and "[1, 2, 0]" in result.stdout,
        name + ": independent oracle failed for an unrelated reason",
    )
    report["runs"][-1]["outcome"] = "expected independent-oracle rejection"


flags = [
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu=generic",
    "--mlir-disable-threading",
]
fixture = source / "tests/e2e/regression/argsort_dynamic.mlir"
try:
    # Literal baseline production with identical candidate fixtures overlaid.
    for name in final_files:
        (source / name).write_bytes(
            subprocess.check_output(
                ["git", "-C", str(source), "show", identity["base"] + ":" + name]
            )
        )
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--name-only", identity["base"]], text=True
    ).splitlines()
    allowed = {
        "tests/e2e/regression/BUILD.bazel",
        "tests/e2e/regression/CMakeLists.txt",
        *[
            "tests/e2e/regression/" + name + ".mlir"
            for name in (
                "argsort_dynamic",
                "argsort_preserve_input_dynamic",
                "argsort_comparator_lanes_dynamic",
                "argsort_views_dynamic",
                "argsort_internal_key_use",
                "argsort_local_keys",
                "two_sorts_original_version",
                "sort_preserve_intermediate_version",
                "sort_dynamic_observed_versions",
                "scatter_preserve_borrowed_input",
            )
        ],
        "compiler/src/iree/compiler/DispatchCreation/test/convert_region_to_workgroups.mlir",
        "compiler/src/iree/compiler/DispatchCreation/test/dispatch_linalg_on_tensors.mlir",
        "compiler/src/iree/compiler/Codegen/Common/test/iree_comprehensive_bufferize.mlir",
        "compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/tiling.mlir",
        "compiler/src/iree/compiler/Dialect/Stream/Transforms/test/materialize_copy_on_write.mlir",
    }
    require(
        set(changed) <= allowed,
        "baseline overlay has an unexpected production difference",
    )
    command("baseline-build", ["ninja", "-C", build, "-j2", "iree-compile"], "build")
    result = command(
        "baseline-compile",
        [
            compiler,
            fixture,
            *flags,
            "--mlir-print-ir-before=iree-dispatch-creation-convert-dispatch-regions-to-workgroups",
            "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-codegen-iree-comprehensive-bufferize",
            "-o",
            out / "baseline.vmfb",
        ],
        "compile",
        status=1,
    )
    require(
        "'memref.alloca' op expected no unbounded stack allocations" in result.stderr,
        "baseline failed for an unrelated reason",
    )
    report["runs"][-1][
        "outcome"
    ] = "expected semantic allocation failure; build succeeded"
    report["baseline_production"] = (
        "matches literal base "
        + identity["base"]
        + "; fixtures overlaid from "
        + identity["head"]
    )

    # Capture both boundary diagnostics from the already-built base compiler.
    # These are comparisons, not success controls: either source may reject a
    # fixture, and the diagnostic must be inspected before classifying it.
    report["baseline_boundaries"] = {}
    for name, fixture_name in (
        ("local-shared-scalar", "local_shared_scalar_observed_keys.mlir"),
        ("dependent-observed", "dependent_observed_dynamic.mlir"),
    ):
        boundary_fixture = root / "validation/fixtures" / fixture_name
        boundary = subprocess.run(
            [
                str(compiler),
                str(boundary_fixture),
                *flags,
                "--mlir-print-ir-before=iree-dispatch-creation-convert-dispatch-regions-to-workgroups",
                "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-codegen-iree-comprehensive-bufferize",
                "-o",
                str(out / ("baseline-" + name + ".vmfb")),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        (out / ("baseline-" + name + ".log")).write_text(
            boundary.stdout + boundary.stderr
        )
        report["baseline_boundaries"][name] = {
            "exit_status": boundary.returncode,
            "fixture_sha256": hashlib.sha256(boundary_fixture.read_bytes()).hexdigest(),
            "compiler_library_sha256": hashlib.sha256(
                (build / "lib/libIREECompiler.so").read_bytes()
            ).hexdigest(),
        }
    save()

    # Preserve allocation/working roles and deterministically destroy initialization.
    restore()
    text = final_files[relative].decode()
    pattern = r"(?<=auto copy =\n          )linalg::CopyOp::create\(rewriter, sortOp.getLoc\(\), ValueRange\{source\},\s*ValueRange\{allocation.getResult\(\)\}\);"
    text = text.replace(
        '#include "mlir/Dialect/Affine/IR/AffineOps.h"',
        '#include "mlir/Dialect/Affine/IR/AffineOps.h"\n#include "mlir/Dialect/Arith/IR/Arith.h"',
    )
    text, count = re.subn(
        pattern,
        """linalg::FillOp::create(rewriter, sortOp.getLoc(),
              ValueRange{arith::ConstantOp::create(rewriter, sortOp.getLoc(),
                  rewriter.getZeroAttr(type.getElementType())).getResult()},
              ValueRange{allocation.getResult()});""",
        text,
    )
    require(count == 1, "initialization mutation anchor changed")
    cpp.write_text(text)
    command(
        "zero-initialization-build",
        ["ninja", "-C", build, "-j2", "iree-compile"],
        "build",
    )
    command(
        "zero-initialization-compile",
        [compiler, fixture, *flags, "-o", out / "zero.vmfb"],
        "compile",
    )
    output = out / "zero-output.npy"
    output.unlink(missing_ok=True)
    command(
        "zero-initialization-run",
        [
            runtime,
            "--module=" + str(out / "zero.vmfb"),
            "--device=local-sync",
            "--function=argsort",
            "--input=3xi64=[3 1 2]",
            "--output=@" + str(output),
        ],
        "runtime",
    )
    reject_ordering(
        "zero-initialization", output, np.array([3, 1, 2], dtype=np.int64), [0, 1, 2]
    )

    # Keep valid initialized storage, restore the old illegal tiling behavior.
    restore()
    (source / tiling_relative).write_bytes(
        subprocess.check_output(
            ["git", "-C", str(source), "show", identity["base"] + ":" + tiling_relative]
        )
    )
    command(
        "partial-sort-build", ["ninja", "-C", build, "-j2", "iree-compile"], "build"
    )
    internal = source / "tests/e2e/regression/argsort_internal_key_use.mlir"
    command(
        "partial-sort-compile",
        [
            compiler,
            internal,
            *flags,
            "--mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize",
            "-o",
            out / "partial-sort.vmfb",
        ],
        "compile",
    )
    output = out / "partial-sort-output.npy"
    output.unlink(missing_ok=True)
    command(
        "partial-sort-run",
        [
            runtime,
            "--module=" + str(out / "partial-sort.vmfb"),
            "--device=local-sync",
            "--function=argsort_internal_key_use",
            "--input=3xi64=[3 1 2]",
            "--output=@" + str(output),
            "--output=@" + str(out / "partial-sort-sum.npy"),
            "--output=@" + str(out / "partial-sort-original.npy"),
        ],
        "runtime",
    )
    reject_ordering(
        "partial-sort", output, np.array([3, 1, 2], dtype=np.int64), [1, 0, 2]
    )

    # Restore the local single-use shortcut. The Scatter result is correct,
    # but its borrowed original input must now be overwritten.
    restore()
    (source / cow_relative).write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "show",
                "96fc7b5935599240e169344736547682e3e8a16d:" + cow_relative,
            ]
        )
    )
    command(
        "borrowed-preservation-build",
        ["ninja", "-C", build, "-j2", "iree-compile"],
        "build",
    )
    require(
        report["runs"][-1]["compiler_library_sha256"]
        != positive["compiler_library_sha256"],
        "borrowed storage control reused the positive compiler library",
    )
    scatter = root / "validation/fixtures/linalg_ext_scatter_observed_input.mlir"
    command(
        "borrowed-preservation-compile",
        [compiler, scatter, *flags, "-o", out / "borrowed-preservation.vmfb"],
        "compile",
    )
    original = np.arange(8, dtype=np.int32)
    updates = np.array([100, 200, 300], dtype=np.int32)
    indices = np.array([[1], [3], [5]], dtype=np.int32)
    inputs = [out / f"borrowed-preservation-input-{i}.npy" for i in range(3)]
    for path, values in zip(inputs, (original, updates, indices)):
        np.save(path, values)
    sorted_output = out / "borrowed-preservation-updated.npy"
    output = out / "borrowed-preservation-original.npy"
    sorted_output.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    command(
        "borrowed-preservation-run",
        [
            runtime,
            "--module=" + str(out / "borrowed-preservation.vmfb"),
            "--device=local-sync",
            "--function=scatter_observed",
            *("--input=@" + str(path) for path in inputs),
            "--output=@" + str(sorted_output),
            "--output=@" + str(output),
        ],
        "runtime",
    )
    require(
        sorted_output.is_file() and output.is_file(),
        "borrowed storage control omitted an output",
    )
    expected = original.copy()
    expected[indices[:, 0]] = updates
    updated = np.load(sorted_output, allow_pickle=False)
    actual = np.load(output, allow_pickle=False)
    require(
        updated.dtype == actual.dtype == np.int32
        and updated.shape == actual.shape == original.shape
        and np.array_equal(updated, expected)
        and np.array_equal(actual, updated),
        "borrowed storage control failed for an unexpected reason",
    )
    oracle = (
        "import numpy as np,sys; actual=np.load(sys.argv[1],allow_pickle=False); "
        "original=np.load(sys.argv[2],allow_pickle=False); "
        "ok=actual.dtype==original.dtype and actual.shape==original.shape and np.array_equal(actual,original); "
        'print("input preserved" if ok else "input changed: " + repr(actual.tolist())); '
        "sys.exit(0 if ok else 1)"
    )
    result = command(
        "borrowed-preservation-oracle",
        [sys.executable, "-c", oracle, output, inputs[0]],
        "oracle",
        status=1,
    )
    require(
        "input changed: [0, 100, 2, 200, 4, 300, 6, 7]" in result.stdout,
        "borrowed storage oracle failed for an unrelated reason",
    )
    report["runs"][-1][
        "outcome"
    ] = "expected independent-oracle rejection of input corruption"

    # Rebuild the reviewed head that lacks both repaired workspace paths.
    restore()
    old_head = "aa08621ce694db3ec54197ad29ec84232f44cce7"
    for name in final_files:
        (source / name).write_bytes(
            subprocess.check_output(
                ["git", "-C", str(source), "show", old_head + ":" + name]
            )
        )
    command(
        "old-paths-build",
        ["ninja", "-C", build, "-j2", "iree-compile", "iree-opt"],
        "build",
    )
    chains = source / "tests/e2e/regression/two_sorts_original_version.mlir"
    result = command(
        "old-paths-chain-compile",
        [
            compiler,
            chains,
            *flags,
            "--mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize",
            "-o",
            out / "old-chain.vmfb",
        ],
        "compile",
        status=1,
    )
    require(
        "'memref.alloca' op expected no unbounded stack allocations" in result.stderr,
        "old workspace path failed for an unrelated reason",
    )
    report["runs"][-1][
        "outcome"
    ] = "expected dependent-chain allocation failure; build succeeded"

    partial = out / "old-buffer-partial-source.mlir"
    partial.write_text(
        """func.func @memref_sort(%input: tensor<4xi32>) -> tensor<4xi32> {
  %result = flow.dispatch.region -> (tensor<4xi32>) {
    %empty = tensor.empty() : tensor<4xi32>
    %copy = linalg.copy ins(%input : tensor<4xi32>) outs(%empty : tensor<4xi32>) -> tensor<4xi32>
    %buffer = bufferization.to_buffer %copy : tensor<4xi32> to memref<4xi32>
    iree_linalg_ext.sort dimension(0) outs(%buffer : memref<4xi32>) {
    ^bb0(%lhs: i32, %rhs: i32):
      %cmp = arith.cmpi sle, %lhs, %rhs : i32
      iree_linalg_ext.yield %cmp : i1
    }
    %tensor = bufferization.to_tensor %buffer restrict writable : memref<4xi32> to tensor<4xi32>
    flow.return %tensor : tensor<4xi32>
  }
  return %result : tensor<4xi32>
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
    transformed = out / "old-buffer-partial.mlir"
    command(
        "old-paths-buffer-transform",
        [
            build / "tools/iree-opt",
            partial,
            "--iree-transform-dialect-interpreter",
            "--iree-transform-dialect-drop-schedule",
            "--canonicalize",
            "-o",
            transformed,
        ],
        "compiler test",
    )
    command(
        "old-paths-buffer-compile",
        [compiler, transformed, *flags, "-o", out / "old-buffer.vmfb"],
        "compile",
    )
    output = out / "old-buffer-output.npy"
    output.unlink(missing_ok=True)
    command(
        "old-paths-buffer-runtime",
        [
            runtime,
            "--module=" + str(out / "old-buffer.vmfb"),
            "--device=local-sync",
            "--function=memref_sort",
            "--input=4xi32=[4 3 2 1]",
            "--output=@" + str(output),
        ],
        "runtime",
    )
    actual = np.load(output, allow_pickle=False)
    require(
        actual.dtype == np.int32
        and actual.shape == (4,)
        and np.array_equal(actual, np.array([3, 4, 1, 2], dtype=np.int32)),
        "old buffer tile control failed for an unexpected reason",
    )
    oracle = (
        "import numpy as np,sys; actual=np.load(sys.argv[1],allow_pickle=False); "
        "expected=np.array(sorted([4,3,2,1]),dtype=np.int32); "
        "ok=actual.dtype==expected.dtype and actual.shape==expected.shape and np.array_equal(actual,expected); "
        'print("correct sort" if ok else "incorrect buffer sort: " + repr(actual.tolist())); '
        "sys.exit(0 if ok else 1)"
    )
    result = command(
        "old-paths-buffer-oracle",
        [sys.executable, "-c", oracle, output],
        "oracle",
        status=1,
    )
    require(
        "incorrect buffer sort: [3, 4, 1, 2]" in result.stdout,
        "old buffer oracle failed for an unrelated reason",
    )
    report["runs"][-1][
        "outcome"
    ] = "expected independent-oracle rejection of partial buffer sort"
    # Rebuild the first repair: destination updates work, but local empty
    # producer destinations still lack managed storage.
    restore()
    cpp.write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "show",
                "4fc318a9a5cb55c95f3d943de81afeb98583df1e:" + relative,
            ]
        )
    )
    command(
        "local-initializer-build",
        ["ninja", "-C", build, "-j2", "iree-compile"],
        "build",
    )
    result = command(
        "local-initializer-compile",
        [
            compiler,
            chains,
            *flags,
            "--mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize",
            "-o",
            out / "local-initializer.vmfb",
        ],
        "compile",
        status=1,
    )
    require(
        "exceeded stack allocation limit" in result.stderr
        and "two_sorts_local_update" in result.stderr,
        "old local initializer failed for an unrelated reason",
    )
    report["runs"][-1][
        "outcome"
    ] = "expected local-initializer allocation failure; build succeeded"

    # The constant-only offset guard rejects legal full-axis tensor and buffer
    # tiles. The native fixture must expose both rejections after a real build.
    restore()
    (source / tiling_relative).write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "show",
                "b2295f752829d790ff36ecff3b83b4149699e74a:" + tiling_relative,
            ]
        )
    )
    command(
        "constant-offset-build",
        ["ninja", "-C", build, "-j2", "iree-compile", "iree-opt"],
        "build",
    )
    result = command(
        "constant-offset-tiling",
        [
            build / "tools/iree-opt",
            source
            / "compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/tiling.mlir",
            "--iree-transform-dialect-interpreter",
            "--split-input-file",
            "--verify-diagnostics",
            "-canonicalize",
            "-cse",
        ],
        "compiler test",
        status=1,
    )
    require(
        result.stderr.count("unexpected error:") == 3
        and result.stderr.count(
            "unexpected error: 'iree_linalg_ext.sort' op failed to generate tiling loops"
        )
        == 2
        and result.stderr.count(
            "unexpected error: 'iree_linalg_ext.sort' op failed to tile operation"
        )
        == 1,
        "constant offset guard failed for an unrelated reason",
    )
    report["runs"][-1][
        "outcome"
    ] = "expected rejection of legal full-axis tensor and buffer tiles; build succeeded"
    report["controls"] = "pass"
finally:
    restore()
    command(
        "restore-build",
        [
            "ninja",
            "-C",
            build,
            "-j2",
            "iree-compile",
            "iree-opt",
            "iree-run-module",
            "iree-run-mlir",
            "FileCheck",
        ],
        "build",
    )
    diff = subprocess.run(
        ["git", "-C", str(source), "diff", "--exit-code"],
        capture_output=True,
        text=True,
    )
    require(diff.returncode == 0, "source restoration left a diff")
    require(
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        == identity["head"],
        "source restoration changed HEAD",
    )
    report["restored"] = True
    save()
require(
    report.get("controls") == "pass" and report["restored"],
    "source controls incomplete",
)
report["verdict"] = "pass"
save()
print(
    json.dumps(
        {
            "verdict": "pass",
            "controls": 8,
            "source_restored": True,
            "head": identity["head"],
        }
    )
)
