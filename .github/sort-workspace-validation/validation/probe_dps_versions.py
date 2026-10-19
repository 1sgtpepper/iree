# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare ordinary DPS version storage with the focused Sort cases."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
source = root.parent / "source"
build = root / "build"
out = root / "evidence/dps-versions"
out.mkdir(parents=True, exist_ok=True)
identity = json.loads((root / "validation/source-identity.json").read_text())
admitted = json.loads((root / "evidence/final-results.json").read_text())
report = {"verdict": "running", "identity": identity, "commands": [], "cases": []}


def save():
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name, argv):
    argv = [str(value) for value in argv]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    (out / (name + ".log")).write_text(result.stdout + result.stderr)
    report["commands"].append({"name": name, "argv": argv, "status": result.returncode})
    save()
    return result


save()
library = build / "lib/libIREECompiler.so"
head = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
tree = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD^{tree}"], text=True).strip()
clean = not subprocess.check_output(["git", "-C", source, "status", "--porcelain"], text=True).strip()
report["compiler_library_sha256"] = sha(library)
if not (
    admitted.get("verdict") == "pass"
    and admitted.get("identity") == identity
    and admitted.get("compiler_library_sha256") == report["compiler_library_sha256"]
    and head == identity["head"]
    and tree == identity["tree"]
    and clean
):
    report.update(verdict="fail", failure="exact-source compiler was not admitted")
    save()
    sys.exit(1)

fixture = root / "validation/fixtures/dps_observed_versions.mlir"
report["fixture_sha256"] = sha(fixture)
compiler = build / "tools/iree-compile"
runtime = build / "tools/iree-run-module"
module = out / "dps_observed_versions.vmfb"
module.unlink(missing_ok=True)
compiled = run(
    "compile",
    [
        compiler,
        fixture,
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=generic",
        "--mlir-disable-threading",
        "--mlir-print-ir-before=iree-dispatch-creation-convert-dispatch-regions-to-workgroups",
        "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-stream-materialize-copy-on-write,iree-stream-schedule-allocation,iree-codegen-iree-comprehensive-bufferize",
        "-o",
        module,
    ],
)
report["compile_status"] = compiled.returncode
report["has_conversion_ir"] = "IR Dump After ConvertDispatchRegionsToWorkgroupsPass" in compiled.stderr
report["has_stream_ir"] = "stream.cmd.dispatch" in compiled.stderr
report["has_bufferization_ir"] = "IR Dump After IREEComprehensiveBufferizePass" in compiled.stderr
if not (
    compiled.returncode == 0
    and module.is_file()
    and report["has_conversion_ir"]
    and report["has_stream_ir"]
    and report["has_bufferization_ir"]
):
    report.update(verdict="fail", failure="DPS control did not compile with all required IR stages")
    save()
    sys.exit(1)

dependent_before = compiled.stderr.split("util.func public @dependent_observed", 1)[1]
dependent_before = dependent_before.split("IR Dump After ConvertDispatchRegionsToWorkgroupsPass:", 1)[0]
dependent_region_start = dependent_before.find("flow.dispatch.region")
dependent_region_end = dependent_before.find("flow.return", dependent_region_start)
dependent_region = (
    dependent_before[dependent_region_start:dependent_region_end]
    if dependent_region_start >= 0 and dependent_region_end >= 0
    else ""
)
report["dependent_control_fused"] = dependent_region.count("linalg.generic") == 1

values_list = [
    np.array([], dtype=np.int64),
    np.array([7], dtype=np.int64),
    np.array([3, 1, 2, 1], dtype=np.int64),
    np.array([-(2**63), 2**63 - 2, 0], dtype=np.int64),
]
for function, increments in (("single_observed", (1, 0)), ("dependent_observed", (2, 1, 0))):
    for device in ("local-sync", "local-task"):
        for index, values in enumerate(values_list):
            name = f"{function}-{device}-{index}"
            input_path = out / (name + "-input.npy")
            np.save(input_path, values)
            output_paths = [out / f"{name}-output-{slot}.npy" for slot in range(len(increments))]
            for path in output_paths:
                path.unlink(missing_ok=True)
            argv = [runtime, "--module=" + str(module), "--device=" + device, "--function=" + function]
            if device == "local-task":
                argv.append("--task_topology_group_count=2")
            argv.append("--input=@" + str(input_path))
            argv.extend("--output=@" + str(path) for path in output_paths)
            result = run(name, argv)
            expected = [values + amount for amount in increments]
            actual = (
                [np.load(path, allow_pickle=False) for path in output_paths]
                if result.returncode == 0 and all(path.is_file() for path in output_paths)
                else []
            )
            passed = len(actual) == len(expected) and all(
                got.dtype == want.dtype
                and got.shape == want.shape
                and np.array_equal(got, want)
                for got, want in zip(actual, expected)
            )
            report["cases"].append({"name": name, "status": result.returncode, "oracle": "pass" if passed else "fail"})
            save()

matmul_fixture = root / "validation/fixtures/dps_matmul_observed_version.mlir"
report["matmul_fixture_sha256"] = sha(matmul_fixture)
matmul_module = out / "dps_matmul_observed_version.vmfb"
matmul_module.unlink(missing_ok=True)
matmul_compiled = run(
    "matmul-compile",
    [
        compiler,
        matmul_fixture,
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=generic",
        "--mlir-disable-threading",
        "--mlir-print-ir-before=iree-dispatch-creation-convert-dispatch-regions-to-workgroups",
        "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-stream-materialize-copy-on-write,iree-stream-schedule-allocation,iree-codegen-iree-comprehensive-bufferize",
        "-o",
        matmul_module,
    ],
)
report["matmul_compile_status"] = matmul_compiled.returncode
report["matmul_has_conversion_ir"] = "IR Dump After ConvertDispatchRegionsToWorkgroupsPass" in matmul_compiled.stderr
report["matmul_has_stream_ir"] = "stream.cmd.dispatch" in matmul_compiled.stderr
report["matmul_has_bufferization_ir"] = "IR Dump After IREEComprehensiveBufferizePass" in matmul_compiled.stderr
if not (
    matmul_compiled.returncode == 0
    and matmul_module.is_file()
    and report["matmul_has_conversion_ir"]
    and report["matmul_has_stream_ir"]
    and report["matmul_has_bufferization_ir"]
):
    report.update(verdict="inconclusive", failure="non-fusible DPS control did not compile with all required IR stages")
    save()
    sys.exit(1)

before_conversion = matmul_compiled.stderr.split("IR Dump Before ConvertDispatchRegionsToWorkgroupsPass:", 1)[1]
before_conversion = before_conversion.split("IR Dump After ConvertDispatchRegionsToWorkgroupsPass:", 1)[0]
region_start = before_conversion.find("flow.dispatch.region")
region_end = before_conversion.find("flow.return", region_start)
region = before_conversion[region_start:region_end] if region_start >= 0 and region_end >= 0 else ""
matmul_updates = re.findall(
    r"(?m)^\s*(%\d+)\s*=\s*linalg\.matmul\b[^\n]*\bouts\((%\d+)\s*:", region
)
report["matmul_has_separate_updates"] = len(matmul_updates) == 2
report["matmul_retains_dependent_init"] = (
    report["matmul_has_separate_updates"] and matmul_updates[1][1] == matmul_updates[0][0]
)
report["matmul_returns_both_versions"] = bool(
    report["matmul_retains_dependent_init"]
    and re.search(
        r"\bflow\.return\s+" + re.escape(matmul_updates[1][0])
        + r"\s*,\s*" + re.escape(matmul_updates[0][0]) + r"\b",
        before_conversion,
    )
)
if not report["matmul_returns_both_versions"]:
    report.update(verdict="inconclusive", failure="pre-conversion IR did not retain a dependent init and both versions")
    save()
    sys.exit(1)

matrices = [
    np.empty((0, 0), dtype=np.float32),
    np.array([[2.5]], dtype=np.float32),
    np.array([[1.0, -3.0], [2.0, 4.0]], dtype=np.float32),
    np.array([[0.0, 1.0, -2.0], [3.0, -4.0, 5.0], [6.0, 0.5, -1.0]], dtype=np.float32),
]
for device in ("local-sync", "local-task"):
    for index, values in enumerate(matrices):
        name = f"dependent_matmul_observed-{device}-{index}"
        input_path = out / f"{name}-input.npy"
        np.save(input_path, values)
        output_paths = [out / f"{name}-output-{slot}.npy" for slot in range(3)]
        for path in output_paths:
            path.unlink(missing_ok=True)
        argv = [runtime, "--module=" + str(matmul_module), "--device=" + device,
                "--function=dependent_matmul_observed"]
        if device == "local-task":
            argv.append("--task_topology_group_count=2")
        argv.append("--input=@" + str(input_path))
        argv.extend("--output=@" + str(path) for path in output_paths)
        result = run(name, argv)
        first = values @ values
        expected = [first + values @ first, first, values]
        actual = (
            [np.load(path, allow_pickle=False) for path in output_paths]
            if result.returncode == 0 and all(path.is_file() for path in output_paths)
            else []
        )
        passed = len(actual) == len(expected) and all(
            got.dtype == want.dtype
            and got.shape == want.shape
            and np.allclose(got, want, rtol=1e-5, atol=1e-5, equal_nan=False)
            for got, want in zip(actual, expected)
        )
        report["cases"].append({"name": name, "status": result.returncode, "oracle": "pass" if passed else "fail"})
        save()

scatter_fixture = root / "validation/fixtures/linalg_ext_scatter_observed_input.mlir"
report["scatter_fixture_sha256"] = sha(scatter_fixture)
scatter_module = out / "linalg_ext_scatter_observed_input.vmfb"
scatter_module.unlink(missing_ok=True)
scatter_compiled = run(
    "scatter-compile",
    [
        compiler,
        scatter_fixture,
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=generic",
        "--mlir-disable-threading",
        "--mlir-print-ir-before=iree-dispatch-creation-convert-dispatch-regions-to-workgroups",
        "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-stream-materialize-copy-on-write,iree-stream-schedule-allocation,iree-codegen-iree-comprehensive-bufferize",
        "-o",
        scatter_module,
    ],
)
report["scatter_compile_status"] = scatter_compiled.returncode
report["scatter_has_op"] = "iree_linalg_ext.scatter" in scatter_compiled.stderr
report["scatter_has_conversion_ir"] = "IR Dump After ConvertDispatchRegionsToWorkgroupsPass" in scatter_compiled.stderr
report["scatter_has_stream_ir"] = "stream.cmd.dispatch" in scatter_compiled.stderr
report["scatter_has_bufferization_ir"] = "IR Dump After IREEComprehensiveBufferizePass" in scatter_compiled.stderr
if not (
    scatter_compiled.returncode == 0
    and scatter_module.is_file()
    and report["scatter_has_op"]
    and report["scatter_has_conversion_ir"]
    and report["scatter_has_stream_ir"]
    and report["scatter_has_bufferization_ir"]
):
    report.update(verdict="inconclusive", failure="Scatter control did not compile with all required IR stages")
    save()
    sys.exit(1)

scatter_inputs = [
    (np.arange(8, dtype=np.int32), np.array([100, 200, 300], dtype=np.int32),
     np.array([[1], [3], [5]], dtype=np.int32)),
    (np.array([-(2**31), 3, -4, 9, 0, 4, 5, 2**31 - 1], dtype=np.int32),
     np.array([-7, 0, 11], dtype=np.int32), np.array([[0], [7], [4]], dtype=np.int32)),
]
for device in ("local-sync", "local-task"):
    for index, (original, updates, indices) in enumerate(scatter_inputs):
        name = f"scatter_observed-{device}-{index}"
        input_paths = [out / f"{name}-input-{slot}.npy" for slot in range(3)]
        for path, values in zip(input_paths, (original, updates, indices)):
            np.save(path, values)
        output_paths = [out / f"{name}-output-{slot}.npy" for slot in range(2)]
        for path in output_paths:
            path.unlink(missing_ok=True)
        argv = [runtime, "--module=" + str(scatter_module), "--device=" + device,
                "--function=scatter_observed"]
        if device == "local-task":
            argv.append("--task_topology_group_count=2")
        argv.extend("--input=@" + str(path) for path in input_paths)
        argv.extend("--output=@" + str(path) for path in output_paths)
        result = run(name, argv)
        scattered = original.copy()
        scattered[indices[:, 0]] = updates
        expected = [scattered, original]
        actual = (
            [np.load(path, allow_pickle=False) for path in output_paths]
            if result.returncode == 0 and all(path.is_file() for path in output_paths)
            else []
        )
        passed = len(actual) == len(expected) and all(
            got.dtype == want.dtype and got.shape == want.shape and np.array_equal(got, want)
            for got, want in zip(actual, expected)
        )
        report["cases"].append({"name": name, "status": result.returncode, "oracle": "pass" if passed else "fail"})
        save()

report["verdict"] = "pass" if all(case["oracle"] == "pass" for case in report["cases"]) else "fail"
save()
print(json.dumps({"verdict": report["verdict"], "case_count": len(report["cases"])}, indent=2))
sys.exit(0 if report["verdict"] == "pass" else 1)
