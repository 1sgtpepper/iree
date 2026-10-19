# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare ordinary DPS version storage with the focused Sort cases."""

import hashlib
import json
from pathlib import Path
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

report["verdict"] = "pass" if all(case["oracle"] == "pass" for case in report["cases"]) else "fail"
save()
print(json.dumps({"verdict": report["verdict"], "case_count": len(report["cases"])}, indent=2))
sys.exit(0 if report["verdict"] == "pass" else 1)
