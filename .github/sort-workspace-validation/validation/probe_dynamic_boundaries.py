# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execute two dynamic Sort ownership boundaries against the built exact head."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
source = root.parent / "source"
build = root / "build"
out = root / "evidence/dynamic-boundaries"
out.mkdir(parents=True, exist_ok=True)
identity = json.loads((root / "validation/source-identity.json").read_text())
previous = json.loads((root / "evidence/final-results.json").read_text())
record = {"verdict": "running", "identity": identity, "cases": {}, "commands": []}


def save():
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name, argv, timeout=180):
    argv = [str(arg) for arg in argv]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        status = completed.returncode
        output = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = (exc.stdout or b"").decode(errors="replace") + (exc.stderr or b"").decode(errors="replace")
    (out / (name + ".log")).write_text(output)
    record["commands"].append({"name": name, "argv": argv, "exit_status": status})
    save()
    return status, output


save()
head = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
tree = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD^{tree}"], text=True).strip()
clean = not subprocess.check_output(["git", "-C", source, "status", "--porcelain"], text=True).strip()
library = build / "lib/libIREECompiler.so"
record["compiler_library_sha256"] = sha(library)
if not (
    previous.get("verdict") == "pass"
    and previous.get("identity") == identity
    and head == identity["head"]
    and tree == identity["tree"]
    and clean
    and previous.get("compiler_library_sha256") == record["compiler_library_sha256"]
):
    record.update(verdict="fail", failure="exact-source build or restored source identity was not admitted")
    save()
    sys.exit(1)

compiler = build / "tools/iree-compile"
runtime = build / "tools/iree-run-module"
record["compiler_sha256"] = sha(compiler)
fixtures = {
    "local-shared-scalar": root / "validation/fixtures/local_shared_scalar_observed_keys.mlir",
    "dependent-observed": root / "validation/fixtures/dependent_observed_dynamic.mlir",
}
record["fixture_sha256"] = {name: sha(path) for name, path in fixtures.items()}
arrays = [
    np.array([], dtype=np.int64),
    np.array([7], dtype=np.int64),
    np.array([3, 1, 2], dtype=np.int64),
    np.array([2, 1, 2, 0], dtype=np.int64),
    np.array([-(2**63), 2**63 - 1, 0, -1, 0], dtype=np.int64),
    np.array([((i * 37) % 23) - 11 for i in range(65)], dtype=np.int64),
]
for name, fixture in fixtures.items():
    case = record["cases"][name] = {"compile": "pending", "runtime": []}
    module = out / (name + ".vmfb")
    module.unlink(missing_ok=True)
    status, diagnostic = run(
        name + "-compile",
        [
            compiler,
            fixture,
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
            "--mlir-disable-threading",
            "--mlir-print-ir-before=iree-dispatch-creation-convert-dispatch-regions-to-workgroups",
            "--mlir-print-ir-after=iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-codegen-iree-comprehensive-bufferize",
            "-o",
            module,
        ],
    )
    case["compile"] = status
    case["has_conversion_ir"] = "IR Dump After ConvertDispatchRegionsToWorkgroupsPass" in diagnostic
    case["has_bufferization_ir"] = "IR Dump After IREEComprehensiveBufferizePass" in diagnostic
    if status != 0 or not module.is_file() or not case["has_conversion_ir"] or not case["has_bufferization_ir"]:
        save()
        continue
    for device in ("local-sync", "local-task"):
        for number, values in enumerate(arrays):
            labels = np.arange(values.size, dtype=np.int64) * 17 - 13
            if name == "local-shared-scalar":
                inputs = [labels]
                expected = [
                    labels[::-1],
                    np.arange(values.size, 0, -1, dtype=np.int64),
                    np.full(values.size, values.size, dtype=np.int64),
                    labels,
                ]
                function = "local_shared_scalar_observed_keys"
            else:
                inputs = [values, labels]
                order = sorted(range(values.size), key=values.__getitem__, reverse=True)
                expected = [
                    labels[order],
                    np.sort(values, kind="stable"),
                    values,
                    labels,
                ]
                function = "dependent_observed_dynamic"
            prefix = f"{name}-{device}-{number}"
            paths = []
            for index, value in enumerate(inputs):
                path = out / f"{prefix}-input-{index}.npy"
                np.save(path, value)
                paths.append("--input=@" + str(path))
            outputs = []
            for index in range(len(expected)):
                path = out / f"{prefix}-output-{index}.npy"
                path.unlink(missing_ok=True)
                outputs.append(path)
            argv = [runtime, "--module=" + str(module), "--device=" + device, "--function=" + function]
            if device == "local-task":
                argv.append("--task_topology_group_count=2")
            argv.extend(paths)
            argv.extend("--output=@" + str(path) for path in outputs)
            status, _ = run(prefix, argv)
            match = status == 0 and all(
                path.is_file()
                and (actual := np.load(path, allow_pickle=False)).dtype == wanted.dtype
                and actual.shape == wanted.shape
                and np.array_equal(actual, wanted)
                for path, wanted in zip(outputs, expected)
            )
            case["runtime"].append({"name": prefix, "exit_status": status, "oracle": "pass" if match else "fail"})
            save()
record["verdict"] = "pass" if all(
    case["compile"] == 0
    and case["has_conversion_ir"]
    and case["has_bufferization_ir"]
    and len(case["runtime"]) == 12
    and all(row["oracle"] == "pass" for row in case["runtime"])
    for case in record["cases"].values()
) else "fail"
save()
print(json.dumps({"verdict": record["verdict"], "cases": record["cases"]}, indent=2))
sys.exit(0 if record["verdict"] == "pass" else 1)
