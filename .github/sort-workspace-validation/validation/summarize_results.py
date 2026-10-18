# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""All required reports must agree with the restored final source identity."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
identity = json.loads((root / "validation/source-identity.json").read_text())
reports = {
    name: json.loads((root / "evidence" / name / "results.json").read_text())
    for name in ("exact-source", "source-controls", "restored-exact-source")
}
additional = json.loads(
    (root / "evidence/correctness-recheck/additional/results.json").read_text()
)
reports["additional"] = additional
for name in ("sort-workspace-paths", "restored-sort-workspace-paths"):
    reports[name] = json.loads((root / "evidence" / name / "results.json").read_text())
for name, report in reports.items():
    if report.get("validation_id") != os.environ.get("SORT_VALIDATION_RUN_ID"):
        raise SystemExit(
            name + ": stale required report from another validation invocation"
        )
    if report["verdict"] != "pass" or report["identity"] != identity:
        raise SystemExit(name + ": required identity/verdict failed")
required_phases = {
    "identity": 1,
    "compile": 8,
    "runtime": 102,
    "oracle": 34,
    "compiler test": 8,
    "runtime fixture": 26,
}
for name in ("exact-source", "restored-exact-source"):
    runs = reports[name]["runs"]
    if (
        len({row["name"] for row in runs}) != sum(required_phases.values())
        or {row["phase"] for row in runs} != required_phases.keys()
        or any(
            sum(row["phase"] == phase for row in runs) != count
            for phase, count in required_phases.items()
        )
    ):
        raise SystemExit(name + ": required primary phases are incomplete")
    if any(
        row["exit_status"] != row["expected_status"]
        or row["expected_status"] != (1 if row["name"] == "wrong-result-control" else 0)
        for row in runs
    ):
        raise SystemExit(name + ": required command status failed")
expected_controls = {
    "baseline-compile",
    "zero-initialization-oracle",
    "partial-sort-oracle",
    "borrowed-preservation-oracle",
    "old-paths-chain-compile",
    "old-paths-buffer-oracle",
    "local-initializer-compile",
    "constant-offset-tiling",
}
if {
    row["name"] for row in reports["source-controls"]["runs"] if "outcome" in row
} != expected_controls:
    raise SystemExit("Required source controls are incomplete")
required_control_commands = {
    "baseline-build",
    "baseline-compile",
    "zero-initialization-build",
    "zero-initialization-compile",
    "zero-initialization-run",
    "zero-initialization-oracle",
    "partial-sort-build",
    "partial-sort-compile",
    "partial-sort-run",
    "partial-sort-oracle",
    "borrowed-preservation-build",
    "borrowed-preservation-compile",
    "borrowed-preservation-run",
    "borrowed-preservation-oracle",
    "old-paths-build",
    "old-paths-chain-compile",
    "old-paths-buffer-transform",
    "old-paths-buffer-compile",
    "old-paths-buffer-runtime",
    "old-paths-buffer-oracle",
    "local-initializer-build",
    "local-initializer-compile",
    "constant-offset-build",
    "constant-offset-tiling",
    "restore-build",
}
control_runs = reports["source-controls"]["runs"]
if (
    len(control_runs) != 25
    or {row["name"] for row in control_runs} != required_control_commands
):
    raise SystemExit("Required control build/compile/runtime phases are incomplete")
if any(
    row["exit_status"] != (1 if row["name"] in expected_controls else 0)
    for row in control_runs
):
    raise SystemExit("Required control build/compile/runtime status failed")
if len(additional["commands"]) != 118 or any(
    row["status"] != 0 for row in additional["commands"]
):
    raise SystemExit("Required additional commands are incomplete or failed")
if len(additional["cases"]) != 116 or any(
    row["verdict"] != "pass" for row in additional["cases"]
):
    raise SystemExit("Required additional runtime cases are incomplete or incorrect")
for name in ("sort-workspace-paths", "restored-sort-workspace-paths"):
    report = reports[name]
    required_commands = {
        "compiler-version",
        "chains-compile",
        "earlier-compile",
        "buffer-partial-rejection",
        "earlier-literal",
        "earlier-literal-oracle",
        *[
            f"{family}-{device}-{index}"
            for device in ("local-sync", "local-task")
            for family, count in (("dependent", 8), ("local-update", 8), ("earlier", 4))
            for index in range(count)
        ],
    }
    required_cases = required_commands - {
        "compiler-version",
        "chains-compile",
        "earlier-compile",
        "buffer-partial-rejection",
        "earlier-literal",
        "earlier-literal-oracle",
    }
    if (
        len(report["commands"]) != 46
        or {row["name"] for row in report["commands"]} != required_commands
        or len(report["cases"]) != 40
        or {row["name"] for row in report["cases"]} != required_cases
    ):
        raise SystemExit(name + ": required workspace phases are incomplete")
    if any(
        row["exit_status"] != (1 if row["name"] == "buffer-partial-rejection" else 0)
        or row["expected_status"]
        != (1 if row["name"] == "buffer-partial-rejection" else 0)
        for row in report["commands"]
    ) or any(row["verdict"] != "pass" for row in report["cases"]):
        raise SystemExit(name + ": required workspace command or oracle failed")
source = Path(reports["restored-exact-source"]["source"])
if (
    subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    != identity["head"]
):
    raise SystemExit("Final source HEAD changed")
if subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain"], text=True
).strip():
    raise SystemExit("Final source is dirty")
actual_hash = hashlib.sha256(
    (root / "build/lib/libIREECompiler.so").read_bytes()
).hexdigest()
if actual_hash != reports["restored-exact-source"]["compiler_library_sha256"]:
    raise SystemExit("Final compiler library changed")
if actual_hash != additional["compiler_library_sha256"]:
    raise SystemExit("Additional cases used a different compiler library")
if any(
    reports[name]["compiler_library_sha256"] != actual_hash
    for name in ("sort-workspace-paths", "restored-sort-workspace-paths")
):
    raise SystemExit("Workspace cases used a different compiler library")
summary = {
    "validation_id": os.environ["SORT_VALIDATION_RUN_ID"],
    "verdict": "pass",
    "identity": identity,
    "compiler_library_sha256": actual_hash,
    "positive_commands": len(reports["restored-exact-source"]["runs"]),
    "additional_commands": len(additional["commands"]),
    "additional_runtime_cases": len(additional["cases"]),
    "workspace_runtime_cases": len(reports["restored-sort-workspace-paths"]["cases"]),
    "source_controls": [
        {"name": r["name"], "exit_status": r["exit_status"], "outcome": r["outcome"]}
        for r in reports["source-controls"]["runs"]
        if "outcome" in r
    ],
    "restored_exact_source": True,
}
(root / "evidence/final-results.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
