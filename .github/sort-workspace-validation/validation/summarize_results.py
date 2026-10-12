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
for name, report in reports.items():
    if report.get("validation_id") != os.environ.get("SORT_VALIDATION_RUN_ID"):
        raise SystemExit(
            name + ": stale required report from another validation invocation"
        )
    if report["verdict"] != "pass" or report["identity"] != identity:
        raise SystemExit(name + ": required identity/verdict failed")
expected_controls = {
    "baseline-compile",
    "zero-initialization-oracle",
    "partial-sort-oracle",
    "borrowed-preservation-oracle",
}
if {
    row["name"] for row in reports["source-controls"]["runs"] if "outcome" in row
} != expected_controls:
    raise SystemExit("Required source controls are incomplete")
if len(additional["cases"]) != 116 or any(
    row["verdict"] != "pass" for row in additional["cases"]
):
    raise SystemExit("Required additional runtime cases are incomplete or incorrect")
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
summary = {
    "validation_id": os.environ["SORT_VALIDATION_RUN_ID"],
    "verdict": "pass",
    "identity": identity,
    "compiler_library_sha256": actual_hash,
    "positive_commands": len(reports["restored-exact-source"]["runs"]),
    "additional_commands": len(additional["commands"]),
    "additional_runtime_cases": len(additional["cases"]),
    "source_controls": [
        {"name": r["name"], "exit_status": r["exit_status"], "outcome": r["outcome"]}
        for r in reports["source-controls"]["runs"]
        if "outcome" in r
    ],
    "restored_exact_source": True,
}
(root / "evidence/final-results.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
