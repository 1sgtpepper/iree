# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Challenge the actual aggregator using the just-completed executable reports."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
names = (
    "exact-source",
    "source-controls",
    "restored-exact-source",
    "correctness-recheck/additional",
    "sort-workspace-paths",
    "restored-sort-workspace-paths",
)
reports = {
    name: json.loads((root / "evidence" / name / "results.json").read_text())
    for name in names
}
results = []


def exercise(name, report_name=None, change=None, reason=None):
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        (fixture / "validation").mkdir()
        shutil.copy(root / "validation/summarize_results.py", fixture / "validation")
        shutil.copy(root / "validation/source-identity.json", fixture / "validation")
        (fixture / "build").symlink_to(root / "build", target_is_directory=True)
        for key, report in reports.items():
            data = json.loads(json.dumps(report))
            if key == report_name:
                change(data)
            path = fixture / "evidence" / key / "results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        result = subprocess.run(
            [sys.executable, fixture / "validation/summarize_results.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        summary = fixture / "evidence/final-results.json"
        if reason is None:
            if (
                result.returncode != 0
                or json.loads(summary.read_text())["verdict"] != "pass"
            ):
                raise RuntimeError(name + ": valid complete reports were rejected")
        elif result.returncode != 1 or reason not in result.stderr or summary.exists():
            raise RuntimeError(
                name + ": incomplete or failed reports escaped their required gate"
            )
        results.append(
            {
                "name": name,
                "exit_status": result.returncode,
                "reason": result.stderr.strip(),
            }
        )


exercise("complete-reports")
for name in ("exact-source", "restored-exact-source"):
    exercise(
        name + "-smoke-only",
        name,
        lambda report: report.update(
            runs=[
                row
                for row in report["runs"]
                if row["name"]
                in {"compiler-version", "original-compile", "original-reproducer"}
            ]
        ),
        name + ": required primary phases are incomplete",
    )
exercise(
    "failed-primary-runtime",
    "exact-source",
    lambda report: next(
        row for row in report["runs"] if row["name"] == "original-reproducer"
    ).update(exit_status=1),
    "exact-source: required command status failed",
)
exercise(
    "missing-control-build",
    "source-controls",
    lambda report: report.update(
        runs=[row for row in report["runs"] if row["name"] != "baseline-build"]
    ),
    "Required control build/compile/runtime phases are incomplete",
)
exercise(
    "failed-control-build",
    "source-controls",
    lambda report: next(
        row for row in report["runs"] if row["name"] == "baseline-build"
    ).update(exit_status=1),
    "Required control build/compile/runtime status failed",
)
exercise(
    "missing-additional-command",
    "correctness-recheck/additional",
    lambda report: report["commands"].pop(),
    "Required additional commands are incomplete or failed",
)
exercise(
    "failed-additional-command",
    "correctness-recheck/additional",
    lambda report: report["commands"][0].update(status=1),
    "Required additional commands are incomplete or failed",
)
exercise(
    "missing-workspace-command",
    "sort-workspace-paths",
    lambda report: report["commands"].pop(),
    "sort-workspace-paths: required workspace phases are incomplete",
)
exercise(
    "failed-workspace-oracle",
    "sort-workspace-paths",
    lambda report: report["cases"][0].update(verdict="fail"),
    "sort-workspace-paths: required workspace command or oracle failed",
)
exercise(
    "missing-restored-workspace-case",
    "restored-sort-workspace-paths",
    lambda report: report["cases"].pop(),
    "restored-sort-workspace-paths: required workspace phases are incomplete",
)
print(json.dumps({"verdict": "pass", "checks": results}, indent=2))
