# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The command-gate tests must reject a deliberately restored log-and-continue gate."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
original = (root / "validation/41-fork-comparison-driver.py").read_text()
anchor = '        if row["verdict"] == "fail":'
if original.count(anchor) != 1:
    raise SystemExit("Actual command-gate mutation anchor changed")
path = root / "evidence/fail-open-driver-control.py"
path.write_text(original.replace(anchor, "        if False:"))
result = subprocess.run(
    [sys.executable, root / "validation/test_comparison_driver.py", path],
    capture_output=True,
    text=True,
)
(root / "evidence/harness-fail-open-control.log").write_text(
    result.stdout + result.stderr
)
if (
    result.returncode != 1
    or "wrong-output-oracle escaped the actual command gate" not in result.stderr
):
    raise SystemExit(
        "Fail-open control failed to establish the intended oracle-gate failure"
    )
print(
    "PASS: deliberately reintroduced fail-open behavior is rejected at wrong-output-oracle"
)

# A missing required setting must overwrite a pre-existing successful report.
import json
import os
import shutil
import tempfile

with tempfile.TemporaryDirectory() as temporary:
    probe_root = Path(temporary)
    report_path = probe_root / "sort-workspace-evidence/results.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps({"verdict": "pass"}))
    result = subprocess.run(
        [sys.executable, root / "validation/41-fork-comparison-driver.py"],
        env={"PATH": os.environ["PATH"], "RUNNER_TEMP": str(probe_root)},
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    if (
        result.returncode != 1
        or report["verdict"] != "fail"
        or "SOURCE_SHA" not in result.stderr
    ):
        raise SystemExit(
            "Missing-config control left a green report or failed for an unrelated reason"
        )
    (root / "evidence/harness-missing-config-control.log").write_text(result.stderr)
    print("PASS: missing driver configuration overwrites stale success")

with tempfile.TemporaryDirectory() as temporary:
    probe_root = Path(temporary)
    (probe_root / "validation").mkdir()
    (probe_root / "evidence").mkdir()
    for name in ("validate.sh", "record_run_status.py"):
        shutil.copyfile(root / "validation" / name, probe_root / "validation" / name)
    report_path = probe_root / "evidence/final-results.json"
    report_path.write_text(
        json.dumps({"verdict": "pass", "validation_id": "old-success"})
    )
    result = subprocess.run(
        ["bash", probe_root / "validation/validate.sh", "deliberate-startup-failure"],
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    if (
        result.returncode != 2
        or report["verdict"] != "fail"
        or report["validation_id"] == "old-success"
        or "arguments are unsupported" not in (result.stdout + result.stderr)
    ):
        raise SystemExit(
            "Shell startup-failure control left stale success or failed for an unrelated reason"
        )
    (root / "evidence/harness-startup-failure-control.log").write_text(
        result.stdout + json.dumps(report)
    )
    print("PASS: shell startup failure records a fresh failing invocation")
