# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Overwrite stale success before starting, and record any failed shell phase."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
report = {"verdict": sys.argv[1], "validation_id": sys.argv[2]}
if len(sys.argv) > 3:
    report["exit_status"] = int(sys.argv[3])
    report["failure"] = (
        "A required validation phase failed; inspect final-validation.log and per-phase logs"
    )
(root / "evidence/final-results.json").write_text(json.dumps(report, indent=2) + "\n")
