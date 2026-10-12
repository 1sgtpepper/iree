# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise the driver's actual command gate with subprocesses and fresh BEP."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

path = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).with_name("41-fork-comparison-driver.py")
)
spec = importlib.util.spec_from_file_location("comparison_driver", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
main = next(
    node
    for node in ast.parse(path.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "main"
)
functions = [
    node
    for node in main.body
    if isinstance(node, ast.FunctionDef) and node.name in ("command", "save_report")
]
if len(functions) != 2:
    raise RuntimeError("Actual command gate was not found")
with tempfile.TemporaryDirectory() as temp:
    out = Path(temp)
    subprocess.run(["git", "init", "-q", str(out)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(out),
            "-c",
            "user.name=Local Tester",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "Initialize isolated harness test",
        ],
        check=True,
    )
    cpp = out / "converter.cpp"
    cpp.write_text("converter source\n")
    bufferization_cpp = out / "bufferization.cpp"
    bufferization_cpp.write_text("bufferization source\n")
    namespace = {
        "out": out,
        "cpp": cpp,
        "bufferization_cpp": bufferization_cpp,
        "report": {"runs": []},
        "subprocess": subprocess,
        "hashlib": hashlib,
        "json": json,
        "Path": Path,
        "shutil": shutil,
        "judge": module.judge,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    command = namespace["command"]
    # command() takes source identities from the checkout; use only this temp repo.
    import os

    previous = Path.cwd()
    os.chdir(out)
    try:
        checks = []

        def require_rejection(name, args, **kwargs):
            try:
                command(name, args, **kwargs)
            except SystemExit:
                row = namespace["report"]["runs"][-1]
                if row["verdict"] != "fail":
                    raise RuntimeError(str(row))
                checks.append(
                    {
                        "name": name,
                        "gate": "rejected",
                        "classification": row["classification"],
                    }
                )
            else:
                raise AssertionError(name + " escaped the actual command gate")

        command("correct-result", [sys.executable, "-c", 'print("1 2 0")'])
        actual = out / "wrong-output.txt"
        actual.write_text("0 1 2\n")
        require_rejection(
            "wrong-output-oracle",
            [
                sys.executable,
                "-c",
                'import pathlib,sys; sys.exit(0 if pathlib.Path(sys.argv[1]).read_text()=="1 2 0\\n" else 1)',
                str(actual),
            ],
            phase="oracle",
        )
        require_rejection(
            "unintended-failure",
            [sys.executable, "-c", 'import sys; print("unknown flag"); sys.exit(1)'],
            expected="semantic-failure",
            reason="wrong indices",
        )
        require_rejection(
            "oracle-wrong-reason",
            [sys.executable, "-c", 'import sys; print("unknown option"); sys.exit(1)'],
            expected="semantic-failure",
            reason="wrong indices",
            phase="oracle",
        )
        require_rejection(
            "build-wrong-reason",
            [sys.executable, "-c", 'import sys; print("wrong indices"); sys.exit(1)'],
            expected="semantic-failure",
            reason="wrong indices",
            phase="build",
        )
        require_rejection(
            "escaped-control",
            [sys.executable, "-c", 'print("success")'],
            expected="semantic-failure",
            reason="wrong indices",
        )
        for name, status, built, summary in [
            ("build-failure", 1, False, None),
            ("skipped", 0, True, "SKIPPED"),
            ("timeout", 3, True, "TIMEOUT"),
        ]:
            events = [
                {
                    "id": {"targetCompleted": {"label": "//case:probe.test"}},
                    "completed": {"success": built},
                }
            ]
            if summary:
                events.append(
                    {
                        "id": {"testSummary": {"label": "//case:probe.test"}},
                        "testSummary": {"overallStatus": summary},
                    }
                )
            code = (
                'import json,pathlib,sys; path=next(a.split("=",1)[1] for a in sys.argv[1:] if a.startswith("--build_event_json_file=")); '
                'pathlib.Path(path).write_text("\\n".join(json.dumps(e) for e in '
                + repr(events)
                + ")); "
                'print("wrong indices"); sys.exit(' + str(status) + ")"
            )
            require_rejection(
                name,
                [sys.executable, "-c", code],
                targets=["//case:probe.test"],
                expected="semantic-failure" if status else "pass",
                reason="wrong indices",
            )
        events = [
            {
                "id": {"targetCompleted": {"label": "//case:probe.test"}},
                "completed": {"success": True},
            },
            {
                "id": {"testSummary": {"label": "//case:probe.test"}},
                "testSummary": {"overallStatus": "FAILED"},
            },
        ]
        code = (
            'import json,pathlib,sys; path=next(a.split("=",1)[1] for a in sys.argv[1:] if a.startswith("--build_event_json_file=")); '
            'pathlib.Path(path).write_text("\\n".join(json.dumps(e) for e in '
            + repr(events)
            + ")); "
            'log=pathlib.Path("bazel-testlogs/case/probe.test/test.log"); log.parent.mkdir(parents=True,exist_ok=True); '
            'log.write_text("wrong indices"); sys.exit(3)'
        )
        command(
            "intended-test-failure",
            [sys.executable, "-c", code],
            targets=["//case:probe.test"],
            expected="semantic-failure",
            reason="wrong indices",
        )
        checks.append(
            {
                "name": "intended-test-failure",
                "gate": "accepted for matching per-target semantic diagnostic",
            }
        )
        # A stale matching log and fresh unrelated failure must not be accepted.
        require_rejection(
            "stale-diagnostic",
            [
                sys.executable,
                "-c",
                code.replace(
                    'log.write_text("wrong indices")', 'print("unrelated failure")'
                ),
            ],
            targets=["//case:probe.test"],
            expected="semantic-failure",
            reason="wrong indices",
        )
        print(json.dumps({"verdict": "pass", "command_gate_checks": checks}, indent=2))
    finally:
        os.chdir(previous)
