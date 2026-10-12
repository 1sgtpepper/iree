# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fork-only differential validation, including a smaller clone experiment."""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def judge(status, output, expected="pass", reason=None, tests=None, phase="execution"):
    """A failed build or skipped test never establishes a semantic control."""
    if tests is not None:
        if status == 1 or any(not test["built"] for test in tests):
            classification = "build-failure"
        elif status == 4 or not tests or any(not test["statuses"] for test in tests):
            classification = "skipped-or-inconclusive"
        elif any(test["statuses"] not in (["PASSED"], ["FAILED"]) for test in tests):
            classification = "skipped-or-inconclusive"
        elif status == 0 and all(test["statuses"] == ["PASSED"] for test in tests):
            classification = "pass"
        elif status == 3 and all(test["statuses"] == ["FAILED"] for test in tests):
            classification = (
                "semantic-failure"
                if reason
                and all(re.search(reason, test.get("diagnostic", "")) for test in tests)
                else "unclassified-test-failure"
            )
        else:
            classification = "inconsistent-test-result"
    elif status == 0:
        classification = "pass"
    elif phase == "build":
        classification = "build-failure"
    elif status == 1 and reason and re.search(reason, output):
        classification = "semantic-failure"
    elif status == 1 and phase == "oracle" and expected == "pass":
        classification = "semantic-failure"
    else:
        classification = "execution-failure"
    if expected == "observe":
        verdict = "inconclusive"
    else:
        verdict = "pass" if classification == expected else "fail"
    return {"classification": classification, "verdict": verdict}


def self_test():
    # Real subprocess statuses challenge the gate, independently of IREE/Bazel.
    cases = [
        ("positive", 0, "correct", "pass", None, None, "pass"),
        ("wrong-output", 1, "wrong indices", "pass", None, None, "fail"),
        (
            "control-escaped",
            0,
            "correct",
            "semantic-failure",
            "wrong indices",
            None,
            "fail",
        ),
        (
            "intended-failure",
            1,
            "wrong indices",
            "semantic-failure",
            "wrong indices",
            None,
            "pass",
        ),
        (
            "wrong-failure-reason",
            1,
            "unknown option",
            "semantic-failure",
            "wrong indices",
            None,
            "fail",
        ),
        (
            "build-failure",
            1,
            "wrong indices",
            "semantic-failure",
            "wrong indices",
            [{"built": False, "statuses": []}],
            "fail",
        ),
        (
            "skipped-test",
            0,
            "correct",
            "pass",
            None,
            [{"built": True, "statuses": ["SKIPPED"]}],
            "fail",
        ),
        (
            "missing-test",
            0,
            "correct",
            "pass",
            None,
            [{"built": True, "statuses": []}],
            "fail",
        ),
        (
            "timeout-control",
            3,
            "wrong indices",
            "semantic-failure",
            "wrong indices",
            [{"built": True, "statuses": ["TIMEOUT"]}],
            "fail",
        ),
        (
            "bazel-control",
            3,
            "wrong indices",
            "semantic-failure",
            "wrong indices",
            [{"built": True, "statuses": ["FAILED"], "diagnostic": "wrong indices"}],
            "pass",
        ),
        (
            "mixed-control",
            3,
            "wrong indices",
            "semantic-failure",
            "wrong indices",
            [
                {"built": True, "statuses": ["FAILED"]},
                {"built": True, "statuses": ["PASSED"]},
            ],
            "fail",
        ),
        ("unknown-experiment", 1, "unknown", "observe", None, None, "inconclusive"),
    ]
    rows = []
    for name, status, output, expected, reason, tests, verdict in cases:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); sys.exit(int(sys.argv[2]))",
                output,
                str(status),
            ],
            capture_output=True,
            text=True,
        )
        result = judge(process.returncode, process.stdout, expected, reason, tests)
        if result["verdict"] != verdict:
            raise SystemExit(f"{name}: {result} != {verdict}")
        rows.append({"name": name, **result})
    print(json.dumps({"self_test": "pass", "cases": rows}, indent=2))


def main():
    source = os.environ["SOURCE_SHA"]
    base = os.environ["BASE_SHA"]
    submitted = os.environ["SUBMITTED_SHA"]
    llvm = os.environ["LLVM_SHA"]
    out = Path(os.environ["RUNNER_TEMP"]) / "sort-workspace-evidence"
    out.mkdir(exist_ok=True)
    report = {
        "candidate": source,
        "base": base,
        "submitted": submitted,
        "llvm": llvm,
        "runs": [],
        "fixture_hashes": {},
    }
    cpp = Path(
        "compiler/src/iree/compiler/DispatchCreation/ConvertDispatchRegionsToWorkgroups.cpp"
    )
    bufferization_cpp = Path(
        "compiler/src/iree/compiler/Codegen/Interfaces/BufferizationInterfaces.cpp"
    )
    fixtures = [
        "tests/e2e/regression/" + name + ".mlir"
        for name in (
            "argsort_dynamic",
            "argsort_preserve_input_dynamic",
            "argsort_comparator_lanes_dynamic",
            "argsort_views_dynamic",
            "argsort_internal_key_use",
            "argsort_local_keys",
            "two_sorts_original_version",
        )
    ]
    registrations = [
        "tests/e2e/regression/BUILD.bazel",
        "tests/e2e/regression/CMakeLists.txt",
    ]
    # Bazel target separators differ from repository file paths.
    runtime = [
        "//tests/e2e/regression:" + Path(path).name + ".test" for path in fixtures
    ]
    compiler = [
        "//compiler/src/iree/compiler/Dialect/LinalgExt/IR/test:canonicalize.mlir.test",
        "//compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test:tiling.mlir.test",
        "//compiler/src/iree/compiler/Dialect/Flow/IR/test:dispatch_folding.mlir.test",
        "//compiler/src/iree/compiler/DispatchCreation/test:convert_region_to_workgroups.mlir.test",
        "//compiler/src/iree/compiler/DispatchCreation/test:dispatch_linalg_on_tensors.mlir.test",
        "//compiler/src/iree/compiler/DispatchCreation/test:form_scalar_dispatches.mlir.test",
        "//compiler/src/iree/compiler/Codegen/Common/test:iree_comprehensive_bufferize.mlir.test",
    ]
    bazel = ["bazel", "--noworkspace_rc", "--bazelrc=build_tools/bazel/iree.bazelrc"]
    flags = [
        "--sandbox_base=" + os.environ["SANDBOX_BASE"],
        "--disk_cache=" + os.environ["BAZEL_DISK_CACHE"],
        "--config=generic_clang_ci",
        "--noremote_upload_local_results",
        "--iree_compiler_plugins=hal_target_llvm_cpu,hal_target_local",
        "--iree_drivers=local-task",
        "--jobs=4",
        "--local_cpu_resources=4",
    ]

    def save_report():
        report["verdict"] = (
            "fail"
            if report.get("failure")
            else (
                "pass"
                if report.get("complete")
                and all(r["verdict"] == "pass" for r in report["runs"])
                else (
                    "fail"
                    if any(r["verdict"] == "fail" for r in report["runs"])
                    else (
                        "inconclusive"
                        if any(r["verdict"] == "inconclusive" for r in report["runs"])
                        else "running"
                    )
                )
            )
        )
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def require(condition, message):
        if not condition:
            report["failure"] = message
            report["classification"] = "preparation-failure"
            save_report()
            raise SystemExit(message)

    def command(
        name, args, targets=(), expected="pass", reason=None, phase="execution"
    ):
        if "bazel" in args[0] and "build" in args and not targets:
            phase = "build"
        print("Running " + name + ": " + " ".join(args), flush=True)
        log = out / (name + ".log")
        events = out / (name + "-events.jsonl")
        events.unlink(missing_ok=True)
        for target in targets:
            package, test = target[2:].split(":")
            (Path("bazel-testlogs") / package / test / "test.log").unlink(
                missing_ok=True
            )
        if targets:
            args = list(args) + ["--build_event_json_file=" + str(events)]
        with log.open("w") as output:
            process = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                output.write(line)
                print(line, end="", flush=True)
            status = process.wait()
        logs = out / name
        logs.mkdir(exist_ok=True)
        evidence = []
        event_rows = (
            [json.loads(line) for line in events.read_text().splitlines()]
            if events.exists()
            else []
        )
        for target in targets:
            package, test = target[2:].split(":")
            built = any(
                event.get("id", {}).get("targetCompleted", {}).get("label") == target
                and event.get("completed", {}).get("success") is True
                for event in event_rows
            )
            summaries = [
                event["testSummary"]["overallStatus"]
                for event in event_rows
                if event.get("id", {}).get("testSummary", {}).get("label") == target
            ]
            original = Path("bazel-testlogs") / package / test / "test.log"
            evidence.append(
                {
                    "target": target,
                    "built": built,
                    "statuses": summaries,
                    "diagnostic": original.read_text() if original.exists() else "",
                }
            )
            if original.exists():
                shutil.copyfile(original, logs / (test + ".log"))
        row = {
            "name": name,
            "command": args,
            "exit_status": status,
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "production_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
            "bufferization_sha256": hashlib.sha256(
                bufferization_cpp.read_bytes()
            ).hexdigest(),
        }
        row.update(
            judge(
                status,
                log.read_text(),
                expected,
                reason,
                evidence if targets else None,
                phase,
            )
        )
        row["expected"] = expected
        row["required_reason"] = reason
        row["test_evidence"] = evidence
        report["runs"].append(row)
        save_report()
        if row["verdict"] == "fail":
            raise SystemExit(
                name + ": " + row["classification"] + " violates " + expected
            )
        return status

    def checkout(ref, overlay=False):
        subprocess.run(["git", "checkout", "--force", "--detach", ref], check=True)
        require(
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            == ref,
            "Required preparation check failed: subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == ref",
        )
        require(
            subprocess.check_output(
                ["git", "-C", "third_party/llvm-project", "rev-parse", "HEAD"],
                text=True,
            ).strip()
            == llvm,
            "Required preparation check failed: subprocess.check_output(['git', '-C', 'third_party/llvm-project', 'rev-parse', 'HEAD'], text=True).strip() == llvm",
        )
        if overlay:
            for path, contents in fixture_contents.items():
                Path(path).write_bytes(contents)
        for path in fixtures:
            require(
                hashlib.sha256(Path(path).read_bytes()).hexdigest()
                == report["fixture_hashes"][path],
                "Required preparation check failed: hashlib.sha256(Path(path).read_bytes()).hexdigest() == report['fixture_hashes'][path]",
            )

    def isolated_runtime_probes(name, expected="pass"):
        # --function selects runtime execution; compilation still sees the whole module.
        # Demote other explicit fixture entry points, then let MLIR's symbol-dce own
        # extraction. Do not infer a function boundary from printed braces.
        tools = (
            bazel
            + ["build"]
            + flags
            + [
                "//tools:iree-opt",
                "//tools:iree-run-mlir",
                "@llvm-project//llvm:FileCheck",
            ]
        )
        if command(name + "-probe-tools", tools):
            return
        query = (
            bazel
            + ["cquery"]
            + flags
            + ["@llvm-project//llvm:FileCheck", "--output=files"]
        )
        paths = subprocess.check_output(query, text=True).strip().splitlines()
        require(len(paths) == 1, paths)
        filecheck = paths[0]
        counter = 0
        for path in fixtures:
            text = Path(path).read_text()
            names = re.findall(r"^func\.func @(\w+)\(", text, re.M)
            require(
                len(names) == len(set(names)) and names,
                "Required preparation check failed: len(names) == len(set(names)) and names",
            )
            for line in text.splitlines():
                if not line.startswith("// RUN: "):
                    continue
                counter += 1
                args = shlex.split(line[len("// RUN: ") :])
                pipe = args.index("|")
                run, check = args[:pipe], args[pipe + 1 :]
                selected = [
                    arg.split("=", 1)[1] for arg in run if arg.startswith("--function=")
                ]
                function = selected[0] if selected else names[0]
                require(
                    len(selected) <= 1 and function in names,
                    "Required preparation check failed: len(selected) <= 1 and function in names",
                )
                probe = name + "-" + str(counter) + "-" + function
                modified = re.sub(
                    r"^func\.func @(\w+)(?=\()",
                    lambda m: (
                        m.group(0)
                        if m.group(1) == function
                        else "func.func private @" + m.group(1)
                    ),
                    text,
                    flags=re.M,
                )
                prepared = out / (probe + "-input.mlir")
                capsule = out / (probe + ".mlir")
                prepared.write_text(modified)
                if command(
                    probe + "-isolate",
                    [
                        "bazel-bin/tools/iree-opt",
                        str(prepared),
                        "--symbol-dce",
                        "-o",
                        str(capsule),
                    ],
                ):
                    continue
                run = ["bazel-bin/tools/iree-run-mlir"] + [
                    str(capsule) if arg == "%s" else arg for arg in run[1:]
                ]
                status = command(
                    probe + "-runtime",
                    run,
                    expected=expected,
                    reason="expected no unbounded stack allocations",
                )
                if status == 0:
                    check = [filecheck] + [
                        path if arg == "%s" else arg for arg in check[1:]
                    ]
                    command(
                        probe + "-oracle",
                        check + ["--input-file=" + str(out / (probe + "-runtime.log"))],
                        phase="oracle",
                    )

    fixture_contents = {
        path: subprocess.check_output(["git", "show", source + ":" + path])
        for path in fixtures + registrations
    }
    for path in fixtures:
        report["fixture_hashes"][path] = hashlib.sha256(
            fixture_contents[path]
        ).hexdigest()
    report["candidate_tree"] = subprocess.check_output(
        ["git", "rev-parse", source + "^{tree}"], text=True
    ).strip()
    save_report()

    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("Use an initially clean disposable checkout for comparison.")
    try:
        checkout(source)
        if command(
            "candidate-positive",
            bazel
            + ["test"]
            + flags
            + ["--keep_going", "--test_output=errors"]
            + compiler
            + runtime,
            compiler + runtime,
        ):
            sys.exit("Candidate tests failed; no comparative verdict is inferred.")
        if command(
            "build-ir-tools",
            bazel + ["build"] + flags + ["//tools:iree-compile", "//tools:iree-opt"],
        ):
            sys.exit("IR tools did not build.")

        isolated_runtime_probes("candidate")

        for name, ref in [("base-runtime", base), ("submitted-runtime", submitted)]:
            checkout(ref, overlay=True)
            command(
                name,
                bazel
                + ["test"]
                + flags
                + ["--keep_going", "--test_output=errors"]
                + runtime,
                runtime,
                expected="observe",
            )
            isolated_runtime_probes(name, expected="observe")

        checkout(source)
        cpp.write_bytes(Path(os.environ["CLONE_EXPERIMENT"]).read_bytes())
        command(
            "clone-experiment-runtime",
            bazel
            + ["test"]
            + flags
            + ["--keep_going", "--test_output=errors"]
            + runtime,
            runtime,
            expected="observe",
        )
        isolated_runtime_probes("clone-experiment", expected="observe")
        if (
            command("clone-ir-tools", bazel + ["build"] + flags + ["//tools:iree-opt"])
            == 0
        ):
            command(
                "clone-experiment-flow",
                [
                    str(Path("bazel-bin/tools/iree-opt")),
                    fixtures[2],
                    "--iree-dispatch-creation-pipeline",
                    "--mlir-disable-threading",
                ],
            )
            command(
                "clone-experiment-conversion",
                [
                    "bazel-bin/tools/iree-opt",
                    "compiler/src/iree/compiler/DispatchCreation/test/convert_region_to_workgroups.mlir",
                    "--pass-pipeline=builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups,iree-flow-canonicalize,cse))",
                    "--split-input-file",
                ],
            )

        # A narrower exemption can permit identical init roles of one Sort, but it
        # cannot resolve writes in one Sort against original-value reads in another.
        model = """struct LinalgExtOpInterface
            : bufferization::DstBufferizableOpInterfaceExternalModel<
                  LinalgExtOpInterface<OpTy>, OpTy> {
        """
        bufferization_source = bufferization_cpp.read_text()
        require(
            bufferization_source.count(model) == 1,
            "Required preparation check failed: bufferization_source.count(model) == 1",
        )
        bufferization_cpp.write_text(
            bufferization_source.replace(
                model,
                model
                + """
          bool isNotConflicting(OpOperand *read, OpOperand *write,
                                const AnalysisState &) const {
            return isa<IREE::LinalgExt::SortOp>(read->getOwner()) &&
                   read->getOwner() == write->getOwner() &&
                   read->get() == write->get();
          }
        """,
            )
        )
        if (
            command(
                "clone-same-sort-exemption-tools",
                bazel + ["build"] + flags + ["//tools:iree-opt"],
            )
            == 0
        ):
            command(
                "clone-same-sort-exemption-analysis",
                [
                    "bazel-bin/tools/iree-opt",
                    os.environ["SHARED_DESTINATION_PROBE"],
                    "--pass-pipeline=builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize{test-analysis-only=true print-conflicts=true}))",
                ],
            )
            command(
                "clone-same-sort-exemption-bufferization",
                [
                    "bazel-bin/tools/iree-opt",
                    os.environ["SHARED_DESTINATION_PROBE"],
                    "--pass-pipeline=builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize,canonicalize,cse,canonicalize))",
                ],
            )

        checkout(source)
        original = cpp.read_text()
        # Controls test the allocation and initialization contracts separately.
        before = "        convertDispatchRegionWithSortWorkspaces(regionOp, rewriter);"
        require(
            original.count(before) == 1,
            "Required preparation check failed: original.count(before) == 1",
        )
        cpp.write_text(
            original.replace(
                before,
                "        rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);",
            )
        )
        command(
            "no-workspace-control",
            bazel + ["test"] + flags + ["--test_output=errors", runtime[0]],
            [runtime[0]],
            expected="semantic-failure",
            reason="expected no unbounded stack allocations",
        )

        checkout(source)
        before = "  eliminateTriviallyDeadOps(rewriter, regionOp.getBody());\n"
        require(
            original.count(before) == 1,
            "Required preparation check failed: original.count(before) == 1",
        )
        cpp.write_text(original.replace(before, ""))
        command(
            "dead-use-cleanup-control",
            bazel
            + ["test"]
            + flags
            + [
                "--test_output=errors",
                "//compiler/src/iree/compiler/DispatchCreation/test:convert_region_to_workgroups.mlir.test",
            ],
            [
                "//compiler/src/iree/compiler/DispatchCreation/test:convert_region_to_workgroups.mlir.test"
            ],
            expected="semantic-failure",
            reason="error: .*expected string not found in input",
        )

        checkout(source)
        before = """linalg::CopyOp::create(rewriter, sortOp.getLoc(), ValueRange{source},
                                 ValueRange{allocation.getResult()});"""
        require(
            original.count(before) == 1,
            "Required preparation check failed: original.count(before) == 1",
        )
        modified = original.replace(
            '#include "mlir/Dialect/Affine/IR/AffineOps.h"',
            '#include "mlir/Dialect/Affine/IR/AffineOps.h"\n#include "mlir/Dialect/Arith/IR/Arith.h"',
        )
        modified = modified.replace(
            before,
            "linalg::FillOp::create(rewriter, sortOp.getLoc(), "
            "ValueRange{arith::ConstantOp::create(rewriter, sortOp.getLoc(), "
            "rewriter.getZeroAttr(type.getElementType())).getResult()}, ValueRange{allocation.getResult()});",
        )
        cpp.write_text(modified)
        command(
            "zero-key-initialization-control",
            bazel
            + ["test"]
            + flags
            + [
                "--test_output=errors",
                runtime[
                    fixtures.index(
                        "tests/e2e/regression/argsort_preserve_input_dynamic.mlir"
                    )
                ],
            ],
            [
                runtime[
                    fixtures.index(
                        "tests/e2e/regression/argsort_preserve_input_dynamic.mlir"
                    )
                ]
            ],
            expected="semantic-failure",
            reason="error: .*expected string not found in input",
        )

        checkout(source)
        # Rebuild the candidate after the experimental source changes.
        if command(
            "candidate-restored",
            bazel
            + ["test"]
            + flags
            + ["--keep_going", "--test_output=errors"]
            + compiler
            + runtime,
            compiler + runtime,
        ):
            sys.exit("Restored candidate did not pass.")
        if command(
            "restored-ir-tools",
            bazel + ["build"] + flags + ["//tools:iree-compile", "//tools:iree-opt"],
        ):
            sys.exit("Restored IR tools did not build.")
        for path in fixtures:
            name = Path(path).stem
            command(
                name + "-flow",
                [
                    str(Path("bazel-bin/tools/iree-opt")),
                    path,
                    "--iree-dispatch-creation-pipeline",
                    "--mlir-disable-threading",
                ],
            )
            command(
                name + "-lowering",
                [
                    str(Path("bazel-bin/tools/iree-compile")),
                    path,
                    "--iree-hal-target-device=local",
                    "--iree-hal-local-target-device-backends=llvm-cpu",
                    "--iree-llvmcpu-target-cpu=generic",
                    "--mlir-disable-threading",
                    "--mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize,iree-codegen-tile-and-distribute-to-workgroups-using-forall-op,iree-llvmcpu-check-ir-before-llvm-conversion",
                    "-o",
                    str(out / (name + ".vmfb")),
                ],
            )
            lowering = (out / (name + "-lowering.log")).read_text()
            require(
                "IR Dump After IREEComprehensiveBufferizePass" in lowering,
                name + ": missing intermediate allocation evidence",
            )
            require(
                "memref.alloca" not in lowering and "memref.alloc(" not in lowering,
                name + ": private kernel allocation",
            )
            if name == "argsort_views_dynamic":
                command(
                    name + "-distributed-lowering",
                    [
                        str(Path("bazel-bin/tools/iree-compile")),
                        path,
                        "--iree-hal-target-device=local",
                        "--iree-hal-local-target-device-backends=llvm-cpu",
                        "--iree-llvmcpu-target-cpu=generic",
                        "--iree-llvmcpu-distribution-size=1",
                        "--mlir-disable-threading",
                        "--mlir-print-ir-after=iree-codegen-iree-comprehensive-bufferize,iree-codegen-tile-and-distribute-to-workgroups-using-forall-op,iree-llvmcpu-check-ir-before-llvm-conversion",
                        "-o",
                        str(out / (name + "-distributed.vmfb")),
                    ],
                )
        command(
            "shared-destination-analysis",
            [
                "bazel-bin/tools/iree-opt",
                os.environ["SHARED_DESTINATION_PROBE"],
                "--pass-pipeline=builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize{test-analysis-only=true print-conflicts=true}))",
            ],
        )
        command(
            "shared-destination-bufferization",
            [
                "bazel-bin/tools/iree-opt",
                os.environ["SHARED_DESTINATION_PROBE"],
                "--pass-pipeline=builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize,canonicalize,cse,canonicalize))",
            ],
        )
        # Unspecified experiment outcomes remain inconclusive; they cannot turn the run green.
        subprocess.run(["git", "diff", "--exit-code"], check=True)
        save_report()

        report["complete"] = True
        report["verdict"] = (
            "inconclusive"
            if any(r["verdict"] == "inconclusive" for r in report["runs"])
            else "pass"
        )
        save_report()
        if report["verdict"] != "pass":
            raise SystemExit(2)
    finally:
        checkout(source)
        subprocess.run(["git", "diff", "--exit-code"], check=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        # An error during preparation cannot leave a previous green report.
        report_path = None
        if os.environ.get("RUNNER_TEMP"):
            report_path = (
                Path(os.environ["RUNNER_TEMP"]) / "sort-workspace-evidence/results.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    {
                        "verdict": "fail",
                        "classification": "preparation-failure",
                        "failure": "required phases have not completed",
                    }
                )
                + "\n"
            )
        try:
            main()
        except BaseException:
            if report_path:
                current = json.loads(report_path.read_text())
                if current.get("verdict") not in ("fail", "inconclusive"):
                    current["verdict"] = "fail"
                    current["failure"] = (
                        "driver terminated before required phases completed"
                    )
                    report_path.write_text(json.dumps(current, indent=2) + "\n")
            raise
