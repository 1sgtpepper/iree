#!/usr/bin/env python3
"""Fork-CI-only source comparison; not part of the upstream patch."""

import json
import os
from pathlib import Path
import subprocess
import sys


source_revision = os.environ["SOURCE_SHA"]
published_revision = "220f8242250780e5f63e7dfb99737e560117eadd"
base_revision = "6ec253ad4e259ad256c8a226f7a7a330debf937c"
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
liveness_fixture = Path(sys.argv[2]).resolve()
util_source = Path("compiler/src/iree/compiler/Dialect/Util/IR/UtilTypes.cpp")
build = [
    "bazel", "--noworkspace_rc", "--bazelrc=build_tools/bazel/iree.bazelrc",
    "build", "--sandbox_base=" + os.environ["SANDBOX_BASE"],
    "--disk_cache=" + os.environ["BAZEL_DISK_CACHE"],
    "--config=generic_clang_ci", "--noremote_upload_local_results",
    "--iree_compiler_plugins=hal_target_llvm_cpu,hal_target_local",
    "--iree_drivers=local-task", "--jobs=4", "--local_cpu_resources=4",
    "//tools:iree-opt",
]
pipeline = "builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups))"
cleanup_pipeline = (
    "builtin.module(util.func("
    "iree-dispatch-creation-convert-dispatch-regions-to-workgroups,"
    "iree-flow-canonicalize,cse))"
)
marker = "required-tie-predecessor-visit\n"
chain_length = 64
chain = [
    "util.func public @sort_chain(%keys: tensor<4xi64>,",
    "    %scratch: tensor<4xi64>, %indices: tensor<4xi64>) -> tensor<4xi64> {",
    "  %result = flow.dispatch.region -> (tensor<4xi64>) {",
]
for index in range(chain_length):
    keys = "%keys" if index == 0 else f"%sorted_{index - 1}#0"
    values = "%indices" if index == chain_length - 1 else "%scratch"
    chain.extend([
        f"    %sorted_{index}:2 = iree_linalg_ext.sort dimension(0)",
        f"        outs({keys}, {values} : tensor<4xi64>, tensor<4xi64>) {{",
        "    ^bb0(%lhs: i64, %rhs: i64, %unused_lhs: i64, %unused_rhs: i64):",
        "      %take_lhs = arith.cmpi sle, %lhs, %rhs : i64",
        "      iree_linalg_ext.yield %take_lhs : i1",
        "    } -> tensor<4xi64>, tensor<4xi64>",
    ])
chain.extend([
    f"    flow.return %sorted_{chain_length - 1}#1 : tensor<4xi64>",
    "  }",
    "  util.return %result : tensor<4xi64>",
    "}",
])
chain_fixture = output / "sort-chain.mlir"
chain_fixture.write_text("\n".join(chain) + "\n")
measurements = {}

subprocess.run(["git", "diff", "--exit-code"], check=True)
try:
    for label, revision in [
        ("published", published_revision),
        ("corrected", source_revision),
        ("base", base_revision),
    ]:
        subprocess.run(["git", "checkout", "--detach", revision], check=True)
        subprocess.run(["git", "submodule", "update", "--init", "--depth=1",
                        "third_party/llvm-project"], check=True)
        original = util_source.read_text()
        try:
            if label != "base":
                start = original.index("Value TiedOpInterface::findTiedBaseValue(")
                end = original.index("\n// static", start)
                function = original[start:end]
                advance = "    baseValue = tiedValue;"
                assert function.count(advance) == 1
                function = function.replace(
                    advance, '    llvm::errs() << "required-tie-predecessor-visit\\n";\n'
                    + advance,
                )
                instrumented = original[:start] + function + original[end:]
                include = '#include "llvm/ADT/BitVector.h"\n'
                assert instrumented.count(include) == 1
                util_source.write_text(instrumented.replace(
                    include, include + '#include "llvm/Support/raw_ostream.h"\n'
                ))
            subprocess.run(build, check=True)
            tool = str(Path("bazel-bin/tools/iree-opt").resolve())
            measurement = {"revision": revision}
            if label != "base":
                result = subprocess.run(
                    [tool, str(chain_fixture), "--mlir-disable-threading",
                     "--pass-pipeline=" + pipeline],
                    check=True, capture_output=True, text=True,
                )
                (output / f"{label}-chain.mlir").write_text(result.stdout)
                (output / f"{label}-chain.stderr").write_text(result.stderr)
                measurement["predecessor_advances"] = result.stderr.count(marker)
            result = subprocess.run(
                [tool, str(liveness_fixture), "--split-input-file",
                 "--mlir-disable-threading", "--pass-pipeline=" + cleanup_pipeline],
                check=True, capture_output=True, text=True,
            )
            (output / f"{label}-liveness.mlir").write_text(result.stdout)
            (output / f"{label}-liveness.stderr").write_text(result.stderr)
            measurement["liveness_fixture_sorts_remaining"] = result.stdout.count(
                "iree_linalg_ext.sort"
            )
            measurements[label] = measurement
            (output / "measurements.json").write_text(json.dumps(measurements, indent=2))
            print(json.dumps({label: measurement}), flush=True)
        finally:
            util_source.write_text(original)

    assert measurements["published"]["predecessor_advances"] >= (
        chain_length * (chain_length - 1) // 2
    ), measurements
    assert measurements["corrected"]["predecessor_advances"] == 0, measurements
    assert (output / "published-chain.mlir").read_bytes() == (
        output / "corrected-chain.mlir"
    ).read_bytes(), "The efficiency correction changed generated dispatch IR"
    assert (output / "published-liveness.mlir").read_bytes() == (
        output / "corrected-liveness.mlir"
    ).read_bytes(), "The efficiency correction changed liveness behavior"
finally:
    subprocess.run(["git", "checkout", "--detach", source_revision], check=True)
    subprocess.run(["git", "submodule", "update", "--init", "--depth=1",
                    "third_party/llvm-project"], check=True)
    subprocess.run(["git", "diff", "--exit-code"], check=True)
