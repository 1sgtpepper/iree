#!/usr/bin/env python3
"""Compare capture bookkeeping in fork CI without changing production IR."""

import json
import os
from pathlib import Path
import subprocess
import sys


published_revision = "c059af93cf68af6df458df22a11f0fdc2576f9f3"
candidate_revision = os.environ["SOURCE_SHA"]
output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=True)
converter = Path(
    "compiler/src/iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.cpp"
)
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
pipeline = (
    "builtin.module(util.func("
    "iree-dispatch-creation-convert-dispatch-regions-to-workgroups))"
)


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def fixture(width, mode):
    types = []
    for index in range(width):
        shape = "4" if mode == "static" else "?"
        if mode == "mixed":
            shape = ("4", "?", "?x?")[index % 3]
        types.append(f"tensor<{shape}xi64>")
    arguments = [
        f"%keys_{i}: {ty}, %indices_{i}: {ty}"
        for i, ty in enumerate(types)
    ]
    arguments.append("%reverse: i1")
    result_types = ", ".join(types)
    lines = [
        f"util.func public @wide_sort({', '.join(arguments)})",
        f"    -> ({result_types}) {{",
    ]
    for dimension in range(max(ty.count("?") for ty in types)):
        lines.append(f"  %c{dimension} = arith.constant {dimension} : index")
    region_types = []
    for index, ty in enumerate(types):
        dims = []
        for dimension in range(ty.count("?")):
            dim = f"%dim_{index}_{dimension}"
            lines.append(
                f"  {dim} = tensor.dim %indices_{index}, %c{dimension} : {ty}"
            )
            dims.append(dim)
        region_types.append(ty + ("{" + ", ".join(dims) + "}" if dims else ""))
    lines.append(
        f"  %results:{width} = flow.dispatch.region -> ({', '.join(region_types)}) {{"
    )
    for index, ty in enumerate(types):
        lines.extend([
            f"    %sorted_{index}:2 = iree_linalg_ext.sort dimension(0)",
            f"        outs(%keys_{index}, %indices_{index} : {ty}, {ty}) {{",
            "    ^bb0(%lhs: i64, %rhs: i64, %lhs_index: i64, %rhs_index: i64):",
            "      %ascending = arith.cmpi sle, %lhs, %rhs : i64",
            "      %descending = arith.cmpi sge, %lhs, %rhs : i64",
            "      %take_lhs = arith.select %reverse, %descending, %ascending : i1",
            "      iree_linalg_ext.yield %take_lhs : i1",
            f"    }} -> {ty}, {ty}",
        ])
    values = ", ".join(f"%sorted_{i}#1" for i in range(width))
    results = ", ".join(f"%results#{i}" for i in range(width))
    lines.extend([
        f"    flow.return {values} : {result_types}",
        "  }",
        f"  util.return {results} : {result_types}",
        "}",
    ])
    return "\n".join(lines) + "\n"


def instrument(label):
    source = converter.read_text()
    source = '#include "llvm/Support/raw_ostream.h"\n' + source
    if label == "published":
        for variable in ("tiedArgument", "tiedBase"):
            source = replace_once(
                source,
                f"llvm::find(argumentsSet, {variable})",
                "llvm::find_if(argumentsSet, [&](Value candidate) { "
                'llvm::errs() << "capture-index-visit\\n"; '
                f"return candidate == {variable}; }})",
            )
        assert source.count("IREE::Util::findDynamicDimsInList(") == 3
        source = source.replace(
            "IREE::Util::findDynamicDimsInList(", "getTrackedDims("
        )
        source = replace_once(source, "  // Find tied results.\n", '''
  auto getTrackedDims = [](unsigned index, ValueRange values,
                           ValueRange dims) -> ValueRange {
    llvm::errs() << "dimension-scan-begin\\n";
    auto result = IREE::Util::findDynamicDimsInList(index, values, dims);
    llvm::errs() << "dimension-scan-end\\n";
    return result;
  };
  // Find tied results.
''')
    else:
        source = replace_once(
            source,
            "    argumentIndices[tensor] = index;",
            '    llvm::errs() << "capture-index-record\\n";\n'
            "    argumentIndices[tensor] = index;",
        )
        source = replace_once(
            source,
            "  auto getArgumentDims = [&](unsigned index) -> ValueRange {",
            "  auto getArgumentDims = [&](unsigned index) -> ValueRange {\n"
            '    llvm::errs() << "dimension-range-lookup\\n";',
        )
        assert source.count("argumentIndices.lookup(") == 2
        source = source.replace("argumentIndices.lookup(", "getTrackedIndex(")
        source = replace_once(source, "  // Find tied results.\n", '''
  auto getTrackedIndex = [&](Value value) {
    llvm::errs() << "capture-index-lookup\\n";
    return argumentIndices.lookup(value);
  };
  // Find tied results.
''')
    converter.write_text(source)
    source = util_source.read_text()
    start = source.index("ValueRange findDynamicDimsInList(")
    end = source.index("\nValue findValueSizeInList(", start)
    function = replace_once(
        source[start:end],
        "  for (unsigned i = 0; i < idx; ++i) {",
        "  for (unsigned i = 0; i < idx; ++i) {\n"
        '    llvm::errs() << "dimension-prefix-visit\\n";',
    )
    source = source[:start] + function + source[end:]
    util_source.write_text('#include "llvm/Support/raw_ostream.h"\n' + source)


def measurements(stderr):
    active = False
    prefixes = 0
    for line in stderr.splitlines():
        if line == "dimension-scan-begin":
            assert not active
            active = True
        elif line == "dimension-scan-end":
            assert active
            active = False
        elif line == "dimension-prefix-visit" and active:
            prefixes += 1
    assert not active
    result = {"converter_dimension_prefix_visits": prefixes}
    for marker in (
        "capture-index-visit", "capture-index-record", "capture-index-lookup",
        "dimension-range-lookup",
    ):
        result[marker.replace("-", "_")] = stderr.splitlines().count(marker)
    return result


cases = {}
for mode in ("static", "dynamic", "mixed"):
    for width in (16, 32, 64):
        name = f"{mode}-{width}"
        path = output / f"{name}.mlir"
        path.write_text(fixture(width, mode))
        cases[name] = (path, width, mode)

report = {}
subprocess.run(["git", "diff", "--exit-code"], check=True)
try:
    for label, revision in (
        ("published", published_revision), ("candidate", candidate_revision)
    ):
        subprocess.run(["git", "checkout", "--detach", revision], check=True)
        originals = {path: path.read_text() for path in (converter, util_source)}
        try:
            instrument(label)
            subprocess.run(build, check=True)
            tool = str(Path("bazel-bin/tools/iree-opt").resolve())
            report[label] = {"revision": revision, "cases": {}}
            for name, (path, width, mode) in cases.items():
                result = subprocess.run(
                    [tool, str(path), "--mlir-disable-threading",
                     "--pass-pipeline=" + pipeline],
                    capture_output=True, text=True, check=True,
                )
                assert result.stdout.count("iree_linalg_ext.sort") == width
                (output / f"{label}-{name}.mlir").write_text(result.stdout)
                (output / f"{label}-{name}.stderr").write_text(result.stderr)
                report[label]["cases"][name] = measurements(result.stderr)
            (output / "measurements.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({label: report[label]}), flush=True)
        finally:
            for path, original in originals.items():
                path.write_text(original)

    for name, (_, width, mode) in cases.items():
        old = report["published"]["cases"][name]
        new = report["candidate"]["cases"][name]
        assert old["capture_index_visit"] >= 2 * width * width, (name, old)
        assert new["capture_index_visit"] == 0, (name, new)
        assert new["capture_index_record"] == 2 * width + 1, (name, new)
        assert new["capture_index_lookup"] == 2 * width, (name, new)
        assert new["dimension_range_lookup"] == 5 * width, (name, new)
        assert new["converter_dimension_prefix_visits"] == 0, (name, new)
        if mode == "dynamic":
            assert old["converter_dimension_prefix_visits"] >= width * width
        assert (output / f"published-{name}.mlir").read_bytes() == (
            output / f"candidate-{name}.mlir"
        ).read_bytes(), f"Changed generated IR for {name}"
    print("All nine width cases preserve IR and remove repeated metadata scans.", flush=True)
finally:
    subprocess.run(["git", "checkout", "--detach", candidate_revision], check=True)
    subprocess.run(["git", "diff", "--exit-code"], check=True)
