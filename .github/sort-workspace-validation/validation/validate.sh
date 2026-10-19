#!/usr/bin/env bash
set -euo pipefail
validation_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_root="${validation_root}/../source"
cd "${validation_root}"
validation_id=$(python3 -c 'import uuid; print(uuid.uuid4())')
export SORT_VALIDATION_RUN_ID="${validation_id}"
python3 validation/record_run_status.py running "${validation_id}"
record_failure() {
  local validation_status=$?
  if [[ "${validation_status}" -ne 0 ]]; then
    python3 validation/record_run_status.py fail "${validation_id}" "${validation_status}"
  fi
}
trap record_failure EXIT
test "$#" -eq 0 || { echo "This command runs every required check; arguments are unsupported." >&2; exit 2; }
source_revision=$(git -C "${source_root}" rev-parse HEAD)
.venv/bin/python validation/41-fork-comparison-driver.py --self-test > evidence/harness-self-test.json
.venv/bin/python -O validation/41-fork-comparison-driver.py --self-test > evidence/harness-optimized-self-test.json
.venv/bin/python validation/test_comparison_driver.py > evidence/harness-command-test.log
.venv/bin/python -O validation/test_comparison_driver.py > evidence/harness-optimized-command-test.log
.venv/bin/python validation/test_fail_open_control.py > evidence/harness-fail-open-check.log
cmake -S "${source_root}" -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  '-DCMAKE_C_FLAGS_RELEASE=-O1 -DNDEBUG' '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG' \
  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-keep-memory -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--no-keep-memory \
  -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_BENCHMARKS=OFF -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_EMBEDDED_RELEASE_INFO=ON "-DIREE_RELEASE_REVISION:STRING=${source_revision}" -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_INPUT_TORCH=OFF -DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TOSA=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_TARGET_BACKEND_LLVM_CPU_WASM=OFF -DIREE_TARGET_BACKEND_VMVX=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
  -DIREE_ENABLE_CPUINFO=OFF -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
  -DIREE_DEFAULT_CPU_LLVM_TARGETS=X86 -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=1 > evidence/reproducible-configure.log 2>&1
ninja -C build -j2 iree-compile iree-opt iree-run-module iree-run-mlir FileCheck > evidence/reproducible-build.log 2>&1
.venv/bin/python validation/validate_exact_source.py
build/tools/iree-opt --split-input-file \
  --pass-pipeline='builtin.module(util.func(iree-stream-materialize-copy-on-write))' \
  "${source_root}/compiler/src/iree/compiler/Dialect/Stream/Transforms/test/materialize_copy_on_write.mlir" \
  > evidence/materialize-copy-on-write.mlir 2> evidence/materialize-copy-on-write.log
build/llvm-project/bin/FileCheck \
  "${source_root}/compiler/src/iree/compiler/Dialect/Stream/Transforms/test/materialize_copy_on_write.mlir" \
  --input-file=evidence/materialize-copy-on-write.mlir \
  > evidence/materialize-copy-on-write-filecheck.log 2>&1
build/tools/iree-opt --split-input-file --iree-stream-elide-async-copies \
  "${source_root}/compiler/src/iree/compiler/Dialect/Stream/Transforms/test/elide_async_copies_tensor_import_consume.mlir" \
  > evidence/elide-borrowed-copies.mlir 2> evidence/elide-borrowed-copies.log
build/llvm-project/bin/FileCheck \
  "${source_root}/compiler/src/iree/compiler/Dialect/Stream/Transforms/test/elide_async_copies_tensor_import_consume.mlir" \
  --input-file=evidence/elide-borrowed-copies.mlir \
  > evidence/elide-borrowed-copies-filecheck.log 2>&1
.venv/bin/python validation/check_additional_sort_cases.py
.venv/bin/python validation/validate_sort_workspace_paths.py --source "${source_root}"
.venv/bin/python validation/validate_source_controls.py
.venv/bin/python validation/validate_exact_source.py --out evidence/restored-exact-source
.venv/bin/python validation/check_additional_sort_cases.py
.venv/bin/python validation/validate_sort_workspace_paths.py --source "${source_root}" \
  --out evidence/restored-sort-workspace-paths \
  --positive-report evidence/restored-exact-source/results.json
.venv/bin/python validation/summarize_results.py
.venv/bin/python validation/test_aggregate_controls.py > evidence/aggregate-command-test.log
