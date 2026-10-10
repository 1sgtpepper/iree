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
import shutil
import subprocess
import sys

source = os.environ['SOURCE_SHA']
base = os.environ['BASE_SHA']
submitted = os.environ['SUBMITTED_SHA']
llvm = os.environ['LLVM_SHA']
out = Path(os.environ['RUNNER_TEMP']) / 'sort-workspace-evidence'
out.mkdir(exist_ok=True)
report = {'candidate': source, 'base': base, 'submitted': submitted, 'llvm': llvm,
          'runs': [], 'fixture_hashes': {}}
cpp = Path('compiler/src/iree/compiler/DispatchCreation/ConvertDispatchRegionsToWorkgroups.cpp')
fixtures = ['tests/e2e/regression/' + name + '.mlir' for name in (
    'argsort_dynamic', 'argsort_preserve_input_dynamic',
    'argsort_comparator_lanes_dynamic', 'argsort_views_dynamic')]
registrations = ['tests/e2e/regression/BUILD.bazel',
                 'tests/e2e/regression/CMakeLists.txt']
# Bazel target separators differ from repository file paths.
runtime = ['//tests/e2e/regression:' + Path(path).name + '.test' for path in fixtures]
compiler = [
    '//compiler/src/iree/compiler/Dialect/LinalgExt/IR/test:canonicalize.mlir.test',
    '//compiler/src/iree/compiler/Dialect/Flow/IR/test:dispatch_folding.mlir.test',
    '//compiler/src/iree/compiler/DispatchCreation/test:convert_region_to_workgroups.mlir.test',
    '//compiler/src/iree/compiler/DispatchCreation/test:dispatch_linalg_on_tensors.mlir.test',
    '//compiler/src/iree/compiler/DispatchCreation/test:form_scalar_dispatches.mlir.test',
    '//compiler/src/iree/compiler/Codegen/Common/test:iree_comprehensive_bufferize.mlir.test',
]
bazel = ['bazel', '--noworkspace_rc', '--bazelrc=build_tools/bazel/iree.bazelrc']
flags = ['--sandbox_base=' + os.environ['SANDBOX_BASE'],
         '--disk_cache=' + os.environ['BAZEL_DISK_CACHE'],
         '--config=generic_clang_ci', '--noremote_upload_local_results',
         '--iree_compiler_plugins=hal_target_llvm_cpu,hal_target_local',
         '--iree_drivers=local-task', '--jobs=4', '--local_cpu_resources=4']


def save_report():
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


def command(name, args, targets=()):
    print('Running ' + name + ': ' + ' '.join(args), flush=True)
    log = out / (name + '.log')
    with log.open('w') as output:
        process = subprocess.Popen(args, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output.write(line)
            print(line, end='', flush=True)
        status = process.wait()
    logs = out / name
    logs.mkdir(exist_ok=True)
    for target in targets:
        package, test = target[2:].split(':')
        original = Path('bazel-testlogs') / package / test / 'test.log'
        if original.exists():
            shutil.copyfile(original, logs / (test + '.log'))
    row = {'name': name, 'command': args, 'exit_status': status,
           'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'production_sha256': hashlib.sha256(cpp.read_bytes()).hexdigest()}
    report['runs'].append(row)
    save_report()
    return status


def checkout(ref, overlay=False):
    subprocess.run(['git', 'checkout', '--force', '--detach', ref], check=True)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == ref
    assert subprocess.check_output(['git', '-C', 'third_party/llvm-project', 'rev-parse', 'HEAD'], text=True).strip() == llvm
    if overlay:
        for path, contents in fixture_contents.items():
            Path(path).write_bytes(contents)
    for path in fixtures:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == report['fixture_hashes'][path]


fixture_contents = {path: subprocess.check_output(['git', 'show', source + ':' + path])
                    for path in fixtures + registrations}
for path in fixtures:
    report['fixture_hashes'][path] = hashlib.sha256(fixture_contents[path]).hexdigest()
report['candidate_tree'] = subprocess.check_output(['git', 'rev-parse', source + '^{tree}'], text=True).strip()
save_report()

checkout(source)
if command('candidate-positive', bazel + ['test'] + flags +
           ['--keep_going', '--test_output=errors'] + compiler + runtime,
           compiler + runtime):
    sys.exit('Candidate tests failed; no comparative verdict is inferred.')
if command('build-ir-tools', bazel + ['build'] + flags + ['//tools:iree-compile', '//tools:iree-opt']):
    sys.exit('IR tools did not build.')

for name, ref in [('base-runtime', base), ('submitted-runtime', submitted)]:
    checkout(ref, overlay=True)
    command(name, bazel + ['test'] + flags + ['--keep_going', '--test_output=errors'] + runtime, runtime)

checkout(source)
cpp.write_bytes(Path(os.environ['CLONE_EXPERIMENT']).read_bytes())
command('clone-experiment-runtime', bazel + ['test'] + flags +
        ['--keep_going', '--test_output=errors'] + runtime, runtime)
command('clone-ir-tools', bazel + ['build'] + flags + ['//tools:iree-opt'])
command('clone-experiment-flow', [str(Path('bazel-bin/tools/iree-opt')),
        fixtures[2], '--iree-dispatch-creation-pipeline', '--mlir-disable-threading'])

checkout(source)
original = cpp.read_text()
# Controls test the allocation and initialization contracts separately.
before = '        convertDispatchRegionWithSortWorkspaces(regionOp, rewriter);'
assert original.count(before) == 1
cpp.write_text(original.replace(before,
    '        rewriteFlowDispatchRegionToFlowDispatchWorkgroups(regionOp, rewriter);'))
command('no-workspace-control', bazel + ['test'] + flags +
        ['--test_output=errors', runtime[0]], [runtime[0]])

checkout(source)
# Make Copy read its own uninitialized destination, retaining all destination plumbing.
before = 'linalg::CopyOp::create(rewriter, sortOp.getLoc(), source, workspace);'
assert original.count(before) == 1
cpp.write_text(original.replace(before,
    'linalg::CopyOp::create(rewriter, sortOp.getLoc(), workspace, workspace);'))
command('uninitialized-workspace-control', bazel + ['test'] + flags +
        ['--test_output=errors', runtime[1]], [runtime[1]])

checkout(source)
# Rebuild the candidate after the experimental source changes.
if command('candidate-restored', bazel + ['test'] + flags +
           ['--keep_going', '--test_output=errors'] + compiler + runtime,
           compiler + runtime):
    sys.exit('Restored candidate did not pass.')
if command('restored-ir-tools', bazel + ['build'] + flags + ['//tools:iree-compile', '//tools:iree-opt']):
    sys.exit('Restored IR tools did not build.')
for path in fixtures:
    name = Path(path).stem
    command(name + '-flow', [str(Path('bazel-bin/tools/iree-opt')), path,
                            '--iree-dispatch-creation-pipeline', '--mlir-disable-threading'])
    command(name + '-lowering', [str(Path('bazel-bin/tools/iree-compile')), path,
            '--iree-hal-target-device=local', '--iree-hal-local-target-device-backends=llvm-cpu',
            '--iree-llvmcpu-target-cpu=generic', '--mlir-disable-threading',
            '--mlir-print-ir-after=iree-comprehensive-bufferize,iree-tile-and-distribute-to-workgroups-using-forall-op,iree-llvmcpu-check-ir-before-llvm-conversion',
            '-o', str(out / (name + '.vmfb'))])
# Experimental results are evidence, not assumed expected failures.
subprocess.run(['git', 'diff', '--exit-code'], check=True)
save_report()
