# Sort workspace CPU validation

This fork-only suite builds and tests source revision
`62cc96b231b187c1aa8eaacf235234abb7dd434b`, with its pinned LLVM submodule.
The workflow checks out that source separately from the validation branch so
workflow changes cannot silently change the compiler revision under test.

The required checks include independent ordering and input-preservation oracles,
allocation checks, compiler fixtures, literal runtime RUN lines, and eight source
controls with successful compiler builds. Build failures, missing checks, unrelated diagnostics, stale
reports, and escaped controls fail the run. The workspace suite requires 40
independent runtime cases across local-sync and local-task, strict dynamic
allocation checks, a bounded preservation-copy
check and the intended buffer-tiling rejection. The final source is restored and
all positive suites run again after the controls.
Aggregate admission also requires complete primary phases and every successful
control build; executable report mutations verify that incomplete or failed
reports cannot become a full validation success.

Run the registered manual workflow against this validation branch:

```bash
gh workflow run ci_linux_x64_clang_byollvm.yml --repo 1sgtpepper/iree --ref validation/sort-workspace-cpu
```

Only this branch overrides the existing workflow entry with the focused suite.
The fork default branch is unchanged. Its
`sort-workspace-cpu-evidence` artifact includes the final verdict, source and
compiler identities, command logs, runtime outputs, and intermediate IR.

To reproduce the same directory layout, place the exact source checkout in
`source/`, copy this directory to the sibling `suite/`, initialize the three
submodules listed in `validation/source-identity.json`, and create `suite/.venv`
with these requirements. With Clang, CMake, Ninja and ccache installed, run:

```bash
mkdir -p suite/evidence
bash suite/validation/validate.sh
```

The original failing local-update case remains a required positive. A seventh
rebuilt control restores the first repair and must reproduce its specific
local-initializer stack-limit failure. The final source initializes local empty
producer destinations in managed storage within the original dispatch.

The existing manual Clang Tidy workflow remains pinned to the earlier LLVM53
analysis tool from upstream run32944069152. It is historical validation only;
its identity gate rejects the newer LLVM32 source. A matching tool is required
before this workflow can validate the integrated revision. Tool revision/hash mismatches and compiler
diagnostics fail the job. Enabled tidy warnings on changed lines fail under
the repository's existing diff policy; deliberate lint and compiler violations
exercise that gate. Run it with:

```bash
gh workflow run clang_tidy.yml --repo 1sgtpepper/iree --ref validation/sort-workspace-cpu
```

Native compiler, sanitizer, Bazel and package workflows use hosted runners on
the fork, bounded build parallelism and local build caches. Their original
build/test scripts remain required. The debug job builds and installs the full
project without the optional test dependencies, matching its documented
build-only contract. The fork source branch and upstream PR are separate from
these validation workflow changes.

The original ARM64 io_uring failure can be checked explicitly with:

```bash
gh workflow run ci_linux_arm64_clang.yml --repo 1sgtpepper/iree --ref validation/sort-workspace-cpu
```

On the fork this runs the existing runtime test on a hosted ARM64 CPU and
requires its explicit success output. A missing or skipped GTest case fails.
The original ARM64 compiler job retains its owner restriction and is reported
as skipped on the fork; this probe does not claim an ARM64 compiler build.

The full-axis regression cases require explicit extent-four tensor and buffer
tiles to succeed. An eighth source control rebuilds the prior constant-only
offset guard and requires both legal cases to fail for their intended tiling
diagnostics. Missing this control or failing its compiler build rejects aggregate
admission. Partial-axis and orthogonal-axis cases remain in the native fixture.
