# Sort workspace CPU validation

This fork-only suite builds and tests source revision
`59201b95789d75836fbabc36eccc6b8b5abdddfa`, with its pinned LLVM submodule.
The workflow checks out that source separately from the validation branch so
workflow changes cannot silently change the compiler revision under test.

The required checks include independent ordering and input-preservation oracles,
allocation checks, compiler fixtures, literal runtime RUN lines, and four rebuilt
source controls. Build failures, missing checks, unrelated diagnostics, stale
reports, and escaped controls fail the run. The final source is restored and the
positive suites run again after the controls.

The workflow runs on push to `validation/sort-workspace-cpu` in this fork. Its
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
