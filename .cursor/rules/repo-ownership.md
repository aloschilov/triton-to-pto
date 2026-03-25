# Repository Ownership Rules

## Workspace Layout

This workspace contains three related repositories under `/Users/aloschilov/articles-workspace/`:

| Directory | Remote | Role |
|-----------|--------|------|
| `triton-to-pto/` | `origin` (your repo) | Main project. Commit changes here. |
| `PTOAS/` | `zhangstevenunity/PTOAS` | **READ-ONLY** upstream clone. NEVER modify or commit. |
| `PTOAS-aloschilov/` | `aloschilov/PTOAS` | Your fork of PTOAS. ALL PTOAS changes go here. |

## Rules

1. **NEVER modify files under `../PTOAS/`** (the upstream clone). This includes:
   - Editing source files
   - Running `git add`, `git commit`, or `git stash`
   - Writing new files (except ephemeral Docker-mounted output that gets cleaned up)

2. **ALL changes to PTOAS code** (generate_testcase.py, run_sh_template.sh, Dockerfile, EmitC passes, etc.) **MUST be made in `../PTOAS-aloschilov/`** (the fork).

3. **When `PTOAS_ROOT` or `e2e_all.sh` references the PTOAS repo**, it mounts `../PTOAS/` read-only for Docker execution. If a file under that path needs modification, edit the corresponding file in `../PTOAS-aloschilov/` instead.

4. **Changes to this repo** (`triton-to-pto/`) such as `cpu_sim_run.py`, `golden_check.py`, `e2e_all.sh`, and MLIR tool sources are committed here directly.

## Quick Reference

- Need to edit `generate_testcase.py`? Edit `../PTOAS-aloschilov/test/npu_validation/scripts/generate_testcase.py`
- Need to edit `PTOToEmitC.cpp`? Edit `../PTOAS-aloschilov/lib/PTO/Transforms/PTOToEmitC.cpp`
- Need to edit `cpu_sim_run.py`? Edit `mlir_tool/test/cpu_sim/cpu_sim_run.py` (this repo)
- Need to edit `e2e_all.sh`? Edit `mlir_tool/test/e2e_all.sh` (this repo)
