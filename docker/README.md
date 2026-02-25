# Reproducible Docker build (triton-to-pto)

This directory provides a Docker-based build that uses a **pre-built LLVM base image** (`ptoas-llvm:19`) so that triton-to-pto image rebuilds do not rebuild LLVM. The builder stage only clones Triton (source), clones PTOAS, copies this repo, and runs configure + ninja. PTOAS is cloned from the [aloschilov/PTOAS](https://github.com/aloschilov/PTOAS) fork. The image extends the PTOAS runtime with `triton-to-pto-mlir`.

## Build order

1. **Build the LLVM base image once** (from the PTOAS repository; requires an empty dir `docker/empty` or use `docker build -f docker/Dockerfile.llvm -t ptoas-llvm:19 .` with a minimal context):
   ```bash
   # From PTOAS repo root; create empty context if needed: mkdir -p docker/empty
   docker build -f docker/Dockerfile.llvm -t ptoas-llvm:19 docker/empty
   ```
   This image contains LLVM at the commit pinned by Triton v3.1.0 (e.g. 10dc3a8e...). Rebuild only when you need a different LLVM version.

2. **Build the PTOAS image** (once). From the PTOAS repository root:
   ```bash
   docker build -f docker/Dockerfile . -t ptoas:py3.11
   ```
   See PTOAS `docker/README.md` for details.

3. **Build the triton-to-pto image.** From the **triton-to-pto** repository root:
   ```bash
   docker build -f docker/Dockerfile -t triton-to-pto:py3.11 .
   ```
   Build context must be the repo root so `mlir_tool/` and `docker/` are available. This step does **not** rebuild LLVM; it only clones Triton, clones PTOAS, copies this repo, and builds the converter.

## Build arguments

- **`TRITON_REF`** (default: `v3.1.0`) – Git branch, tag, or commit for the [Triton](https://github.com/triton-lang/triton) repo (source only; LLVM comes from the base image). Must match the LLVM commit in `ptoas-llvm:19` (v3.1.0 pin).
- **`PTOAS_REF`** (default: empty) – Optional. Git ref to checkout after cloning PTOAS in the builder (e.g. a tag or commit) for reproducibility.

Example with a different Triton ref (only use a ref whose llvm-hash.txt matches the LLVM in `ptoas-llvm:19`, e.g. v3.1.0):
```bash
docker build -f docker/Dockerfile -t triton-to-pto:py3.11 \
  --build-arg TRITON_REF=v3.1.0 .
```

## Triton ↔ LLVM version mapping

The **pre-built base image** `ptoas-llvm:19` contains LLVM at a fixed commit (same as Triton v3.1.0’s pin in `cmake/llvm-hash.txt`, e.g. 10dc3a8e...). The triton-to-pto build does not clone or build LLVM; it only uses that base. Triton pins a specific **llvm-project** commit in **cmake/llvm-hash.txt** (see [Triton README](https://github.com/triton-lang/triton)).

- **How to check any Triton ref:**  
  `curl -sL "https://raw.githubusercontent.com/triton-lang/triton/<REF>/cmake/llvm-hash.txt"`  
  or clone Triton and run `cat cmake/llvm-hash.txt` after checking out the ref.
- **LLVM version for that commit:**  
  `https://raw.githubusercontent.com/llvm/llvm-project/<HASH>/cmake/Modules/LLVMVersion.cmake`

**Default TRITON_REF=v3.1.0** pins commit **10dc3a8e916d73291269e5e2b82dd22681489aa1** (LLVM 19.0.0git), chosen to minimize drift from PTOAS's typical 19.1.7 environment. If a future Triton ref pins a 19.1.x commit, the default can be updated.

**Re-running the check:**  
From the repo root (with network):  
`bash docker/scripts/check_triton_llvm_hash.sh`  
This script resolves llvmorg-19.1.7 to a commit, fetches llvm-hash.txt for several Triton refs, and reports if any ref gives an exact match. You can use it to pick a Triton ref whose pinned LLVM is closest to a given tag.

## Running the converter and E2E

The image includes `triton-to-pto-mlir` on `PATH` (installed at `/usr/local/bin/triton-to-pto-mlir`).

**E2E (Triton → .pto → PTOAS sim):** mount the triton-to-pto repo and (if needed) the PTOAS repo, then run the e2e script:
```bash
# From triton-to-pto repo root; PTOAS repo at /path/to/PTOAS
docker run --rm -v "$(pwd)":/workspace -v /path/to/PTOAS:/path/to/PTOAS -w /workspace \
  -e PTOAS_ROOT=/path/to/PTOAS \
  triton-to-pto:py3.11 bash mlir_tool/test/e2e_run_ptoas_sim.sh
```
If the script expects PTOAS at `/workspace` (e.g. it uses `PTOAS_ROOT=/workspace`), mount PTOAS at `/workspace` and triton-to-pto elsewhere, or set `PTOAS_ROOT` to the path where you mounted PTOAS inside the container.

**Quick test:**
```bash
docker run --rm triton-to-pto:py3.11 triton-to-pto-mlir --help
```

## Known issues / troubleshooting

- **Previous linker blocker:** The `triton-to-pto-mlir` executable used to fail to link with `undefined reference to mlir::getReshapeDecomposition(llvm::ArrayRef<long>, llvm::ArrayRef<long>)'` (from Triton’s TritonGPU dialect). The fix was to link **TritonAnalysis** (Triton’s lib/Analysis) in [mlir_tool/tools/triton-to-pto-mlir/CMakeLists.txt](../mlir_tool/tools/triton-to-pto-mlir/CMakeLists.txt) so the dependency chain that provides that symbol is included. If the symbol moves in a future LLVM/Triton version, you can locate it with: `nm -D /path/to/llvm-build/lib/libMLIR*.so 2>/dev/null | grep getReshapeDecomposition` (or add the corresponding CMake target).

## Version summary

- **Build status:** Image builds with LLVM 19.0.0 (Triton’s pinned commit), Triton v3.1.0, PTOAS + in-repo compatibility fixes, and triton-to-pto-mlir linking TritonAnalysis for the reshape-decomposition symbol.
- **LLVM:** exact commit from Triton's `cmake/llvm-hash.txt` (e.g. for v3.1.0: **10dc3a8e916d73291269e5e2b82dd22681489aa1**, LLVM 19.0.0git). No fixed tag; Triton drives the version.
- **Triton:** default `TRITON_REF=v3.1.0`. Override with build-arg for another ref.
- **PTOAS:** cloned from [aloschilov/PTOAS](https://github.com/aloschilov/PTOAS) (fork with LLVM 19 compatibility applied upstream). Use build-arg **`PTOAS_REF`** to pin a branch or tag of that fork. Deprecation warnings may remain.
- **pybind11:** pinned to `>=2.6,<3` in the builder; MLIR Python bindings use `keep_alive` with `def_property_readonly`, which pybind11 3.x disallows.
- **Runtime base:** `ptoas:py3.11` (must exist before building this image).
