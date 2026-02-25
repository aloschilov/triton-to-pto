# Multi-stage: builder = pre-built LLVM (ptoas-llvm:19) + Triton + PTOAS + triton-to-pto;
# runtime = ptoas:py3.11 + triton-to-pto-mlir.
#
# Build order (no LLVM rebuild on triton-to-pto changes):
#   1. Build ptoas-llvm:19 once (from PTOAS repo): docker build -f docker/Dockerfile.llvm -t ptoas-llvm:19 docker/empty
#   2. Build ptoas:py3.11 (from PTOAS repo)
#   3. Build this image: docker build -f docker/Dockerfile -t triton-to-pto:py3.11 .

ARG ARCH=x86_64

# -----------------------------------------------------------------------------
# Stage 1: build triton-to-pto on top of pre-built LLVM (no LLVM clone/build)
# -----------------------------------------------------------------------------
FROM ptoas-llvm:19 AS builder
# Triton ref for source (must match LLVM in base; ptoas-llvm:19 uses Triton v3.1.0 pin)
ARG TRITON_REF=v3.1.0
# Optional: pin PTOAS for reproducibility (empty = use default clone)
ARG PTOAS_REF=

# Base image already sets WORKSPACE_DIR, LLVM_SOURCE_DIR, LLVM_BUILD_DIR, PY_PATH
ENV TRITON_SOURCE_DIR=$WORKSPACE_DIR/triton
ENV TRITON_TO_PTO_SOURCE_DIR=$WORKSPACE_DIR/triton-to-pto
# PTO_SOURCE_DIR already set in base; ensure it is what we use
ENV PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS

# Clone Triton (source only; LLVM already built in base)
WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/triton-lang/triton.git $TRITON_SOURCE_DIR \
    && cd $TRITON_SOURCE_DIR && git checkout $TRITON_REF

# Clone PTOAS (triton-to-pto will build PTOIR via add_subdirectory)
RUN git clone https://github.com/aloschilov/PTOAS.git $PTO_SOURCE_DIR
RUN if [ -n "$PTOAS_REF" ]; then cd $PTO_SOURCE_DIR && git checkout $PTOAS_REF && cd ..; fi

# Copy triton-to-pto from build context (build from repo root: docker build -f docker/Dockerfile .)
COPY . $TRITON_TO_PTO_SOURCE_DIR

# Build triton-to-pto (builds PTOAS and Triton as subprojects)
RUN bash $TRITON_TO_PTO_SOURCE_DIR/docker/configure_mlir_tool.sh

WORKDIR $TRITON_TO_PTO_SOURCE_DIR/mlir_tool
# Build Triton GPU tablegen targets first so TritonGPUAttrInterfaces.h.inc exists before TritonIR
RUN ninja -C build TritonGPUAttrDefsIncGen TritonGPUTableGen 2>/dev/null || true
RUN ninja -C build triton-to-pto-mlir

# Optional: collect .so deps for triton-to-pto-mlir (for runtime if needed)
ENV T2P_DEPS_DIR=/triton-to-pto-deps
COPY docker/copy_triton_to_pto_deps.sh /tmp/copy_triton_to_pto_deps.sh
RUN mkdir -p $T2P_DEPS_DIR && bash /tmp/copy_triton_to_pto_deps.sh

# -----------------------------------------------------------------------------
# Stage 2: runtime = PTOAS image + triton-to-pto-mlir
# -----------------------------------------------------------------------------
FROM ptoas:py3.11 AS runtime

# Copy converter binary
COPY --from=builder /llvm-workspace/triton-to-pto/mlir_tool/build/tools/triton-to-pto-mlir/triton-to-pto-mlir /usr/local/bin/triton-to-pto-mlir

# Copy any extra .so required by triton-to-pto-mlir (same lib path as ptoas)
COPY --from=builder /triton-to-pto-deps/. /usr/local/lib/ptoas/

RUN triton-to-pto-mlir --help || true

WORKDIR /workspace
CMD ["/bin/bash"]
