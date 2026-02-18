# Multi-stage: builder = LLVM (from Triton's llvm-hash.txt) + PTOAS + Triton + triton-to-pto;
# runtime = ptoas:py3.11 + triton-to-pto-mlir.
# Build PTOAS image first: from PTOAS repo, docker build -f docker/Dockerfile docker/ -t ptoas:py3.11

ARG ARCH=x86_64

# -----------------------------------------------------------------------------
# Stage 1: build triton-to-pto; LLVM = exact commit from Triton's cmake/llvm-hash.txt
# -----------------------------------------------------------------------------
FROM quay.io/pypa/manylinux_2_34_${ARCH} AS builder
ARG ARCH
# Triton ref: drives LLVM version via cmake/llvm-hash.txt; v3.1.0 pins LLVM 19.0.0 to minimize PTOAS adjustments
ARG TRITON_REF=v3.1.0
# Optional: pin PTOAS for reproducibility (empty = use default clone)
ARG PTOAS_REF=

ENV PY_PATH="/opt/python/cp311-cp311"
ENV PATH="${PY_PATH}/bin:${PATH}"

RUN dnf install -y ninja-build cmake git ccache gcc-c++ lld zip && dnf clean all
# MLIR Python bindings use def_property_readonly + keep_alive, disallowed in pybind11 3.x
RUN pip install --no-cache-dir "pybind11>=2.6,<3" numpy nanobind setuptools wheel auditwheel

ENV WORKSPACE_DIR=/llvm-workspace
ENV LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
ENV LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
ENV PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
ENV TRITON_SOURCE_DIR=$WORKSPACE_DIR/triton
ENV TRITON_TO_PTO_SOURCE_DIR=$WORKSPACE_DIR/triton-to-pto

# Clone Triton first so we can read its pinned LLVM commit
WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/triton-lang/triton.git $TRITON_SOURCE_DIR \
    && cd $TRITON_SOURCE_DIR && git checkout $TRITON_REF \
    && tr -d ' \n\r' < cmake/llvm-hash.txt > /tmp/llvm_commit.txt

# Clone LLVM at the exact commit Triton pins (guarantees Triton + LLVM build without patches)
RUN git clone https://github.com/llvm/llvm-project.git $LLVM_SOURCE_DIR \
    && cd $LLVM_SOURCE_DIR && git checkout $(cat /tmp/llvm_commit.txt)

WORKDIR $LLVM_SOURCE_DIR
RUN cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=${PY_PATH}/bin/python \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host"

ARG NINJA_JOBS=4
RUN ninja -j $NINJA_JOBS -C $LLVM_BUILD_DIR

# Clone PTOAS (triton-to-pto will build PTOIR via add_subdirectory)
WORKDIR $WORKSPACE_DIR
RUN git clone https://github.com/zhangstevenunity/PTOAS.git
RUN if [ -n "$PTOAS_REF" ]; then cd PTOAS && git checkout $PTOAS_REF && cd ..; fi

# Copy triton-to-pto from build context (build from repo root: docker build -f docker/Dockerfile .)
COPY . $TRITON_TO_PTO_SOURCE_DIR

# Apply PTOAS LLVM 19 compatibility fixes (LogicalResult in mlir, EffectInstance API)
RUN PTO_CPP=$PTO_SOURCE_DIR/lib/PTO/IR/PTO.cpp && \
    sed -i 's/llvm::LogicalResult mlir::pto::CvtOp_DPS::verify()/mlir::LogicalResult mlir::pto::CvtOp_DPS::verify()/g' "$PTO_CPP" && \
    sed -i 's/llvm::LogicalResult mlir::pto::GatherOp_DPS::verify()/mlir::LogicalResult mlir::pto::GatherOp_DPS::verify()/g' "$PTO_CPP" && \
    sed -i 's/llvm::LogicalResult mlir::pto::TCvtOp::verify()/mlir::LogicalResult mlir::pto::TCvtOp::verify()/g' "$PTO_CPP" && \
    sed -i 's/llvm::LogicalResult mlir::pto::TGatherOp::verify()/mlir::LogicalResult mlir::pto::TGatherOp::verify()/g' "$PTO_CPP" && \
    sed -i 's/effects.emplace_back(effect, operand, SideEffects::DefaultResource::get())/effects.emplace_back(effect, SideEffects::DefaultResource::get())/g' "$PTO_CPP"

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
