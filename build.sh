#!/bin/bash
set -euo pipefail

OUTPUT="${OUTPUT:-libhip_quantize.so}"
CDNA="${CDNA:-false}"
ALL="${ALL:-false}"
ARCH="${ARCH:-}"

ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
HIPCC="${ROCM_HOME}/bin/hipcc"

if [ ! -f "$HIPCC" ]; then
    echo "hipcc not found at $HIPCC"
    echo "Set ROCM_HOME to your ROCm installation directory"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Compiling HIP quantization library..."

ARCHS=()

if [ "$ALL" = "true" ]; then
    ARCHS=("gfx90a" "gfx942" "gfx1100" "gfx1101" "gfx1102" "gfx1103" "gfx1200" "gfx1201")
elif [ "$CDNA" = "true" ]; then
    ARCHS=("gfx90a" "gfx942" "gfx1200" "gfx1201")
    echo "Building with CDNA + RDNA4 support"
elif [ -n "$ARCH" ]; then
    IFS=',' read -ra ARCHS <<< "$ARCH"
else
    ARCHS=("gfx90a" "gfx942" "gfx1100" "gfx1101" "gfx1102" "gfx1103" "gfx1200" "gfx1201")
fi

echo "Target architectures: ${ARCHS[*]}"

OFFLOAD_ARGS=()
for a in "${ARCHS[@]}"; do
    OFFLOAD_ARGS+=("--offload-arch=$a")
done

ARGS=(
    "-O3"
    "-mno-wavefrontsize64"
    "-ffp-contract=off"
    "-shared"
    "-fPIC"
    "-Wno-ignored-attributes"
    "-I" "$SCRIPT_DIR"
)

for a in "${ARCHS[@]}"; do
    if [[ "$a" == gfx9* ]]; then
        ARGS+=("-DHIP_QUANT_HAS_CDNA=1")
        break
    fi
done

ARGS+=("${OFFLOAD_ARGS[@]}")
ARGS+=("-o" "${SCRIPT_DIR}/${OUTPUT}" "${SCRIPT_DIR}/hip_quantize.cpp")

"$HIPCC" "${ARGS[@]}"

echo "Library created: ${OUTPUT}"
echo "Architectures: ${ARCHS[*]}"
