import argparse
import sys

import numpy as np

from . import GGML_TYPE, HipQuant


def _normalize_type(value):
    if isinstance(value, int):
        return value
    text = str(value).upper()
    if text in GGML_TYPE:
        return GGML_TYPE[text]
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"unknown quantization type: {value}") from exc


def _type_name(type_id):
    for name, value in GGML_TYPE.items():
        if value == type_id:
            return name
    return str(type_id)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hip-quant",
        description="Quantize .npy tensors with the HIP-Quant NumPy/DLL backend.",
        epilog=(
            "Note: this CLI is for offline conversion. It is not a PyTorch training "
            "extension and does not preserve autograd. Fused ROCm WMMA PyTorch ops "
            "are exposed as hip_quant.fp8_linear_forward, "
            "hip_quant.fp8_linear_backward_input, and "
            "hip_quant.fp8_linear_backward_weight after building _C; scaled variants "
            "are used by Fp8ScaledLinear and Fp8ShadowLinear."
        ),
    )
    parser.add_argument("input", nargs="?", help="input .npy file containing a 1-D or 2-D array")
    parser.add_argument("output", nargs="?", help="output .bin file for packed quantized bytes")
    parser.add_argument("--type", "-t", default="Q4_K", help="quantization type name or ID, e.g. Q4_K, F8_E4M3, F8_E5M2, or 12")
    parser.add_argument("--fp8-source", choices=("E4M3", "E5M2"), help="treat input as uint8 FP8 bytes and expand on GPU before quantizing")
    parser.add_argument("--imatrix", help="optional .npy importance matrix")
    parser.add_argument("--dll", help="path to hip_quantize.dll")
    parser.add_argument("--info", action="store_true", help="print DLL/device information")
    parser.add_argument("--compat", action="store_true", help="print CDNA/RDNA compatibility report")
    parser.add_argument("--emulate", choices=("auto", "cpu", "gpu"), default=None, help="set emulation mode for CDNA testing")
    parser.add_argument("--list-types", action="store_true", help="list supported quantization types and exit")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_types:
        for name, type_id in sorted(GGML_TYPE.items(), key=lambda item: item[1]):
            print(f"{type_id:>2}  {name}")
        return 0

    if args.emulate:
        from .cdna_compat import set_emulation_mode
        set_emulation_mode(args.emulate)

    if args.compat:
        from . import suggest_cdna_emulation
        print(suggest_cdna_emulation())
        if not args.input and not args.output and not args.info:
            return 0

    hq = HipQuant(args.dll)
    if args.info:
        from . import report_device
        print(report_device(args.dll))
        if not args.input and not args.output:
            return 0

    if not args.input or not args.output:
        parser.error("input and output are required unless using --info without files or --list-types")

    type_id = _normalize_type(args.type)
    arr = np.load(args.input)
    imatrix = np.load(args.imatrix) if args.imatrix else None

    if args.fp8_source:
        out = hq.quantize_from_fp8(arr, type_id, imatrix=imatrix, source_format=args.fp8_source)
    else:
        out = hq.quantize_numpy(arr, type_id, imatrix=imatrix)

    with open(args.output, "wb") as f:
        f.write(out.tobytes())

    logical = arr.reshape(1, -1) if arr.ndim == 1 else arr
    print(f"Quantized {logical.shape[0]}x{logical.shape[1]} to {_type_name(type_id)} ({type_id})")
    print(f"Wrote {out.size} bytes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
