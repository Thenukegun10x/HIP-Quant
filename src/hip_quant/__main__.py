import argparse
import sys

import numpy as np

from . import GGML_TYPE, HipQuant, normalize_type, supported_types, type_name


def main(argv=None):
    parser = argparse.ArgumentParser(description="Quantize a .npy float32 tensor with HIP-Quant.")
    parser.add_argument("input", nargs="?", help="Input .npy file containing a 1-D or 2-D float32 array")
    parser.add_argument("output", nargs="?", help="Output .bin file for packed GGML bytes")
    parser.add_argument("--type", "-t", default="Q4_K", help="GGML type name or ID, e.g. Q4_K or 12")
    parser.add_argument("--imatrix", help="Optional .npy importance matrix")
    parser.add_argument("--dll", help="Path to hip_quantize.dll")
    parser.add_argument("--rows", action="store_true", help="Write row-shaped bytes internally before flattening")
    parser.add_argument("--allow-missing-imatrix", action="store_true", help="Allow IQ types that normally require an imatrix")
    parser.add_argument("--info", action="store_true", help="Print device and supported type info")
    parser.add_argument("--list-types", action="store_true", help="List supported GGML types and exit")
    args = parser.parse_args(argv)

    if args.list_types:
        for name, type_id in sorted(supported_types().items(), key=lambda item: item[1]):
            print(f"{type_id:>2}  {name}")
        return 0

    hq = HipQuant(args.dll)
    if args.info:
        print(f"DLL: {hq.dll_path}")
        print(f"Device: {hq.device_name}")
        print(f"Device count: {hq.device_count}")
        if not args.input and not args.output:
            return 0

    if not args.input or not args.output:
        parser.error("input and output are required unless using --info without files or --list-types")

    type_id = normalize_type(args.type)
    arr = np.load(args.input)
    imatrix = np.load(args.imatrix) if args.imatrix else None
    require_imatrix = not args.allow_missing_imatrix

    if args.rows:
        out = hq.quantize_rows(arr, type_id, imatrix=imatrix, require_imatrix=require_imatrix).reshape(-1)
    else:
        out = hq.quantize_numpy(arr, type_id, imatrix=imatrix, require_imatrix=require_imatrix)

    with open(args.output, "wb") as f:
        f.write(out.tobytes())

    rows, n_per_row = (arr.reshape(1, -1).shape if arr.ndim == 1 else arr.shape)
    print(f"Quantized {rows}x{n_per_row} to {type_name(type_id)} ({type_id})")
    print(f"Wrote {out.size} bytes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
