#!/usr/bin/env python3
"""Build reproducible runtime I-Quant codebook assets from source headers.

The native DLL deliberately does not include the multi-megabyte GGML-derived
tables.  It loads these versioned little-endian blobs from ``codebooks/`` at
runtime instead.  Run this script after updating a generated ``*_data.h``
source file, and use ``--check`` in CI/release validation.
"""

from __future__ import annotations

import argparse
import array
import re
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"HQIQCB01"
VERSION = 1
HEADER = struct.Struct("<8sIIIII")


@dataclass(frozen=True)
class ArraySpec:
    symbol: str
    ctype: str
    count: int


@dataclass(frozen=True)
class CodebookSpec:
    name: str
    header: str
    arrays: tuple[ArraySpec, ArraySpec, ArraySpec]


CODEBOOKS = (
    CodebookSpec(
        "iq1_s", "hip_quant_iq1s_data.h",
        (ArraySpec("h_iq1s_grid", "int8_t", 2048 * 8),
         ArraySpec("h_iq1s_map", "int", 43692),
         ArraySpec("h_iq1s_neighbours", "uint16_t", 1375339)),
    ),
    CodebookSpec(
        "iq2_xs", "hip_quant_iq2xs_data.h",
        (ArraySpec("h_iq2xs_grid", "int8_t", 512 * 8),
         ArraySpec("h_iq2xs_map", "int", 43692),
         ArraySpec("h_iq2xs_neighbours", "uint16_t", 551722)),
    ),
    CodebookSpec(
        "iq2_xxs", "hip_quant_iq2xxs_data.h",
        (ArraySpec("h_iq2xxs_grid", "int8_t", 256 * 8),
         ArraySpec("h_iq2xxs_map", "int", 43692),
         ArraySpec("h_iq2xxs_neighbours", "uint16_t", 417400)),
    ),
    CodebookSpec(
        "iq3_s", "hip_quant_iq3s_data.h",
        (ArraySpec("h_iq3s_grid", "int8_t", 512 * 4),
         ArraySpec("h_iq3s_map", "int", 4096),
         ArraySpec("h_iq3s_neighbours", "uint16_t", 28317)),
    ),
    CodebookSpec(
        "iq3_xxs", "hip_quant_iq3xxs_data.h",
        (ArraySpec("h_iq3xxs_grid", "int8_t", 256 * 4),
         ArraySpec("h_iq3xxs_map", "int", 4096),
         ArraySpec("h_iq3xxs_neighbours", "uint16_t", 22825)),
    ),
)


def _initializer_numbers(text: str, spec: ArraySpec) -> list[int]:
    match = re.search(
        rf"static\s+const\s+{re.escape(spec.ctype)}\s+{re.escape(spec.symbol)}"
        rf"(?:\s*\[[^\]]+\])+\s*=\s*\{{",
        text,
    )
    if match is None:
        raise ValueError(f"could not find {spec.symbol}")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                values = [int(value) for value in re.findall(r"-?\d+", text[start + 1:index])]
                if len(values) != spec.count:
                    raise ValueError(
                        f"{spec.symbol}: expected {spec.count} values, found {len(values)}"
                    )
                return values
    raise ValueError(f"unterminated initializer for {spec.symbol}")


def _pack_values(values: list[int], ctype: str) -> bytes:
    if ctype == "int8_t":
        return bytes(value & 0xFF for value in values)
    if ctype == "uint16_t":
        packed = array.array("H", values)
    elif ctype == "int":
        if array.array("i").itemsize != 4:
            raise RuntimeError("host Python does not have a 32-bit C int")
        packed = array.array("i", values)
    else:
        raise ValueError(f"unsupported C type {ctype}")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def expected_blob(root: Path, codebook: CodebookSpec) -> bytes:
    source = (root / codebook.header).read_text(encoding="utf-8")
    sections = [
        _pack_values(_initializer_numbers(source, array_spec), array_spec.ctype)
        for array_spec in codebook.arrays
    ]
    payload = b"".join(sections)
    return HEADER.pack(
        MAGIC,
        VERSION,
        len(sections[0]),
        len(sections[1]),
        len(sections[2]),
        zlib.crc32(payload) & 0xFFFFFFFF,
    ) + payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="fail if an asset is missing or stale")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = root / "codebooks"
    failures = []

    for codebook in CODEBOOKS:
        asset = destination / f"{codebook.name}.bin"
        blob = expected_blob(root, codebook)
        if args.check:
            if not asset.is_file() or asset.read_bytes() != blob:
                failures.append(str(asset))
            continue
        destination.mkdir(parents=True, exist_ok=True)
        if not asset.is_file() or asset.read_bytes() != blob:
            asset.write_bytes(blob)
            print(f"wrote {asset.relative_to(root)} ({len(blob):,} bytes)")
        else:
            print(f"up-to-date {asset.relative_to(root)}")

    if failures:
        for asset in failures:
            print(f"stale or missing codebook asset: {asset}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
