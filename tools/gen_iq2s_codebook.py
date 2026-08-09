#!/usr/bin/env python3
"""Generate hip_quant_iq2s_data.h from the llama.cpp reference tables.

Extracts kgrid_2bit_1024 from the local llama.cpp checkout (ggml-quants.c),
builds the int8 grid, the 43692-entry map, and the nwant=1 neighbour lists
using the exact same algorithm as llama.cpp's iq2xs_init_impl.

The generator self-verifies: it also rebuilds the IQ1_S tables (nwant=3,
kgrid_1bit_2048) from the same source and compares them byte-for-byte
against the committed hip_quant_iq1s_data.h.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUANTS_C = ROOT / "Own Quant" / "llama_cpp_stock" / "ggml" / "src" / "ggml-quants.c"
IQ1S_HEADER = ROOT / "hip_quant_iq1s_data.h"
OUT_HEADER = ROOT / "hip_quant_iq2s_data.h"

KGRID_SYMBOL = "kgrid_2bit_1024"
NGRID = 1024
KMAP_SIZE = 43692
NWANT_IQ2S = 1


def extract_kgrid(text: str, symbol: str, expected: int) -> list[int]:
    match = re.search(
        rf"static\s+const\s+uint16_t\s+{re.escape(symbol)}\s*\[\s*(?:[A-Za-z_][A-Za-z0-9_]*|\d+)\s*\]\s*=\s*\{{",
        text,
    )
    if match is None:
        raise ValueError(f"could not find {symbol} in {QUANTS_C}")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                values = [int(v) for v in re.findall(r"\d+", text[start + 1:index])]
                if len(values) != expected:
                    raise ValueError(f"{symbol}: expected {expected} values, found {len(values)}")
                return values
    raise ValueError(f"unterminated initializer for {symbol}")


def build_tables(kgrid: list[int], nwant: int) -> tuple[list[list[int]], list[int], list[int]]:
    """grid (NGRID x 8 int8), map (43692 int), neighbours (flat uint16)."""
    grid = [[2 * ((code >> 2 * k) & 0x3) + 1 for k in range(8)] for code in kgrid]

    kmap = [-1] * KMAP_SIZE
    for i, row in enumerate(grid):
        index = 0
        for k in range(8):
            index |= ((row[k] - 1) // 2) << 2 * k
        kmap[index] = i

    n_per_i = [0] * KMAP_SIZE
    num_neighbors = 0
    num_not_in_map = 0
    for i in range(KMAP_SIZE):
        if kmap[i] >= 0:
            continue
        num_not_in_map += 1
        pos = [2 * ((i >> 2 * k) & 0x3) + 1 for k in range(8)]
        dist2 = sorted(
            (sum((row[k] - pos[k]) ** 2 for k in range(8)), j) for j, row in enumerate(grid)
        )
        n = 0
        d2 = dist2[0][0]
        nhave = 1
        for d, _ in dist2:
            if d > d2:
                if nhave == nwant:
                    break
                d2 = d
                nhave += 1
            n += 1
        n_per_i[i] = n
        num_neighbors += n

    offsets = [0] * KMAP_SIZE
    counter = 0
    for i in range(KMAP_SIZE):
        if kmap[i] >= 0:
            offsets[i] = -1
            continue
        offsets[i] = counter
        counter += 1 + n_per_i[i]

    neighbours = [0] * (num_neighbors + num_not_in_map)
    for i in range(KMAP_SIZE):
        if kmap[i] >= 0:
            continue
        local_counter = offsets[i]
        kmap[i] = -(local_counter + 1)
        pos = [2 * ((i >> 2 * k) & 0x3) + 1 for k in range(8)]
        dist2 = sorted(
            (sum((row[k] - pos[k]) ** 2 for k in range(8)), j) for j, row in enumerate(grid)
        )
        start = local_counter
        local_counter += 1
        n = 0
        d2 = dist2[0][0]
        nhave = 1
        for d, j in dist2:
            if d > d2:
                if nhave == nwant:
                    break
                d2 = d
                nhave += 1
            neighbours[local_counter] = j
            local_counter += 1
            n += 1
        neighbours[start] = n

    return grid, kmap, neighbours


def load_header_values(path: Path, symbol: str, expected: int) -> list[int]:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"static\s+const\s+(?:int|uint16_t|int8_t)\s+{re.escape(symbol)}\s*(\[[^\]]+\])+\s*=\s*\{{", text)
    if match is None:
        raise ValueError(f"could not find {symbol} in {path}")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                values = [int(v) for v in re.findall(r"-?\d+", text[start + 1:index])]
                if len(values) != expected:
                    raise ValueError(f"{symbol}: expected {expected} values, found {len(values)}")
                return values
    raise ValueError(f"unterminated initializer for {symbol}")


def verify_against_iq1s(quants_text: str) -> None:
    """Rebuild IQ1_S tables from llama.cpp and compare with the committed header."""
    kgrid = extract_kgrid(quants_text, "kgrid_1bit_2048", 2048)
    grid, kmap, neighbours = build_tables(kgrid, nwant=3)

    ref_grid = load_header_values(IQ1S_HEADER, "h_iq1s_grid", 2048 * 8)
    ref_map = load_header_values(IQ1S_HEADER, "h_iq1s_map", 43692)
    ref_neighbours = load_header_values(IQ1S_HEADER, "h_iq1s_neighbours", 1375339)

    flat = [v for row in grid for v in row]
    problems = []
    if flat != ref_grid:
        problems.append(f"grid mismatch: {sum(a != b for a, b in zip(flat, ref_grid))} of {len(flat)}")
    if kmap != ref_map:
        problems.append(f"map mismatch: {sum(a != b for a, b in zip(kmap, ref_map))} of {len(kmap)}")
    if neighbours != ref_neighbours:
        problems.append(
            f"neighbours mismatch: {sum(a != b for a, b in zip(neighbours, ref_neighbours))} of {len(neighbours)}"
        )
    if problems:
        raise SystemExit("self-check failed vs hip_quant_iq1s_data.h:\n  " + "\n  ".join(problems))
    print(f"self-check OK: IQ1_S tables rebuilt from llama.cpp match the committed header ({len(neighbours)} neighbours)")


def write_header(grid, kmap, neighbours) -> None:
    lines = ["#pragma once", "#include <stdint.h>", ""]
    lines.append(f"static const int8_t h_iq2s_grid[{NGRID}][8] = {{")
    for row in grid:
        lines.append("    {" + ", ".join(str(v) for v in row) + "},")
    lines.append("};")
    lines.append("")
    lines.append(f"static const int h_iq2s_map[{KMAP_SIZE}] = {{")
    for v in kmap:
        lines.append(f"    {v},")
    lines.append("};")
    lines.append("")
    lines.append(f"static const uint16_t h_iq2s_neighbours[{len(neighbours)}] = {{")
    for v in neighbours:
        lines.append(f"    {v},")
    lines.append("};")
    lines.append("")
    lines.append(f"static const int h_iq2s_neighbours_len = {len(neighbours)};")
    lines.append("")
    OUT_HEADER.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_HEADER.relative_to(ROOT)} (grid {NGRID * 8}, map {KMAP_SIZE}, neighbours {len(neighbours)})")


def main() -> int:
    text = QUANTS_C.read_text(encoding="utf-8")
    verify_against_iq1s(text)
    kgrid = extract_kgrid(text, KGRID_SYMBOL, NGRID)
    grid, kmap, neighbours = build_tables(kgrid, nwant=NWANT_IQ2S)
    write_header(grid, kmap, neighbours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
