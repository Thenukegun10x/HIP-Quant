"""
hip_quant/gguf.py
=================

Minimal pure-stdlib GGUF v3 reader for hip_quant.

Parses the header, metadata KV store, and tensor infos of a ``.gguf``
file without loading tensor data. Tensor bytes are exposed as
``memoryview`` slices over an ``mmap`` so the GPU loader can stage them
with zero copies.

Layout reference (llama.cpp ``ggml/src/gguf.cpp``):
  magic u32 ``GGUF``, version u32, n_tensors u64, n_kv u64,
  then n_kv metadata pairs, n_tensors tensor infos, then tensor data
  starting at the first ``general.alignment`` (default 32) boundary
  after the infos. Each tensor's ``offset`` is relative to data start.

Notes:
  * GGUF stores dims reversed (``ne[0]`` is the fastest axis). ``shape``
    is returned in torch order (slowest-first), i.e. ``reversed(ne)``.
  * Type-size/block-size table mirrors ``ggml.c`` ``ggml_type_size``.
    Removed types (Q4_2/Q4_3/Q4_0_4_4/...) have size 0 and are rejected.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# ggml_type enum value -> (name, type_size_bytes, block_size_elements).
# Sizes verified against ggml/src/ggml.c + ggml/src/ggml-common.h.
GGML_TYPES: Dict[int, Tuple[str, int, int]] = {
    0:  ("F32",     4,   1),
    1:  ("F16",     2,   1),
    2:  ("Q4_0",    18,  32),
    3:  ("Q4_1",    20,  32),
    6:  ("Q5_0",    22,  32),
    7:  ("Q5_1",    24,  32),
    8:  ("Q8_0",    34,  32),
    9:  ("Q8_1",    36,  32),
    10: ("Q2_K",    84,  256),
    11: ("Q3_K",    110, 256),
    12: ("Q4_K",    144, 256),
    13: ("Q5_K",    176, 256),
    14: ("Q6_K",    210, 256),
    15: ("Q8_K",    292, 256),
    16: ("IQ2_XXS", 66,  256),
    17: ("IQ2_XS",  74,  256),
    18: ("IQ3_XXS", 98,  256),
    19: ("IQ1_S",   50,  256),
    20: ("IQ4_NL",  18,  32),
    21: ("IQ3_S",   110, 256),
    22: ("IQ2_S",   82,  256),
    23: ("IQ4_XS",  136, 256),
    24: ("I8",      1,   1),
    25: ("I16",     2,   1),
    26: ("I32",     4,   1),
    27: ("I64",     8,   1),
    28: ("F64",     8,   1),
    29: ("IQ1_M",   56,  256),
    30: ("BF16",    2,   1),
    34: ("TQ1_0",   54,  256),
    35: ("TQ2_0",   66,  256),
    39: ("MXFP4",   17,  32),
    40: ("NVFP4",   36,  64),
}

# GGUF metadata value types.
_GGUF_U8, _GGUF_I8, _GGUF_U16, _GGUF_I16 = 0, 1, 2, 3
_GGUF_U32, _GGUF_I32, _GGUF_F32, _GGUF_BOOL = 4, 5, 6, 7
_GGUF_STR, _GGUF_ARR, _GGUF_U64, _GGUF_I64, _GGUF_F64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    _GGUF_U8: "<B", _GGUF_I8: "<b", _GGUF_U16: "<H", _GGUF_I16: "<h",
    _GGUF_U32: "<I", _GGUF_I32: "<i", _GGUF_F32: "<f", _GGUF_BOOL: "<B",
    _GGUF_U64: "<Q", _GGUF_I64: "<q", _GGUF_F64: "<d",
}


class GGUFError(ValueError):
    """Raised for malformed GGUF files or unsupported types."""


@dataclass
class GGUFTensor:
    name: str
    ggml_type: int
    type_name: str
    shape: Tuple[int, ...]      # torch order (slowest-first)
    n_elements: int
    n_bytes: int
    offset: int                 # relative to data start
    index: int = 0


@dataclass
class GGUFFile:
    path: str
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: List[GGUFTensor] = field(default_factory=list)
    alignment: int = 32
    data_offset: int = 0
    file_size: int = 0
    _mmap: Optional[mmap.mmap] = field(default=None, repr=False)

    # -- queries ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tensors)

    def names(self) -> List[str]:
        return [t.name for t in self.tensors]

    def find(self, prefix: str) -> List[GGUFTensor]:
        return [t for t in self.tensors if t.name.startswith(prefix)]

    def get(self, name: str) -> GGUFTensor:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(f"tensor {name!r} not found in {self.path}")

    @property
    def arch(self) -> Optional[str]:
        return self.metadata.get("general.architecture")

    @property
    def n_params(self) -> int:
        return sum(t.n_elements for t in self.tensors)

    # -- data access -----------------------------------------------------
    def raw_bytes(self, tensor: GGUFTensor) -> memoryview:
        """Zero-copy view of a tensor's packed bytes (requires open())."""
        if self._mmap is None:
            raise GGUFError("GGUFFile is closed; use open() first")
        start = self.data_offset + tensor.offset
        return memoryview(self._mmap)[start:start + tensor.n_bytes]

    def open(self) -> "GGUFFile":
        if self._mmap is None:
            f = open(self.path, "rb")
            self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            f.close()
        return self

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __enter__(self) -> "GGUFFile":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()


class _Reader:
    """Buffered header reader: parses fields from 1 MiB blocks, never
    byte-by-byte and never touching tensor data. Peak RAM is one block."""

    _BLOCK = 1 << 20

    def __init__(self, f: BinaryIO) -> None:
        self._f = f
        self._buf = bytearray()
        self._pos = 0
        self.offset = 0  # absolute file offset of _buf[0]

    def _fill(self, n: int) -> None:
        while len(self._buf) - self._pos < n:
            # compact consumed prefix before growing
            if self._pos:
                del self._buf[:self._pos]
                self.offset += self._pos
                self._pos = 0
            chunk = self._f.read(max(n, self._BLOCK))
            if not chunk:
                raise GGUFError(f"truncated file: wanted {n} more bytes")
            self._buf += chunk

    def read(self, n: int) -> bytes:
        self._fill(n)
        out = bytes(self._buf[self._pos:self._pos + n])
        self._pos += n
        return out

    def tell(self) -> int:
        return self.offset + self._pos

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8")

    def value(self, vtype: int) -> Any:
        if vtype == _GGUF_STR:
            return self.string()
        if vtype == _GGUF_ARR:
            atype = self.u32()
            return [self.value(atype) for _ in range(self.u64())]
        fmt = _SCALAR_FMT.get(vtype)
        if fmt is None:
            raise GGUFError(f"unknown metadata type {vtype}")
        val = struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]
        return bool(val) if vtype == _GGUF_BOOL else val


def _align(offset: int, alignment: int) -> int:
    return offset + (-offset % alignment)


def load(path: str, parse_data: bool = False) -> GGUFFile:
    """Parse a GGUF file's header/metadata/tensor infos.

    With ``parse_data=False`` only the infos are read (fast, no mmap).
    Use ``gguf.open()`` afterwards for zero-copy tensor access.
    """
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _Reader(f)
        if r.u32() != GGUF_MAGIC:
            raise GGUFError(f"{path}: bad magic (not a GGUF file)")
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"{path}: unsupported GGUF version {version}")
        n_tensors = r.u64()
        n_kv = r.u64()

        metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            metadata[key] = r.value(r.u32())

        tensors: List[GGUFTensor] = []
        for i in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            ne = [r.u64() for _ in range(n_dims)]
            ggml_type = r.u32()
            offset = r.u64()
            info = GGML_TYPES.get(ggml_type)
            if info is None:
                raise GGUFError(
                    f"{path}: tensor {name!r} has unsupported ggml type {ggml_type}")
            type_name, type_size, block_size = info
            n_elements = 1
            for d in ne:
                n_elements *= d
            if n_elements % block_size != 0:
                raise GGUFError(
                    f"{path}: tensor {name!r} has {n_elements} elements, "
                    f"not a multiple of block size {block_size}")
            n_bytes = n_elements // block_size * type_size
            tensors.append(GGUFTensor(
                name=name, ggml_type=ggml_type, type_name=type_name,
                shape=tuple(reversed(ne)), n_elements=n_elements,
                n_bytes=n_bytes, offset=offset, index=i))

        alignment = int(metadata.get("general.alignment", 32))
        data_offset = _align(r.tell(), alignment)

    # bounds check: every tensor must lie inside the file, and tensor
    # data must not overlap the header.
    data_end = file_size - data_offset
    if data_end < 0:
        raise GGUFError(f"{path}: data offset beyond EOF")
    for t in tensors:
        if t.offset + t.n_bytes > data_end:
            raise GGUFError(
                f"{path}: tensor {t.name!r} extends past EOF")

    gf = GGUFFile(path=path, version=version, metadata=metadata,
                  tensors=tensors, alignment=alignment,
                  data_offset=data_offset, file_size=file_size)
    if parse_data:
        gf.open()
    return gf
