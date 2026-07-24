"""Small, NumPy-only reader for local SafeTensors diagnostics.

The normal SafeTensors PyTorch adapter is appropriate for quantization and
inference, but importing Torch can initialise the Windows ROCm runtime even
when an operation only needs to inspect a CPU checkpoint.  This module parses
the documented SafeTensors header and exposes copied NumPy arrays without
importing either ``torch`` or ``safetensors``.  It deliberately stays narrow:
it is for bounded, read-only analysis rather than checkpoint loading.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


_NUMPY_DTYPES: dict[str, np.dtype[Any]] = {
    "BOOL": np.dtype("?"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
}


class SafeTensorNumpyFile:
    """Read tensor metadata and bounded CPU arrays from one SafeTensors file.

    Returned arrays own their memory.  That avoids retaining an open memory
    map or file handle while callers compare an individual model tensor.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as source:
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"{self.path} is too short to be a SafeTensors file")
            self._header_size = struct.unpack("<Q", length_bytes)[0]
            if self._header_size > self.path.stat().st_size - 8:
                raise ValueError(f"{self.path} declares an invalid SafeTensors header length")
            header = source.read(self._header_size)
        try:
            decoded = json.loads(header)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{self.path} has an invalid SafeTensors JSON header") from error
        if not isinstance(decoded, dict):
            raise ValueError(f"{self.path} SafeTensors header must be a JSON object")
        self._header: dict[str, Any] = decoded
        self._data_start = 8 + self._header_size
        self._file_size = self.path.stat().st_size

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._header if name != "__metadata__")

    def descriptor(self, name: str) -> dict[str, Any]:
        try:
            descriptor = self._header[name]
        except KeyError as error:
            raise KeyError(f"{name!r} is absent from {self.path}") from error
        if name == "__metadata__" or not isinstance(descriptor, dict):
            raise ValueError(f"{name!r} is not a tensor descriptor in {self.path}")
        return descriptor

    def array(self, name: str) -> np.ndarray:
        """Return one tensor as a CPU NumPy copy, including BF16 conversion."""

        descriptor = self.descriptor(name)
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype_name, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name!r} has an invalid tensor descriptor in {self.path}")
        if not all(isinstance(value, int) and value >= 0 for value in shape):
            raise ValueError(f"{name!r} has an invalid shape in {self.path}")
        if not all(isinstance(value, int) and value >= 0 for value in offsets) or offsets[1] < offsets[0]:
            raise ValueError(f"{name!r} has invalid data offsets in {self.path}")
        begin, end = offsets
        byte_offset = self._data_start + begin
        nbytes = end - begin
        if byte_offset + nbytes > self._file_size:
            raise ValueError(f"{name!r} extends beyond the end of {self.path}")
        count = int(np.prod(shape, dtype=np.int64))

        if dtype_name == "BF16":
            if nbytes != count * 2:
                raise ValueError(f"{name!r} BF16 payload length does not match its shape")
            raw = np.memmap(self.path, mode="r", dtype=np.dtype("<u2"), offset=byte_offset, shape=tuple(shape))
            try:
                bits = np.array(raw, dtype=np.uint32, copy=True) << 16
            finally:
                del raw
            return bits.view(np.float32)

        try:
            dtype = _NUMPY_DTYPES[dtype_name]
        except KeyError as error:
            raise ValueError(f"{name!r} uses unsupported SafeTensors dtype {dtype_name!r}") from error
        if nbytes != count * dtype.itemsize:
            raise ValueError(f"{name!r} payload length does not match its shape and dtype")
        raw = np.memmap(self.path, mode="r", dtype=dtype, offset=byte_offset, shape=tuple(shape))
        try:
            return np.array(raw, copy=True)
        finally:
            del raw

    def rows(self, name: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Return a bounded leading-dimension slice without loading a full tensor.

        This is intentionally limited to a contiguous range in dimension zero.
        It is sufficient for representative matrix-codec screens while keeping
        a 12B BF16 checkpoint out of RAM and avoiding a Torch/ROCm import.
        """

        descriptor = self.descriptor(name)
        dtype_name = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype_name, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name!r} has an invalid tensor descriptor in {self.path}")
        if not shape or not all(isinstance(value, int) and value >= 0 for value in shape):
            raise ValueError(f"{name!r} has an invalid shape in {self.path}")
        if not all(isinstance(value, int) and value >= 0 for value in offsets) or offsets[1] < offsets[0]:
            raise ValueError(f"{name!r} has invalid data offsets in {self.path}")
        rows = int(shape[0])
        start = int(start)
        stop = rows if stop is None else int(stop)
        if start < 0 or stop < start or stop > rows:
            raise ValueError(f"Row range [{start}, {stop}) is invalid for {name!r} with {rows} rows")
        row_values = int(np.prod(shape[1:], dtype=np.int64))

        if dtype_name == "BF16":
            element_dtype = np.dtype("<u2")
        else:
            try:
                element_dtype = _NUMPY_DTYPES[dtype_name]
            except KeyError as error:
                raise ValueError(f"{name!r} uses unsupported SafeTensors dtype {dtype_name!r}") from error
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * element_dtype.itemsize
        if offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(f"{name!r} payload length does not match its shape and dtype")
        byte_offset = self._data_start + offsets[0] + start * row_values * element_dtype.itemsize
        count = (stop - start) * row_values
        if byte_offset + count * element_dtype.itemsize > self._file_size:
            raise ValueError(f"{name!r} row slice extends beyond the end of {self.path}")
        raw = np.memmap(
            self.path,
            mode="r",
            dtype=element_dtype,
            offset=byte_offset,
            shape=(stop - start, *shape[1:]),
        )
        try:
            if dtype_name == "BF16":
                bits = np.array(raw, dtype=np.uint32, copy=True) << 16
                return bits.view(np.float32)
            return np.array(raw, copy=True)
        finally:
            del raw
