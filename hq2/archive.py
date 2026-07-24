"""Direct-loadable HQ-family model archives.

The first archive revision was deliberately HQ2-only.  Version 2 keeps the
same HQ2 72-byte block payload bit-for-bit, but makes the *container* an
HQ-family envelope: each tensor points at a versioned format descriptor.  An
HQ1 or HQ3 implementation can therefore introduce its own block layout and
kernel without inventing another model-file format or invalidating HQ2 files.

Payloads are always uncompressed and 4 KiB aligned.  The compact index is
written at the end of the file once streaming conversion finishes, so model
payloads can start at the first page rather than after a one-megabyte reserved
JSON header.  This is intentionally an inference-oriented format, not a
general-purpose GGUF replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, Iterator, Mapping

import numpy as np

from .format import (
    BITS_PER_WEIGHT,
    BLOCK_BYTES,
    BLOCK_SIZE,
    FORMAT_NAME,
    FORMAT_VERSION,
    HQ2Tensor,
    packed_nbytes,
    validate_shape,
)
from .hq3 import (
    HQ3_BITS_PER_WEIGHT,
    HQ3_BLOCK_BYTES,
    HQ3_BLOCK_SIZE,
    HQ3_FORMAT_NAME,
    HQ3_FORMAT_VERSION,
    HQ3Tensor,
    hq3_packed_nbytes,
    validate_hq3_shape,
)
from .hq8 import (
    HQ8_G128_BITS_PER_WEIGHT,
    HQ8_G128_BLOCK_BYTES,
    HQ8_G128_BLOCK_SIZE,
    HQ8_G128_FORMAT_NAME,
    HQ8_G128_FORMAT_VERSION,
    HQ8Tensor,
    hq8_g128_packed_nbytes,
    validate_hq8_g128_shape,
)
from .raw import HQRawTensor, is_raw_descriptor


# v1 was HQ2-only and must remain readable because it was used for the first
# Gemma conversion.  v2 has a format-neutral magic and footer index.
LEGACY_MODEL_MAGIC = b"HQ2MODL\x00"
LEGACY_MODEL_VERSION = 1
MODEL_MAGIC = b"HQMODL2\x00"
MODEL_VERSION = 2
PAYLOAD_ALIGNMENT = 4096

_LEGACY_HEADER = struct.Struct("<8sIQQ")
_HEADER = struct.Struct("<8sIIQQI28x")  # exactly one 64-byte header record
_INDEX_HEADER = struct.Struct("<4sIIIIII")
_ENTRY = struct.Struct("<IIHBBIQQII")
_INDEX_MAGIC = b"HQIX"
_INDEX_VERSION = 1

_FORMAT_ID_HQ2 = 1
_FLAG_IMPORTANCE_WEIGHTED = 1
_U32_MAX = (1 << 32) - 1
_ARCHIVE_SUFFIXES = frozenset({".hq", ".hq1", ".hq2", ".hq3", ".hq8"})


def _aligned(value: int, alignment: int = PAYLOAD_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _require_archive_path(path: Path) -> None:
    if path.suffix.lower() not in _ARCHIVE_SUFFIXES:
        allowed = ", ".join(sorted(_ARCHIVE_SUFFIXES))
        raise ValueError(f"HQ model archives must use one of {allowed}, got {path}")


def _json_bytes(value: Any, *, what: str) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be JSON-serializable") from exc


def _positive_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in shape)
    if not values or any(value <= 0 or value > _U32_MAX for value in values):
        raise ValueError(f"HQ archive shapes must contain positive uint32 dimensions, got {values}")
    return values


@dataclass(frozen=True)
class HQFormatDescriptor:
    """Physical layout contract for a family member stored in an HQ archive.

    ``packing`` is deliberately an explicit identifier rather than an
    inference from a name such as ``HQ3``.  That permits a future HQ3 revision
    to coexist with an earlier one and gives CUDA/Vulkan/ROCm kernels one
    stable format contract to dispatch on.
    """

    name: str
    version: int
    layout: str
    block_size: int
    block_bytes: int
    bits_per_weight: float
    packing: str
    parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("HQ format name must be a non-empty string")
        if int(self.version) <= 0:
            raise ValueError("HQ format version must be positive")
        if not isinstance(self.layout, str) or not self.layout:
            raise ValueError("HQ format layout must be a non-empty string")
        if int(self.block_size) <= 0 or int(self.block_bytes) <= 0:
            raise ValueError("HQ format block_size and block_bytes must be positive")
        if float(self.bits_per_weight) <= 0:
            raise ValueError("HQ format bits_per_weight must be positive")
        if not isinstance(self.packing, str) or not self.packing:
            raise ValueError("HQ format packing must be a non-empty string")
        params = dict(self.parameters or {})
        _json_bytes(params, what="HQ format parameters")
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "block_bytes", int(self.block_bytes))
        object.__setattr__(self, "bits_per_weight", float(self.bits_per_weight))
        object.__setattr__(self, "parameters", params)

    def as_dict(self, *, format_id: int | None = None) -> dict[str, Any]:
        result = {
            "name": self.name,
            "version": self.version,
            "layout": self.layout,
            "block_size": self.block_size,
            "block_bytes": self.block_bytes,
            "bits_per_weight": self.bits_per_weight,
            "packing": self.packing,
            "parameters": dict(self.parameters or {}),
        }
        if format_id is not None:
            result["id"] = int(format_id)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HQFormatDescriptor":
        try:
            return cls(
                name=str(value["name"]),
                version=int(value["version"]),
                layout=str(value["layout"]),
                block_size=int(value["block_size"]),
                block_bytes=int(value["block_bytes"]),
                bits_per_weight=float(value["bits_per_weight"]),
                packing=str(value["packing"]),
                parameters=value.get("parameters", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid HQ format descriptor") from exc

    @property
    def key(self) -> bytes:
        return _json_bytes(self.as_dict(), what="HQ format descriptor")


HQ2_FORMAT = HQFormatDescriptor(
    name=FORMAT_NAME,
    version=FORMAT_VERSION,
    layout="linear_out_in_row_major_blocks256",
    block_size=BLOCK_SIZE,
    block_bytes=BLOCK_BYTES,
    bits_per_weight=BITS_PER_WEIGHT,
    packing="centroids-f16-le+selectors-u2-lsb0",
)

HQ3_FORMAT = HQFormatDescriptor(
    name=HQ3_FORMAT_NAME,
    version=HQ3_FORMAT_VERSION,
    layout="linear_out_in_row_major_blocks256",
    block_size=HQ3_BLOCK_SIZE,
    block_bytes=HQ3_BLOCK_BYTES,
    bits_per_weight=HQ3_BITS_PER_WEIGHT,
    packing="centroids-f16-le+selectors-u3-lsb0-packed8x3",
    parameters={"centroids": 8, "selector_bits": 3, "selector_group": "8x3bytes"},
)

HQ8_G128_FORMAT = HQFormatDescriptor(
    name=HQ8_G128_FORMAT_NAME,
    version=HQ8_G128_FORMAT_VERSION,
    layout="linear_out_in_row_major_blocks128",
    block_size=HQ8_G128_BLOCK_SIZE,
    block_bytes=HQ8_G128_BLOCK_BYTES,
    bits_per_weight=HQ8_G128_BITS_PER_WEIGHT,
    packing="fp16-scale-le+i8-values-row-major",
    parameters={"quantization": "symmetric-maxabs-int8", "group_size": HQ8_G128_BLOCK_SIZE},
)


def _is_hq2_descriptor(format: HQFormatDescriptor) -> bool:
    """Only this exact payload contract can become an :class:`HQ2Tensor`."""
    return (
        format.name == FORMAT_NAME
        and format.version == FORMAT_VERSION
        and format.layout == HQ2_FORMAT.layout
        and format.block_size == BLOCK_SIZE
        and format.block_bytes == BLOCK_BYTES
        and format.packing == HQ2_FORMAT.packing
    )


def _is_hq3_descriptor(format: HQFormatDescriptor) -> bool:
    """Only this exact payload contract can become an :class:`HQ3Tensor`."""
    return (
        format.name == HQ3_FORMAT_NAME
        and format.version == HQ3_FORMAT_VERSION
        and format.layout == HQ3_FORMAT.layout
        and format.block_size == HQ3_BLOCK_SIZE
        and format.block_bytes == HQ3_BLOCK_BYTES
        and format.packing == HQ3_FORMAT.packing
    )


def _is_hq8_g128_descriptor(format: HQFormatDescriptor) -> bool:
    """Only this reference HQ8_G128 contract can become an :class:`HQ8Tensor`."""
    return (
        format.name == HQ8_G128_FORMAT_NAME
        and format.version == HQ8_G128_FORMAT_VERSION
        and format.layout == HQ8_G128_FORMAT.layout
        and format.block_size == HQ8_G128_BLOCK_SIZE
        and format.block_bytes == HQ8_G128_BLOCK_BYTES
        and format.packing == HQ8_G128_FORMAT.packing
        and dict(format.parameters or {}) == dict(HQ8_G128_FORMAT.parameters or {})
    )


@dataclass(frozen=True)
class HQTensorDescriptor:
    """Small, eagerly-read descriptor for one mmap-able archive payload."""

    name: str
    shape: tuple[int, ...]
    format: HQFormatDescriptor
    offset: int
    nbytes: int
    iterations: int = 0
    importance_weighted: bool = False


@dataclass(frozen=True)
class HQModel:
    """Lazy collection of named, potentially mixed-HQ-format tensors.

    ``tensor(name)`` materializes canonical HQ2, HQ3, and reference HQ8 tensors. ``payload``
    remains useful for an unknown future HQ-family codec without decoding it.
    """

    path: Path
    metadata: Mapping[str, Any]
    _entries: Mapping[str, Mapping[str, Any]]
    _formats: Mapping[int, HQFormatDescriptor]
    container_version: int = MODEL_VERSION

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def formats(self) -> tuple[HQFormatDescriptor, ...]:
        return tuple(self._formats[key] for key in sorted(self._formats))

    def descriptor(self, name: str) -> HQTensorDescriptor:
        try:
            entry = self._entries[name]
            format = self._formats[int(entry["format_id"])]
        except KeyError as exc:
            raise KeyError(f"HQ model {self.path} has no tensor {name!r}") from exc
        return HQTensorDescriptor(
            name=name,
            shape=tuple(int(value) for value in entry["shape"]),
            format=format,
            offset=int(entry["offset"]),
            nbytes=int(entry["nbytes"]),
            iterations=int(entry.get("iterations", 0)),
            importance_weighted=bool(entry.get("importance_weighted", False)),
        )

    def payload(self, name: str) -> np.memmap:
        """Memory-map raw packed bytes for a future HQ-family backend."""
        entry = self.descriptor(name)
        return np.memmap(self.path, mode="r", offset=entry.offset, dtype=np.uint8, shape=(entry.nbytes,), order="C")

    def tensor(self, name: str) -> HQ2Tensor | HQ3Tensor | HQ8Tensor:
        """Open one supported packed tensor without reading any other layer."""
        entry = self.descriptor(name)
        if _is_hq2_descriptor(entry.format):
            shape = validate_shape(entry.shape)
            expected = packed_nbytes(shape)
            tensor_type = HQ2Tensor
            backend = "hq2-file"
            block_bytes = BLOCK_BYTES
        elif _is_hq3_descriptor(entry.format):
            shape = validate_hq3_shape(entry.shape)
            expected = hq3_packed_nbytes(shape)
            tensor_type = HQ3Tensor
            backend = "hq3-file"
            block_bytes = HQ3_BLOCK_BYTES
        elif _is_hq8_g128_descriptor(entry.format):
            shape = validate_hq8_g128_shape(entry.shape)
            expected = hq8_g128_packed_nbytes(shape)
            tensor_type = HQ8Tensor
            backend = "hq8-g128-file"
            block_bytes = HQ8_G128_BLOCK_BYTES
        else:
            raise NotImplementedError(
                f"{name!r} uses {entry.format.name} v{entry.format.version} "
                f"({entry.format.packing}); this installation has no matching tensor codec"
            )
        if entry.nbytes != expected:
            raise ValueError(f"Invalid {entry.format.name} payload size for {name!r}: {entry.nbytes}, expected {expected}")
        packed = np.memmap(
            self.path,
            mode="r",
            offset=entry.offset,
            dtype=np.uint8,
            shape=(expected // block_bytes, block_bytes),
            order="C",
        )
        return tensor_type(
            packed=packed,
            shape=shape,
            backend=backend,
            iterations=entry.iterations,
            importance_weighted=entry.importance_weighted,
        )

    def hq3_tensor(self, name: str) -> HQ3Tensor:
        """Open one canonical HQ3 tensor, rejecting other archive entries."""
        value = self.tensor(name)
        if not isinstance(value, HQ3Tensor):
            raise ValueError(f"{name!r} is not an HQ3 tensor")
        return value

    def hq8_g128_tensor(self, name: str) -> HQ8Tensor:
        """Open one canonical row-major HQ8_G128 tensor."""
        value = self.tensor(name)
        if not isinstance(value, HQ8Tensor):
            raise ValueError(f"{name!r} is not an HQ8_G128 tensor")
        return value

    def raw_tensor(self, name: str) -> HQRawTensor:
        """Open one lossless RAW tensor for standalone-package loading."""
        entry = self.descriptor(name)
        if not is_raw_descriptor(entry.format):
            raise NotImplementedError(f"{name!r} is not a supported HQ RAW tensor")
        dtype_name = str(entry.format.parameters["dtype"])
        return HQRawTensor(packed=self.payload(name), shape=entry.shape, dtype_name=dtype_name)

    def items(self) -> Iterator[tuple[str, HQ2Tensor | HQ3Tensor | HQ8Tensor]]:
        """Iterate supported HQ tensors; unknown codecs fail at the first use."""
        for name in self.tensor_names:
            yield name, self.tensor(name)


# Keep the original public type name source-compatible.  New users may prefer
# HQModel because it accurately describes a future mixed HQ1/HQ2/HQ3 archive.
HQ2Model = HQModel


class HQModelWriter:
    """Streaming writer for a compact, direct-loadable HQ-family archive.

    Payloads are appended in inference order.  A binary table plus a tiny
    JSON format registry is committed only on ``close()``, eliminating the
    v1 front-reserved metadata page while retaining single-layer mmap reads.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        header_reserve_bytes: int | None = None,
    ) -> None:
        self.path = Path(path)
        _require_archive_path(self.path)
        # Kept as a harmless keyword for v1 callers.  v2 uses a footer index,
        # so reserving a large front area would only waste disk space.
        if header_reserve_bytes is not None and header_reserve_bytes < _HEADER.size:
            raise ValueError(f"header_reserve_bytes must be at least {_HEADER.size} when supplied")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = dict(metadata or {})
        _json_bytes(self._metadata, what="HQ archive metadata")
        self._entries: dict[str, dict[str, Any]] = {}
        self._formats: dict[int, HQFormatDescriptor] = {}
        self._format_ids: dict[bytes, int] = {}
        self._offset = PAYLOAD_ALIGNMENT
        self._file = self.path.open("wb+")
        self._file.write(b"\0" * PAYLOAD_ALIGNMENT)
        self._closed = False

    def _format_id(self, format: HQFormatDescriptor) -> int:
        key = format.key
        existing = self._format_ids.get(key)
        if existing is not None:
            return existing
        # Canonical HQ2 always gets id 1 for easy inspection; other codecs
        # receive stable insertion-order ids within this one archive.
        if _is_hq2_descriptor(format) and _FORMAT_ID_HQ2 not in self._formats:
            format_id = _FORMAT_ID_HQ2
        else:
            format_id = max(self._formats, default=0) + 1
            if format_id == _FORMAT_ID_HQ2:
                format_id += 1
        if format_id > 0xFFFF:
            raise ValueError("HQ archive supports at most 65535 distinct format descriptors")
        self._formats[format_id] = format
        self._format_ids[key] = format_id
        return format_id

    def add(self, name: str, tensor: HQ2Tensor | HQ3Tensor | HQ8Tensor) -> None:
        """Append canonical bytes without decoding or re-quantizing."""
        if isinstance(tensor, HQ2Tensor):
            format = HQ2_FORMAT
        elif isinstance(tensor, HQ3Tensor):
            format = HQ3_FORMAT
        elif isinstance(tensor, HQ8Tensor):
            format = HQ8_G128_FORMAT
        else:
            raise TypeError("HQModelWriter.add requires an HQ2Tensor, HQ3Tensor, or HQ8Tensor")
        self.add_raw(
            name,
            tensor.numpy(),
            shape=tensor.shape,
            format=format,
            iterations=tensor.iterations,
            importance_weighted=tensor.importance_weighted,
        )

    def add_raw(
        self,
        name: str,
        packed: np.ndarray,
        *,
        shape: tuple[int, ...] | list[int],
        format: HQFormatDescriptor,
        iterations: int = 0,
        importance_weighted: bool = False,
    ) -> None:
        """Append an HQ-family payload with its explicit layout contract.

        This intentionally does not decode or validate unimplemented codecs;
        it records their bytes faithfully. Canonical HQ2 and HQ3 receive
        strict size/shape validation and can immediately use ``HQModel.tensor``.
        """
        if self._closed:
            raise RuntimeError("Cannot add tensors after HQModelWriter.close()")
        if not isinstance(name, str) or not name:
            raise ValueError("HQ tensor names must be non-empty strings")
        if name in self._entries:
            raise ValueError(f"HQ model already contains tensor {name!r}")
        if not isinstance(format, HQFormatDescriptor):
            raise TypeError("format must be an HQFormatDescriptor")
        array = np.ascontiguousarray(packed, dtype=np.uint8)
        if array.size <= 0:
            raise ValueError("HQ packed payload must not be empty")
        normalized_shape = _positive_shape(shape)
        if _is_hq2_descriptor(format):
            normalized_shape = validate_shape(normalized_shape)
            expected = packed_nbytes(normalized_shape)
            if array.nbytes != expected:
                raise ValueError(f"HQ2 tensor {name!r} has {array.nbytes} bytes; expected {expected}")
        elif _is_hq3_descriptor(format):
            normalized_shape = validate_hq3_shape(normalized_shape)
            expected = hq3_packed_nbytes(normalized_shape)
            if array.nbytes != expected:
                raise ValueError(f"HQ3 tensor {name!r} has {array.nbytes} bytes; expected {expected}")
        elif _is_hq8_g128_descriptor(format):
            normalized_shape = validate_hq8_g128_shape(normalized_shape)
            expected = hq8_g128_packed_nbytes(normalized_shape)
            if array.nbytes != expected:
                raise ValueError(f"HQ8_G128 tensor {name!r} has {array.nbytes} bytes; expected {expected}")
        if int(iterations) < 0 or int(iterations) > _U32_MAX:
            raise ValueError("iterations must fit an unsigned 32-bit value")

        offset = _aligned(self._offset)
        self._file.seek(offset)
        array.tofile(self._file)
        self._offset = offset + array.nbytes
        self._entries[name] = {
            "shape": list(normalized_shape),
            "format_id": self._format_id(format),
            "offset": offset,
            "nbytes": int(array.nbytes),
            "iterations": int(iterations),
            "importance_weighted": bool(importance_weighted),
        }

    def _encoded_index(self) -> bytes:
        metadata = _json_bytes(self._metadata, what="HQ archive metadata")
        formats = _json_bytes(
            [self._formats[format_id].as_dict(format_id=format_id) for format_id in sorted(self._formats)],
            what="HQ archive format table",
        )
        names = bytearray()
        shapes = bytearray()
        entries = bytearray()
        for name, entry in self._entries.items():
            encoded_name = name.encode("utf-8")
            if not encoded_name or len(encoded_name) > _U32_MAX:
                raise ValueError(f"Invalid HQ tensor name size for {name!r}")
            name_offset = len(names)
            names.extend(encoded_name)
            shape_offset = len(shapes)
            for dimension in entry["shape"]:
                shapes.extend(struct.pack("<I", int(dimension)))
            flags = _FLAG_IMPORTANCE_WEIGHTED if entry["importance_weighted"] else 0
            entries.extend(
                _ENTRY.pack(
                    name_offset,
                    len(encoded_name),
                    int(entry["format_id"]),
                    len(entry["shape"]),
                    flags,
                    int(entry["iterations"]),
                    int(entry["offset"]),
                    int(entry["nbytes"]),
                    shape_offset,
                    0,
                )
            )
        if len(self._entries) > _U32_MAX:
            raise ValueError("HQ archive has too many tensors")
        return b"".join(
            (
                _INDEX_HEADER.pack(
                    _INDEX_MAGIC,
                    _INDEX_VERSION,
                    len(metadata),
                    len(formats),
                    len(entries),
                    len(names),
                    len(shapes),
                ),
                metadata,
                formats,
                entries,
                names,
                shapes,
            )
        )

    def close(self) -> HQModel:
        """Commit the footer index and return a lazy reader for the archive."""
        if self._closed:
            return load_model(self.path)
        if not self._entries:
            self._file.close()
            self._closed = True
            raise ValueError("HQ archive must contain at least one named tensor")
        index = self._encoded_index()
        index_offset = _aligned(self._offset)
        self._file.seek(index_offset)
        self._file.write(index)
        self._file.seek(0)
        self._file.write(_HEADER.pack(MODEL_MAGIC, MODEL_VERSION, _HEADER.size, index_offset, len(index), len(self._entries)))
        self._file.flush()
        self._file.close()
        self._closed = True
        return load_model(self.path)

    def __enter__(self) -> "HQModelWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        elif not self._closed:
            self._file.close()
            self._closed = True


# Original spelling retained for existing callers.
HQ2ModelWriter = HQModelWriter


def save_model(
    path: str | Path,
    tensors: Mapping[str, HQ2Tensor | HQ3Tensor],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> HQModel:
    """Persist named supported HQ-family weights to a compact archive."""
    with HQModelWriter(path, metadata=metadata) as writer:
        for name, tensor in tensors.items():
            writer.add(name, tensor)
    return load_model(path)


def repack_model(
    source: str | Path,
    destination: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> HQModel:
    """Copy an HQ archive into compact v2 without decoding or requantizing.

    This is also the migration path for the original HQ2-only v1 archives.
    Each payload is streamed directly from its memory map to the destination;
    the peak working set is one packed layer rather than the model size.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("repack_model destination must differ from source")
    source_model = load_model(source_path)
    with HQModelWriter(destination_path, metadata=source_model.metadata if metadata is None else metadata) as writer:
        for name in source_model.tensor_names:
            entry = source_model.descriptor(name)
            writer.add_raw(
                name,
                source_model.payload(name),
                shape=entry.shape,
                format=entry.format,
                iterations=entry.iterations,
                importance_weighted=entry.importance_weighted,
            )
    return load_model(destination_path)


def _parse_v1(path: Path, raw_header: bytes, file_size: int) -> HQModel:
    magic, version, manifest_size, payload_start = _LEGACY_HEADER.unpack(raw_header[: _LEGACY_HEADER.size])
    if magic != LEGACY_MODEL_MAGIC or version != LEGACY_MODEL_VERSION:
        raise ValueError(f"Unsupported legacy HQ2 model archive {path}")
    if payload_start < _LEGACY_HEADER.size or payload_start % PAYLOAD_ALIGNMENT:
        raise ValueError(f"Invalid legacy HQ2 payload start {payload_start}")
    if manifest_size > payload_start - _LEGACY_HEADER.size:
        raise ValueError("Legacy HQ2 archive manifest exceeds reserved header region")
    with path.open("rb") as file:
        file.seek(_LEGACY_HEADER.size)
        try:
            manifest = json.loads(file.read(manifest_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid legacy HQ2 archive manifest in {path}") from exc
    if manifest.get("format") != "HQ2MODEL" or manifest.get("version") != LEGACY_MODEL_VERSION:
        raise ValueError(f"Invalid legacy HQ2 archive identity in {path}")
    if manifest.get("layout") != HQ2_FORMAT.layout:
        raise ValueError(f"Unsupported legacy HQ2 layout {manifest.get('layout')!r}")
    if manifest.get("block_size") != BLOCK_SIZE or manifest.get("block_bytes") != BLOCK_BYTES:
        raise ValueError("Legacy HQ2 archive block layout does not match this library")
    raw_entries = manifest.get("tensors")
    if not isinstance(raw_entries, dict) or not raw_entries:
        raise ValueError("Legacy HQ2 archive must contain at least one named tensor")
    previous_end = payload_start
    entries: dict[str, dict[str, Any]] = {}
    for name, raw in raw_entries.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError("Invalid legacy HQ2 tensor descriptor")
        try:
            shape = validate_shape(tuple(int(value) for value in raw["shape"]))
            offset = int(raw["offset"])
            nbytes = int(raw["nbytes"])
            iterations = int(raw["iterations"])
            importance_weighted = bool(raw["importance_weighted"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid legacy HQ2 descriptor for {name!r}") from exc
        if nbytes != packed_nbytes(shape) or offset < payload_start or offset % PAYLOAD_ALIGNMENT or offset < previous_end:
            raise ValueError(f"Invalid legacy HQ2 payload bounds for {name!r}")
        if offset + nbytes > file_size:
            raise ValueError(f"Legacy HQ2 tensor {name!r} extends beyond end of {path}")
        entries[name] = {
            "shape": list(shape),
            "format_id": _FORMAT_ID_HQ2,
            "offset": offset,
            "nbytes": nbytes,
            "iterations": iterations,
            "importance_weighted": importance_weighted,
        }
        previous_end = offset + nbytes
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Legacy HQ2 archive metadata must be a JSON object")
    return HQModel(
        path=path,
        metadata=metadata,
        _entries=entries,
        _formats={_FORMAT_ID_HQ2: HQ2_FORMAT},
        container_version=LEGACY_MODEL_VERSION,
    )


def _parse_v2(path: Path, raw_header: bytes, file_size: int) -> HQModel:
    magic, version, header_size, index_offset, index_size, entry_count = _HEADER.unpack(raw_header[: _HEADER.size])
    if magic != MODEL_MAGIC or version != MODEL_VERSION:
        raise ValueError(f"Unsupported HQ model archive version {version}")
    if header_size != _HEADER.size:
        raise ValueError(f"Unsupported HQ archive header size {header_size}")
    if index_offset < PAYLOAD_ALIGNMENT or index_offset % PAYLOAD_ALIGNMENT:
        raise ValueError(f"Invalid HQ archive index offset {index_offset}")
    if index_size < _INDEX_HEADER.size or index_offset + index_size > file_size:
        raise ValueError("HQ archive index extends beyond end of file")
    with path.open("rb") as file:
        file.seek(index_offset)
        index = file.read(index_size)
    if len(index) != index_size:
        raise ValueError("Could not read complete HQ archive index")
    try:
        index_magic, index_version, metadata_size, formats_size, entries_size, names_size, shapes_size = _INDEX_HEADER.unpack(
            index[: _INDEX_HEADER.size]
        )
    except struct.error as exc:
        raise ValueError("Invalid HQ archive index header") from exc
    if index_magic != _INDEX_MAGIC or index_version != _INDEX_VERSION:
        raise ValueError("Unsupported HQ archive index version")
    expected_size = _INDEX_HEADER.size + metadata_size + formats_size + entries_size + names_size + shapes_size
    if expected_size != index_size or entries_size != entry_count * _ENTRY.size or shapes_size % 4:
        raise ValueError("Invalid HQ archive index section sizes")
    cursor = _INDEX_HEADER.size
    metadata_bytes = index[cursor : cursor + metadata_size]
    cursor += metadata_size
    format_bytes = index[cursor : cursor + formats_size]
    cursor += formats_size
    entry_bytes = index[cursor : cursor + entries_size]
    cursor += entries_size
    names = index[cursor : cursor + names_size]
    cursor += names_size
    shapes = index[cursor : cursor + shapes_size]
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        raw_formats = json.loads(format_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid HQ archive JSON index sections") from exc
    if not isinstance(metadata, dict) or not isinstance(raw_formats, list) or not raw_formats:
        raise ValueError("Invalid HQ archive metadata or format table")
    formats: dict[int, HQFormatDescriptor] = {}
    for raw in raw_formats:
        if not isinstance(raw, dict):
            raise ValueError("Invalid HQ archive format table entry")
        try:
            format_id = int(raw["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid HQ archive format id") from exc
        if format_id <= 0 or format_id > 0xFFFF or format_id in formats:
            raise ValueError("Duplicate or invalid HQ archive format id")
        formats[format_id] = HQFormatDescriptor.from_dict(raw)

    entries: dict[str, dict[str, Any]] = {}
    previous_end = PAYLOAD_ALIGNMENT
    for index_in_file in range(entry_count):
        start = index_in_file * _ENTRY.size
        (
            name_offset,
            name_size,
            format_id,
            rank,
            flags,
            iterations,
            offset,
            nbytes,
            shape_offset,
            reserved,
        ) = _ENTRY.unpack(entry_bytes[start : start + _ENTRY.size])
        if reserved != 0 or format_id not in formats or rank == 0 or flags & ~_FLAG_IMPORTANCE_WEIGHTED:
            raise ValueError("Invalid HQ archive tensor entry")
        if name_offset + name_size > len(names) or shape_offset + rank * 4 > len(shapes):
            raise ValueError("HQ archive tensor entry points outside its index section")
        try:
            name = names[name_offset : name_offset + name_size].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("HQ archive tensor name is not UTF-8") from exc
        if not name or name in entries:
            raise ValueError("Duplicate or empty HQ archive tensor name")
        shape = tuple(struct.unpack_from("<I", shapes, shape_offset + dimension * 4)[0] for dimension in range(rank))
        normalized_shape = _positive_shape(shape)
        if offset < PAYLOAD_ALIGNMENT or offset % PAYLOAD_ALIGNMENT or offset < previous_end:
            raise ValueError(f"Invalid payload bounds for HQ tensor {name!r}")
        if nbytes <= 0 or offset + nbytes > index_offset:
            raise ValueError(f"HQ tensor {name!r} extends into or beyond the footer index")
        format = formats[format_id]
        if _is_hq2_descriptor(format):
            shape_for_hq2 = validate_shape(normalized_shape)
            if nbytes != packed_nbytes(shape_for_hq2):
                raise ValueError(f"Invalid HQ2 payload size for {name!r}")
        elif _is_hq3_descriptor(format):
            shape_for_hq3 = validate_hq3_shape(normalized_shape)
            if nbytes != hq3_packed_nbytes(shape_for_hq3):
                raise ValueError(f"Invalid HQ3 payload size for {name!r}")
        entries[name] = {
            "shape": list(normalized_shape),
            "format_id": format_id,
            "offset": offset,
            "nbytes": nbytes,
            "iterations": iterations,
            "importance_weighted": bool(flags & _FLAG_IMPORTANCE_WEIGHTED),
        }
        previous_end = offset + nbytes
    return HQModel(
        path=path,
        metadata=metadata,
        _entries=entries,
        _formats=formats,
        container_version=MODEL_VERSION,
    )


def load_model(path: str | Path) -> HQModel:
    """Read an HQ archive index without loading any packed tensor payloads.

    Both compact format-neutral v2 files and the original v1 HQ2-only files
    are supported.  New conversions always write v2.
    """
    path = Path(path)
    _require_archive_path(path)
    file_size = path.stat().st_size
    if file_size < _LEGACY_HEADER.size:
        raise ValueError(f"{path} is too small to be an HQ model archive")
    with path.open("rb") as file:
        raw_header = file.read(_HEADER.size)
    magic = raw_header[:8]
    if magic == LEGACY_MODEL_MAGIC:
        return _parse_v1(path, raw_header, file_size)
    if magic == MODEL_MAGIC:
        if len(raw_header) < _HEADER.size:
            raise ValueError(f"{path} is too small for an HQ v2 header")
        return _parse_v2(path, raw_header, file_size)
    raise ValueError(f"{path} is not an HQ model archive")


__all__ = [
    "HQ2_FORMAT",
    "HQ3_FORMAT",
    "HQ8_G128_FORMAT",
    "HQ2Model",
    "HQ2ModelWriter",
    "HQFormatDescriptor",
    "HQModel",
    "HQModelWriter",
    "HQTensorDescriptor",
    "MODEL_VERSION",
    "PAYLOAD_ALIGNMENT",
    "load_model",
    "repack_model",
    "save_model",
]
