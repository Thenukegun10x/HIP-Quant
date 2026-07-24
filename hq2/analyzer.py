"""Inspect, account for, and optionally deep-scan HQ-family archives.

The analyzer is deliberately read-only.  It uses the same strict archive
loader as inference, then produces a JSON-serializable summary suitable for a
command-line report, CI validation, or a model catalogue.  Unknown future HQ
formats are reported from their descriptors without pretending to decode them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .archive import HQ2_FORMAT, HQ3_FORMAT, HQFormatDescriptor, load_model


def _value_count(shape: tuple[int, ...]) -> int:
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return total


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


def _format_label(format: HQFormatDescriptor) -> str:
    return f"{format.name} v{format.version} ({format.packing})"


_HQ2_BLOCK_BYTES = 72
_HQ2_SCAN_CHUNK_BYTES = _HQ2_BLOCK_BYTES * 16_384
_HQ3_BLOCK_BYTES = 112
_HQ3_SCAN_CHUNK_BYTES = _HQ3_BLOCK_BYTES * 16_384
_DEFAULT_DEEP_SAMPLE_BYTES = 2 * 1024 * 1024


class _HQ2BlockAccumulator:
    """Incremental HQ2 physical scan that never retains a model-layer map."""

    def __init__(self, *, checksum: bool) -> None:
        self.block_count = 0
        self.finite_centroid_count = 0
        self.nonfinite_centroid_count = 0
        self.zero_centroid_block_count = 0
        self.monotonic_centroid_block_count = 0
        self.selector_histogram = np.zeros(4, dtype=np.int64)
        self.centroid_min: float | None = None
        self.centroid_max: float | None = None
        self.digest = hashlib.sha256() if checksum else None

    def update(self, payload: bytes | np.ndarray) -> None:
        byte_values = (
            np.asarray(payload, dtype=np.uint8)
            if isinstance(payload, np.ndarray)
            else np.frombuffer(payload, dtype=np.uint8)
        )
        if byte_values.size % _HQ2_BLOCK_BYTES:
            raise ValueError("HQ2 analyzer received a non-block-aligned payload chunk")
        chunk = byte_values.reshape(-1, _HQ2_BLOCK_BYTES)
        centroids = chunk[:, :8].copy().view("<f2").reshape(-1, 4)
        finite = np.isfinite(centroids)
        self.block_count += int(chunk.shape[0])
        self.finite_centroid_count += int(finite.sum())
        self.nonfinite_centroid_count += int(centroids.size - finite.sum())
        self.zero_centroid_block_count += int(np.all(centroids == 0, axis=1).sum())
        self.monotonic_centroid_block_count += int(np.all(np.diff(centroids, axis=1) >= 0, axis=1).sum())
        finite_values = centroids[finite]
        if finite_values.size:
            current_min = float(finite_values.min())
            current_max = float(finite_values.max())
            self.centroid_min = current_min if self.centroid_min is None else min(self.centroid_min, current_min)
            self.centroid_max = current_max if self.centroid_max is None else max(self.centroid_max, current_max)
        selector_bytes = chunk[:, 8:]
        for shift in (0, 2, 4, 6):
            self.selector_histogram += np.bincount(
                ((selector_bytes >> shift) & 3).reshape(-1), minlength=4
            ).astype(np.int64, copy=False)
        if self.digest is not None:
            self.digest.update(memoryview(byte_values))

    def finish(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "block_count": self.block_count,
            "centroid_count": self.block_count * 4,
            "finite_centroid_count": self.finite_centroid_count,
            "nonfinite_centroid_count": self.nonfinite_centroid_count,
            "zero_centroid_block_count": self.zero_centroid_block_count,
            "monotonic_centroid_block_count": self.monotonic_centroid_block_count,
            "selector_histogram": [int(value) for value in self.selector_histogram],
            "centroid_min": self.centroid_min,
            "centroid_max": self.centroid_max,
        }
        if self.digest is not None:
            result["sha256"] = self.digest.hexdigest()
        return result


def _hq2_block_stats(raw: np.ndarray, *, checksum: bool) -> dict[str, Any]:
    """Scan an already-open payload; retained for small in-process diagnostics."""
    accumulator = _HQ2BlockAccumulator(checksum=checksum)
    for first_byte in range(0, int(raw.size), _HQ2_SCAN_CHUNK_BYTES):
        accumulator.update(raw[first_byte : first_byte + _HQ2_SCAN_CHUNK_BYTES])
    return accumulator.finish()


def _hq2_file_stats(path: Path, offset: int, nbytes: int, *, checksum: bool) -> dict[str, Any]:
    """Deep-scan one payload with buffered reads, avoiding cumulative mmaps."""
    accumulator = _HQ2BlockAccumulator(checksum=checksum)
    with path.open("rb", buffering=_HQ2_SCAN_CHUNK_BYTES) as file:
        file.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = file.read(min(_HQ2_SCAN_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("HQ analyzer could not read a complete HQ2 payload")
            accumulator.update(chunk)
            remaining -= len(chunk)
    return accumulator.finish()


class _HQ3BlockAccumulator:
    """Incremental HQ3 physical scan that never retains a model-layer map."""

    def __init__(self, *, checksum: bool) -> None:
        self.block_count = 0
        self.finite_centroid_count = 0
        self.nonfinite_centroid_count = 0
        self.zero_centroid_block_count = 0
        self.monotonic_centroid_block_count = 0
        self.selector_histogram = np.zeros(8, dtype=np.int64)
        self.centroid_min: float | None = None
        self.centroid_max: float | None = None
        self.digest = hashlib.sha256() if checksum else None

    def update(self, payload: bytes | np.ndarray) -> None:
        byte_values = (
            np.asarray(payload, dtype=np.uint8)
            if isinstance(payload, np.ndarray)
            else np.frombuffer(payload, dtype=np.uint8)
        )
        if byte_values.size % _HQ3_BLOCK_BYTES:
            raise ValueError("HQ3 analyzer received a non-block-aligned payload chunk")
        chunk = byte_values.reshape(-1, _HQ3_BLOCK_BYTES)
        centroids = chunk[:, :16].copy().view("<f2").reshape(-1, 8)
        finite = np.isfinite(centroids)
        self.block_count += int(chunk.shape[0])
        self.finite_centroid_count += int(finite.sum())
        self.nonfinite_centroid_count += int(centroids.size - finite.sum())
        self.zero_centroid_block_count += int(np.all(centroids == 0, axis=1).sum())
        self.monotonic_centroid_block_count += int(np.all(np.diff(centroids, axis=1) >= 0, axis=1).sum())
        finite_values = centroids[finite]
        if finite_values.size:
            current_min = float(finite_values.min())
            current_max = float(finite_values.max())
            self.centroid_min = current_min if self.centroid_min is None else min(self.centroid_min, current_min)
            self.centroid_max = current_max if self.centroid_max is None else max(self.centroid_max, current_max)
        groups = chunk[:, 16:].reshape(-1, 32, 3).astype(np.uint32)
        words = groups[:, :, 0] | (groups[:, :, 1] << 8) | (groups[:, :, 2] << 16)
        shifts = (3 * np.arange(8, dtype=np.uint32)).reshape(1, 1, 8)
        codes = ((words[:, :, None] >> shifts) & 7).reshape(-1)
        self.selector_histogram += np.bincount(codes, minlength=8).astype(np.int64, copy=False)
        if self.digest is not None:
            self.digest.update(memoryview(byte_values))

    def finish(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "block_count": self.block_count,
            "centroid_count": self.block_count * 8,
            "finite_centroid_count": self.finite_centroid_count,
            "nonfinite_centroid_count": self.nonfinite_centroid_count,
            "zero_centroid_block_count": self.zero_centroid_block_count,
            "monotonic_centroid_block_count": self.monotonic_centroid_block_count,
            "selector_histogram": [int(value) for value in self.selector_histogram],
            "centroid_min": self.centroid_min,
            "centroid_max": self.centroid_max,
        }
        if self.digest is not None:
            result["sha256"] = self.digest.hexdigest()
        return result


def _hq3_file_stats(path: Path, offset: int, nbytes: int, *, checksum: bool) -> dict[str, Any]:
    accumulator = _HQ3BlockAccumulator(checksum=checksum)
    with path.open("rb", buffering=_HQ3_SCAN_CHUNK_BYTES) as file:
        file.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = file.read(min(_HQ3_SCAN_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("HQ analyzer could not read a complete HQ3 payload")
            accumulator.update(chunk)
            remaining -= len(chunk)
    return accumulator.finish()


def _file_checksum(path: Path, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=_HQ2_SCAN_CHUNK_BYTES) as file:
        file.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = file.read(min(_HQ2_SCAN_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("HQ analyzer could not read a complete payload")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HQAnalysis:
    """Read-only analysis result returned by :func:`analyze_model`."""

    path: Path
    container_version: int
    metadata: Mapping[str, Any]
    storage: Mapping[str, Any]
    formats: tuple[Mapping[str, Any], ...]
    tensors: tuple[Mapping[str, Any], ...]
    integrity: Mapping[str, Any]

    @property
    def total_values(self) -> int:
        return int(self.storage["logical_value_count"])

    @property
    def payload_bits_per_weight(self) -> float | None:
        value = self.storage.get("payload_bits_per_weight")
        return None if value is None else float(value)

    def as_dict(self, *, include_tensors: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable document for tools or model catalogues."""
        result: dict[str, Any] = {
            "path": str(self.path),
            "container": {"family": "HQ", "version": self.container_version},
            "metadata": dict(self.metadata),
            "storage": dict(self.storage),
            "formats": [dict(item) for item in self.formats],
            "integrity": dict(self.integrity),
        }
        if include_tensors:
            result["tensors"] = [dict(item) for item in self.tensors]
        return result

    def to_json(self, *, include_tensors: bool = True, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(include_tensors=include_tensors), indent=indent, sort_keys=True)


def analyze_model(
    path: str | Path,
    *,
    deep: bool = False,
    checksums: bool = False,
    deep_sample_bytes: int | None = _DEFAULT_DEEP_SAMPLE_BYTES,
) -> HQAnalysis:
    """Analyze an HQ archive without loading a full-precision model.

    Args:
        path: ``.hq``, ``.hq1``, ``.hq2``, or ``.hq3`` archive to inspect.
        deep: Scan a bounded sample of every canonical HQ2/HQ3 payload for
            non-finite centroids, zero blocks, centroid ordering, and selector
            use.  The report labels sampled versus exhaustive results.
        checksums: Include per-tensor SHA-256 values.  This reads payloads but
            does not require a codec-specific deep scan for unknown HQ formats.
        deep_sample_bytes: Maximum bytes to inspect per HQ2/HQ3 tensor when
            ``deep`` is true.  Use ``None`` only for an exhaustive scan of a
            modest archive; the default 2 MiB is deliberate for large models.
    """
    model = load_model(path)
    if deep_sample_bytes is not None and int(deep_sample_bytes) <= 0:
        raise ValueError("deep_sample_bytes must be positive or None")
    file_bytes = model.path.stat().st_size
    entries = [model.descriptor(name) for name in model.tensor_names]
    ordered = sorted(entries, key=lambda entry: entry.offset)
    payload_bytes = sum(entry.nbytes for entry in entries)
    total_values = sum(_value_count(entry.shape) for entry in entries)
    previous_end = 0
    alignment_padding = 0
    for entry in ordered:
        alignment_padding += max(0, entry.offset - previous_end)
        previous_end = entry.offset + entry.nbytes
    trailing_bytes = max(0, file_bytes - previous_end)

    grouped: dict[bytes, dict[str, Any]] = {}
    tensor_rows: list[dict[str, Any]] = []
    deep_hq2_payloads = 0
    finite_centroids = True
    hq2_blocks_scanned = 0
    hq2_centroids_scanned = 0
    hq2_nonfinite_centroids = 0
    hq2_zero_blocks = 0
    hq2_monotonic_blocks = 0
    hq2_selector_histogram = np.zeros(4, dtype=np.int64)
    hq2_bytes_deep_scanned = 0
    hq2_fully_scanned_payloads = 0
    deep_hq3_payloads = 0
    hq3_finite_centroids = True
    hq3_blocks_scanned = 0
    hq3_centroids_scanned = 0
    hq3_nonfinite_centroids = 0
    hq3_zero_blocks = 0
    hq3_monotonic_blocks = 0
    hq3_selector_histogram = np.zeros(8, dtype=np.int64)
    hq3_bytes_deep_scanned = 0
    hq3_fully_scanned_payloads = 0
    unsupported_deep_formats: set[str] = set()
    for entry in entries:
        value_count = _value_count(entry.shape)
        format_key = entry.format.key
        summary = grouped.get(format_key)
        if summary is None:
            summary = {
                "format": entry.format.as_dict(),
                "tensor_count": 0,
                "logical_value_count": 0,
                "payload_bytes": 0,
            }
            grouped[format_key] = summary
        summary["tensor_count"] += 1
        summary["logical_value_count"] += value_count
        summary["payload_bytes"] += entry.nbytes

        row: dict[str, Any] = {
            "name": entry.name,
            "shape": list(entry.shape),
            "logical_value_count": value_count,
            "format": entry.format.as_dict(),
            "offset": entry.offset,
            "payload_bytes": entry.nbytes,
            "payload_bits_per_weight": _ratio(entry.nbytes * 8, value_count),
            "iterations": entry.iterations,
            "importance_weighted": entry.importance_weighted,
        }
        if deep:
            if entry.format == HQ2_FORMAT:
                scan_bytes = entry.nbytes if deep_sample_bytes is None else min(entry.nbytes, int(deep_sample_bytes))
                scan_bytes -= scan_bytes % _HQ2_BLOCK_BYTES
                if scan_bytes == 0:
                    scan_bytes = _HQ2_BLOCK_BYTES
                codec = _hq2_file_stats(model.path, entry.offset, scan_bytes, checksum=False)
                codec["scanned_payload_bytes"] = scan_bytes
                codec["scan_fraction"] = _ratio(scan_bytes, entry.nbytes)
                row["codec_analysis"] = codec
                if checksums:
                    row["sha256"] = _file_checksum(model.path, entry.offset, entry.nbytes)
                deep_hq2_payloads += 1
                hq2_bytes_deep_scanned += scan_bytes
                hq2_fully_scanned_payloads += int(scan_bytes == entry.nbytes)
                finite_centroids = finite_centroids and codec["nonfinite_centroid_count"] == 0
                hq2_blocks_scanned += int(codec["block_count"])
                hq2_centroids_scanned += int(codec["centroid_count"])
                hq2_nonfinite_centroids += int(codec["nonfinite_centroid_count"])
                hq2_zero_blocks += int(codec["zero_centroid_block_count"])
                hq2_monotonic_blocks += int(codec["monotonic_centroid_block_count"])
                hq2_selector_histogram += np.asarray(codec["selector_histogram"], dtype=np.int64)
            elif entry.format == HQ3_FORMAT:
                scan_bytes = entry.nbytes if deep_sample_bytes is None else min(entry.nbytes, int(deep_sample_bytes))
                scan_bytes -= scan_bytes % _HQ3_BLOCK_BYTES
                if scan_bytes == 0:
                    scan_bytes = _HQ3_BLOCK_BYTES
                codec = _hq3_file_stats(model.path, entry.offset, scan_bytes, checksum=False)
                codec["scanned_payload_bytes"] = scan_bytes
                codec["scan_fraction"] = _ratio(scan_bytes, entry.nbytes)
                row["codec_analysis"] = codec
                if checksums:
                    row["sha256"] = _file_checksum(model.path, entry.offset, entry.nbytes)
                deep_hq3_payloads += 1
                hq3_bytes_deep_scanned += scan_bytes
                hq3_fully_scanned_payloads += int(scan_bytes == entry.nbytes)
                hq3_finite_centroids = hq3_finite_centroids and codec["nonfinite_centroid_count"] == 0
                hq3_blocks_scanned += int(codec["block_count"])
                hq3_centroids_scanned += int(codec["centroid_count"])
                hq3_nonfinite_centroids += int(codec["nonfinite_centroid_count"])
                hq3_zero_blocks += int(codec["zero_centroid_block_count"])
                hq3_monotonic_blocks += int(codec["monotonic_centroid_block_count"])
                hq3_selector_histogram += np.asarray(codec["selector_histogram"], dtype=np.int64)
            else:
                unsupported_deep_formats.add(_format_label(entry.format))
                if checksums:
                    row["sha256"] = _file_checksum(model.path, entry.offset, entry.nbytes)
        elif checksums:
            row["sha256"] = _file_checksum(model.path, entry.offset, entry.nbytes)
        tensor_rows.append(row)

    format_rows: list[dict[str, Any]] = []
    for summary in grouped.values():
        values = int(summary["logical_value_count"])
        payload = int(summary["payload_bytes"])
        summary["payload_bits_per_weight"] = _ratio(payload * 8, values)
        summary["declared_bits_per_weight"] = float(summary["format"]["bits_per_weight"])
        format_rows.append(summary)
    format_rows.sort(key=lambda row: (row["format"]["name"], row["format"]["version"], row["format"]["packing"]))

    storage = {
        "file_bytes": file_bytes,
        "payload_bytes": payload_bytes,
        "container_overhead_bytes": file_bytes - payload_bytes,
        "alignment_and_header_bytes": alignment_padding,
        "footer_or_trailing_bytes": trailing_bytes,
        "logical_value_count": total_values,
        "payload_bits_per_weight": _ratio(payload_bytes * 8, total_values),
        "file_bits_per_weight": _ratio(file_bytes * 8, total_values),
    }
    integrity = {
        "structural_validation": "passed",
        "deep_scan": bool(deep),
        "checksums": bool(checksums),
        "hq2_payloads_deep_scanned": deep_hq2_payloads,
        "hq2_payloads_fully_deep_scanned": hq2_fully_scanned_payloads,
        "hq2_deep_scan_mode": (
            "full" if deep_hq2_payloads and hq2_fully_scanned_payloads == deep_hq2_payloads else "sampled"
        ) if deep else "disabled",
        "hq2_payload_bytes_deep_scanned": hq2_bytes_deep_scanned,
        "all_scanned_hq2_centroids_finite": finite_centroids if deep_hq2_payloads else None,
        "hq2_blocks_deep_scanned": hq2_blocks_scanned,
        "hq2_centroids_deep_scanned": hq2_centroids_scanned,
        "hq2_nonfinite_centroid_count": hq2_nonfinite_centroids,
        "hq2_zero_centroid_block_count": hq2_zero_blocks,
        "hq2_monotonic_centroid_block_count": hq2_monotonic_blocks,
        "hq2_selector_histogram": [int(value) for value in hq2_selector_histogram],
        "hq3_payloads_deep_scanned": deep_hq3_payloads,
        "hq3_payloads_fully_deep_scanned": hq3_fully_scanned_payloads,
        "hq3_deep_scan_mode": (
            "full" if deep_hq3_payloads and hq3_fully_scanned_payloads == deep_hq3_payloads else "sampled"
        ) if deep else "disabled",
        "hq3_payload_bytes_deep_scanned": hq3_bytes_deep_scanned,
        "all_scanned_hq3_centroids_finite": hq3_finite_centroids if deep_hq3_payloads else None,
        "hq3_blocks_deep_scanned": hq3_blocks_scanned,
        "hq3_centroids_deep_scanned": hq3_centroids_scanned,
        "hq3_nonfinite_centroid_count": hq3_nonfinite_centroids,
        "hq3_zero_centroid_block_count": hq3_zero_blocks,
        "hq3_monotonic_centroid_block_count": hq3_monotonic_blocks,
        "hq3_selector_histogram": [int(value) for value in hq3_selector_histogram],
        "deep_scan_not_implemented_for": sorted(unsupported_deep_formats),
    }
    return HQAnalysis(
        path=model.path,
        container_version=model.container_version,
        metadata=dict(model.metadata),
        storage=storage,
        formats=tuple(format_rows),
        tensors=tuple(tensor_rows),
        integrity=integrity,
    )


def render_analysis(analysis: HQAnalysis, *, include_tensors: bool = False) -> str:
    """Render a compact, human-readable report similar to a GGUF inspector."""
    storage = analysis.storage
    bpw = storage["payload_bits_per_weight"]
    lines = [
        f"HQ archive: {analysis.path}",
        f"Container: HQ v{analysis.container_version} | tensors: {len(analysis.tensors)} | formats: {len(analysis.formats)}",
        "Storage:",
        f"  file { _human_bytes(int(storage['file_bytes'])) } | payload { _human_bytes(int(storage['payload_bytes'])) } | overhead { _human_bytes(int(storage['container_overhead_bytes'])) }",
        f"  logical values {int(storage['logical_value_count']):,} | payload {bpw:.6g} bpw" if bpw is not None else "  logical values 0",
        f"  alignment/header { _human_bytes(int(storage['alignment_and_header_bytes'])) } | footer/trailing { _human_bytes(int(storage['footer_or_trailing_bytes'])) }",
        "Formats:",
    ]
    for item in analysis.formats:
        format = item["format"]
        lines.append(
            "  "
            f"{format['name']} v{format['version']}: {item['tensor_count']} tensors, "
            f"{int(item['logical_value_count']):,} values, {item['payload_bits_per_weight']:.6g} bpw "
            f"(declared {item['declared_bits_per_weight']:.6g}), {format['packing']}"
        )
    if analysis.metadata:
        lines.append("Metadata:")
        for key, value in sorted(analysis.metadata.items()):
            lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
    integrity = analysis.integrity
    lines.append(
        "Integrity: "
        f"structure {integrity['structural_validation']}; deep scan={integrity['deep_scan']}; checksums={integrity['checksums']}"
    )
    if integrity["deep_scan"] and integrity["hq2_payloads_deep_scanned"]:
        lines.append(
            "  HQ2 "
            f"{integrity['hq2_deep_scan_mode']} scan: {_human_bytes(int(integrity['hq2_payload_bytes_deep_scanned']))}, "
            f"{int(integrity['hq2_payloads_fully_deep_scanned'])}/{int(integrity['hq2_payloads_deep_scanned'])} payloads full; blocks "
            f"{int(integrity['hq2_blocks_deep_scanned']):,}; centroids finite="
            f"{int(integrity['hq2_nonfinite_centroid_count']) == 0}; zero blocks "
            f"{int(integrity['hq2_zero_centroid_block_count']):,}; monotonic blocks "
            f"{int(integrity['hq2_monotonic_centroid_block_count']):,}"
        )
        lines.append(f"  selector histogram: {integrity['hq2_selector_histogram']}")
    if integrity["deep_scan"] and integrity["hq3_payloads_deep_scanned"]:
        lines.append(
            "  HQ3 "
            f"{integrity['hq3_deep_scan_mode']} scan: {_human_bytes(int(integrity['hq3_payload_bytes_deep_scanned']))}, "
            f"{int(integrity['hq3_payloads_fully_deep_scanned'])}/{int(integrity['hq3_payloads_deep_scanned'])} payloads full; blocks "
            f"{int(integrity['hq3_blocks_deep_scanned']):,}; centroids finite="
            f"{int(integrity['hq3_nonfinite_centroid_count']) == 0}; zero blocks "
            f"{int(integrity['hq3_zero_centroid_block_count']):,}; monotonic blocks "
            f"{int(integrity['hq3_monotonic_centroid_block_count']):,}"
        )
        lines.append(f"  selector histogram: {integrity['hq3_selector_histogram']}")
    if integrity["deep_scan_not_implemented_for"]:
        lines.append("  deep codec scan skipped: " + ", ".join(integrity["deep_scan_not_implemented_for"]))
    if include_tensors:
        lines.append("Tensors:")
        for item in analysis.tensors:
            lines.append(
                f"  {item['name']} | {tuple(item['shape'])} | {item['format']['name']} "
                f"| {_human_bytes(int(item['payload_bytes']))} | {item['payload_bits_per_weight']:.6g} bpw "
                f"| offset {item['offset']:,}"
            )
    else:
        largest = sorted(analysis.tensors, key=lambda item: int(item["payload_bytes"]), reverse=True)[:8]
        if largest:
            lines.append("Largest tensors (use --tensors for all):")
            for item in largest:
                lines.append(f"  {item['name']} | {tuple(item['shape'])} | {_human_bytes(int(item['payload_bytes']))}")
    return "\n".join(lines)


__all__ = ["HQAnalysis", "analyze_model", "render_analysis"]
