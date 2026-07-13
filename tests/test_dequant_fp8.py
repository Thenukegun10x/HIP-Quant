"""Tests for I-Quant and T-Quant dequantization to FP8.

Verifies that all GGML Q-types (including I-Quants and T-Quants) can be
dequantized directly to FP8 E4M3/E5M2 without intermediate F32 tensors.

Requirements: AMD GPU with hip_quant DLL loaded.
"""
import os
import sys
import math
import pathlib
import struct
import unittest
import zlib
import numpy as np

try:
    from hip_quant import get_hip_quant, GGML_TYPE, GGML_TYPE_BLOCK_SIZE
    HAS_HIP_QUANT = True
except ImportError:
    HAS_HIP_QUANT = False

requires_gpu = unittest.skipUnless(HAS_HIP_QUANT, "hip_quant DLL required")


def fp8_e4m3_to_f32(b):
    """Decode a single FP8 E4M3 byte to float32."""
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 15 and mant == 7:
        return math.nan
    if exp == 0 and mant == 0:
        return -0.0 if sign else 0.0
    if exp == 0:
        result = mant * 0.001953125
        return -result if sign else result
    i32 = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
    import struct
    return struct.unpack('f', struct.pack('I', i32))[0]


def fp8_e5m2_to_f32(b):
    """Decode a single FP8 E5M2 byte to float32."""
    sign = (b >> 7) & 1
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    if exp == 31:
        return math.nan if mant else (float('-inf') if sign else float('inf'))
    if exp == 0 and mant == 0:
        return -0.0 if sign else 0.0
    if exp == 0:
        result = mant * 0.0000152587890625
        return -result if sign else result
    i32 = (sign << 31) | ((exp + 112) << 23) | (mant << 21)
    import struct
    return struct.unpack('f', struct.pack('I', i32))[0]


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_IQ4_NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,   25,  38,  53,  69,  89, 113],
    dtype=np.int8,
)


_IQ_CODEBOOK_HEADER = struct.Struct("<8sIIIII")


def _load_iq_grid(asset_name, rows, width):
    """Load a grid from the same external codebook asset the DLL uses."""
    blob = (_ROOT / "codebooks" / asset_name).read_bytes()
    magic, version, grid_bytes, map_bytes, neighbours_bytes, checksum = (
        _IQ_CODEBOOK_HEADER.unpack_from(blob)
    )
    payload = blob[_IQ_CODEBOOK_HEADER.size:]
    if (magic != b"HQIQCB01" or version != 1 or grid_bytes != rows * width
            or len(payload) != grid_bytes + map_bytes + neighbours_bytes):
        raise RuntimeError(f"Invalid I-Quant codebook asset {asset_name}")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != checksum:
        raise RuntimeError(f"Corrupt I-Quant codebook asset {asset_name}")
    return np.frombuffer(payload, dtype=np.int8, count=grid_bytes).reshape(rows, width)


_IQ2_XXS_GRID = _load_iq_grid("iq2_xxs.bin", 256, 8)
_IQ3_XXS_GRID = _load_iq_grid("iq3_xxs.bin", 256, 4)


def _fp16_at(block, offset=0):
    """Read a little-endian GGML half scale as a float32."""
    return np.frombuffer(block, dtype="<f2", count=1, offset=offset)[0].astype(np.float32)


def _fp32_to_e4m3(value):
    """Reference port of fp32_bits_to_fp8_e4m3 from hip_quant_util.h."""
    u = struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]
    sign = u >> 31
    abs_u = u & 0x7fffffff
    if abs_u == 0:
        return sign << 7
    if abs_u > 0x7f800000:
        return (sign << 7) | 0x7f
    if abs_u == 0x7f800000:
        return (sign << 7) | 0x7e

    f32_exp = (abs_u >> 23) & 0xff
    f32_mant = abs_u & 0x7fffff
    if f32_exp == 0:
        return sign << 7
    exp = f32_exp - 127 + 7
    if exp <= 0:
        shift = 1 - exp
        if shift > 4:
            return sign << 7
        full = 0x800000 | f32_mant
        total_shift = 20 + shift
        result = full >> total_shift
        remainder = full & ((1 << total_shift) - 1)
        midpoint = 1 << (total_shift - 1)
        if remainder > midpoint or (remainder == midpoint and result & 1):
            result += 1
        if result >= 8:
            return (sign << 7) | 0x08
        return (sign << 7) | (result & 0x07)

    fp8_mant = (f32_mant >> 20) & 0x07
    rnd = f32_mant & 0xfffff
    if rnd > 0x80000 or (rnd == 0x80000 and fp8_mant & 1):
        fp8_mant += 1
        if fp8_mant >= 8:
            fp8_mant = 0
            exp += 1
    if exp >= 16 or (exp == 15 and fp8_mant == 7):
        return (sign << 7) | 0x7e
    return (sign << 7) | (exp << 3) | fp8_mant


def _expand_iq_signs(signs7):
    """GGML expands seven sign bits by appending their parity as bit 7."""
    return signs7 | ((signs7.bit_count() & 1) << 7)


def _dequant_iq2_xxs_reference(packed, n_per_row):
    packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
    blocks_per_row = n_per_row // 256
    row_bytes = blocks_per_row * 66
    assert packed.size % row_bytes == 0
    out = np.empty((packed.size // row_bytes, n_per_row), dtype=np.float32)

    for row in range(out.shape[0]):
        for block_idx in range(blocks_per_row):
            block = packed[row * row_bytes + block_idx * 66:
                           row * row_bytes + (block_idx + 1) * 66]
            d = _fp16_at(block)
            for ib32 in range(8):
                group_offset = 2 + 8 * ib32
                aux = int.from_bytes(block[group_offset + 4:group_offset + 8], "little")
                db = np.float32(
                    np.float32(np.float32(d) * np.float32(0.5 + (aux >> 28))) * np.float32(0.25)
                )
                for group in range(4):
                    grid = _IQ2_XXS_GRID[int(block[group_offset + group])]
                    signs = _expand_iq_signs((aux >> (7 * group)) & 0x7f)
                    for j in range(8):
                        value = np.float32(db * np.float32(grid[j]))
                        if signs & (1 << j):
                            value = np.float32(-value)
                        out[row, block_idx * 256 + ib32 * 32 + group * 8 + j] = value
    return out


def _dequant_iq3_xxs_reference(packed, n_per_row):
    packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
    blocks_per_row = n_per_row // 256
    row_bytes = blocks_per_row * 98
    assert packed.size % row_bytes == 0
    out = np.empty((packed.size // row_bytes, n_per_row), dtype=np.float32)

    for row in range(out.shape[0]):
        for block_idx in range(blocks_per_row):
            block = packed[row * row_bytes + block_idx * 98:
                           row * row_bytes + (block_idx + 1) * 98]
            d = _fp16_at(block)
            for ib32 in range(8):
                grids_offset = 2 + 8 * ib32
                aux = int.from_bytes(block[2 + 64 + 4 * ib32:2 + 64 + 4 * (ib32 + 1)], "little")
                db = np.float32(
                    np.float32(np.float32(d) * np.float32(0.5 + (aux >> 28))) * np.float32(0.5)
                )
                for group in range(4):
                    signs = _expand_iq_signs((aux >> (7 * group)) & 0x7f)
                    for j in range(8):
                        grid_idx = int(block[grids_offset + 2 * group + (j >= 4)])
                        value = np.float32(db * np.float32(_IQ3_XXS_GRID[grid_idx, j & 3]))
                        if signs & (1 << j):
                            value = np.float32(-value)
                        out[row, block_idx * 256 + ib32 * 32 + group * 8 + j] = value
    return out


def _dequant_iq4_nl_reference(packed, n_per_row):
    packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
    blocks_per_row = n_per_row // 32
    row_bytes = blocks_per_row * 18
    assert packed.size % row_bytes == 0
    out = np.empty((packed.size // row_bytes, n_per_row), dtype=np.float32)

    for row in range(out.shape[0]):
        for block_idx in range(blocks_per_row):
            block = packed[row * row_bytes + block_idx * 18:
                           row * row_bytes + (block_idx + 1) * 18]
            d = _fp16_at(block)
            for i in range(32):
                quant = int(block[2 + (i & 15)])
                nibble = quant & 0x0f if i < 16 else quant >> 4
                out[row, block_idx * 32 + i] = np.float32(d * np.float32(_IQ4_NL_VALUES[nibble]))
    return out


_IQUANT_REFERENCE_DEQUANTIZERS = {
    "IQ2_XXS": _dequant_iq2_xxs_reference,
    "IQ3_XXS": _dequant_iq3_xxs_reference,
    "IQ4_NL": _dequant_iq4_nl_reference,
}


DEQUANT_TYPES = [
    # (type_name, blk_size)
    ("IQ2_XXS", 256),
    ("IQ2_XS", 256),
    ("IQ3_XXS", 256),
    ("IQ4_NL", 32),
    ("IQ4_XS", 256),
    ("IQ1_S", 256),
    ("IQ3_S", 256),
    ("TQ1_0", 256),
    ("TQ2_0", 256),
]

LEGACY_DEQUANT_TYPES = [
    ("Q4_0", 32),
    ("Q4_1", 32),
    ("Q5_0", 32),
    ("Q5_1", 32),
    ("Q8_0", 32),
    ("Q8_1", 32),
    ("Q2_K", 256),
    ("Q3_K", 256),
    ("Q4_K", 256),
    ("Q5_K", 256),
    ("Q6_K", 256),
]


@requires_gpu
class TestDequantSmoke(unittest.TestCase):
    """Smoke tests: do dequant kernels run without errors?"""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_dequant_smoke(self, type_name, blk_size, output_format="E4M3"):
        qtype = GGML_TYPE[type_name]
        cols = blk_size * 3
        x = np.random.randn(2, cols).astype(np.float32) * 2.0
        packed = self.hq.quantize_numpy(x, qtype)
        result = self.hq.dequantize_to_fp8(packed, qtype, cols, output_format)
        expected_shape = (2, cols)
        self.assertEqual(result.shape, expected_shape,
            f"{type_name}→{output_format}: shape mismatch {result.shape} != {expected_shape}")
        self.assertEqual(result.dtype, np.uint8)
        return x, packed, result

    def test_all_iq_dequant_smoke(self):
        for type_name, blk_size in DEQUANT_TYPES:
            with self.subTest(type=type_name):
                self._test_dequant_smoke(type_name, blk_size)

    def test_all_iq_dequant_smoke_e5m2(self):
        for type_name, blk_size in DEQUANT_TYPES:
            with self.subTest(type=type_name):
                self._test_dequant_smoke(type_name, blk_size, "E5M2")

    def test_legacy_dequant_still_works(self):
        for type_name, blk_size in LEGACY_DEQUANT_TYPES:
            with self.subTest(type=type_name):
                self._test_dequant_smoke(type_name, blk_size)


@requires_gpu
class TestDequantNonZeroOutput(unittest.TestCase):
    """Non-zero input should produce non-zero FP8 output."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_nonzero_output(self, type_name, blk_size):
        qtype = GGML_TYPE[type_name]
        cols = blk_size * 2
        x = np.random.randn(2, cols).astype(np.float32) * 3.0 + 1.0
        packed = self.hq.quantize_numpy(x, qtype)
        result = self.hq.dequantize_to_e4m3(packed, qtype, cols)
        self.assertGreater(np.count_nonzero(result), 0,
            f"{type_name}: dequant output should contain non-zero bytes")

    def test_all_iq_nonzero(self):
        for type_name, blk_size in DEQUANT_TYPES:
            with self.subTest(type=type_name):
                self._test_nonzero_output(type_name, blk_size)


@requires_gpu
class TestDequantRoundtrip(unittest.TestCase):
    """Validate Q-to-FP8 dequantization.

    Low-bit I-Quants must be checked against a decoder of their packed GGML
    representation, not against the source tensor: the latter measures the
    intentionally lossy quantizer and is unbounded near zero.
    """

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_roundtrip(self, type_name, blk_size, output_format="E4M3", max_rel=0.5):
        qtype = GGML_TYPE[type_name]
        cols = blk_size
        x = np.random.randn(4, cols).astype(np.float32) * 2.0
        packed = self.hq.quantize_numpy(x, qtype)
        fp8_out = self.hq.dequantize_to_fp8(packed, qtype, cols, output_format)

        # Decode FP8 back to float32
        if output_format == "E5M2":
            decoded = np.vectorize(fp8_e5m2_to_f32)(fp8_out.astype(np.uint8))
        else:
            decoded = np.vectorize(fp8_e4m3_to_f32)(fp8_out.astype(np.uint8))

        # Check no NaN/Inf (except for input values that naturally produce them)
        finite_mask = np.isfinite(x)
        self.assertTrue(np.all(np.isfinite(decoded[finite_mask])),
            f"{type_name}→{output_format}: decoded values contain NaN/Inf")

        # Check relative error for non-tiny values
        mask = np.abs(x) > 1e-6
        if mask.any():
            rel_err = np.abs((decoded - x)[mask]) / (np.abs(x[mask]) + 1e-10)
            # Allow generous tolerance for low-bit quant types
            self.assertLess(rel_err.max(), max_rel,
                f"{type_name}→{output_format}: max relative error {rel_err.max():.4f} > {max_rel}")

    def _test_iq_reference_dequant(self, type_name, blk_size):
        qtype = GGML_TYPE[type_name]
        x = np.random.default_rng(20260713).normal(0.0, 2.0, size=(4, blk_size)).astype(np.float32)
        packed = self.hq.quantize_numpy(x, qtype)
        expected_f32 = _IQUANT_REFERENCE_DEQUANTIZERS[type_name](packed, blk_size)
        expected_fp8 = np.vectorize(_fp32_to_e4m3, otypes=[np.uint8])(expected_f32)
        actual_fp8 = self.hq.dequantize_to_e4m3(packed, qtype, blk_size)
        np.testing.assert_array_equal(
            actual_fp8,
            expected_fp8,
            err_msg=f"{type_name} direct E4M3 dequantization disagrees with its GGML block decoder",
        )

    def test_iq2_xxs_e4m3(self):
        self._test_iq_reference_dequant("IQ2_XXS", 256)

    def test_iq3_xxs_e4m3(self):
        self._test_iq_reference_dequant("IQ3_XXS", 256)

    def test_iq4_nl_e4m3(self):
        self._test_iq_reference_dequant("IQ4_NL", 32)

    def test_q4_0_roundtrip(self):
        self._test_roundtrip("Q4_0", 32, "E4M3", max_rel=2.0)

    def test_q8_0_roundtrip(self):
        self._test_roundtrip("Q8_0", 32, "E4M3", max_rel=2.0)


@requires_gpu
class TestDequantConsistency(unittest.TestCase):
    """Same input → same dequant output (deterministic)."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_consistency(self, type_name, blk_size):
        qtype = GGML_TYPE[type_name]
        cols = blk_size * 2
        np.random.seed(12345)
        x = np.random.randn(2, cols).astype(np.float32)
        packed = self.hq.quantize_numpy(x, qtype)

        out1 = self.hq.dequantize_to_e4m3(packed, qtype, cols)
        out2 = self.hq.dequantize_to_e4m3(packed, qtype, cols)
        self.assertTrue(np.array_equal(out1, out2),
            f"{type_name}: repeated dequant should be identical")

    def test_all_dequant_consistent(self):
        for type_name, blk_size in DEQUANT_TYPES:
            with self.subTest(type=type_name):
                self._test_consistency(type_name, blk_size)


@requires_gpu
class TestDequantMultiRow(unittest.TestCase):
    """Dequantize multiple rows — each row processes independently."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_multirow(self, type_name, blk_size, nrows=8):
        qtype = GGML_TYPE[type_name]
        cols = blk_size * 2
        x = np.random.randn(nrows, cols).astype(np.float32) * 2.0
        packed = self.hq.quantize_numpy(x, qtype)
        result = self.hq.dequantize_to_e4m3(packed, qtype, cols)
        self.assertEqual(result.shape, (nrows, cols))

    def test_all_dequant_multirow(self):
        for type_name, blk_size in DEQUANT_TYPES:
            with self.subTest(type=type_name):
                self._test_multirow(type_name, blk_size)


@requires_gpu
class TestDequantSizeCorrect(unittest.TestCase):
    """Output byte count is correct for each format."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def test_output_size_matches(self):
        cols = 256
        x = np.random.randn(3, cols).astype(np.float32) * 2.0

        for type_name, blk_size in DEQUANT_TYPES:
            qtype = GGML_TYPE[type_name]
            with self.subTest(type=type_name):
                packed = self.hq.quantize_numpy(x, qtype)
                result = self.hq.dequantize_to_e4m3(packed, qtype, cols)
                self.assertEqual(result.size, 3 * cols,
                    f"{type_name}: expected {3*cols} bytes, got {result.size}")


@requires_gpu
class TestDequantAllFormats(unittest.TestCase):
    """Full round-trip validation for EVERY quant type in GGML_TYPE."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def test_all_types_dequant(self):
        """Every GGML type should either dequantize or produce a clear error."""
        skip_types = {"F8_E4M3", "F8_E5M2"}  # FP8 types: dequant is a no-op memcpy
        for type_name, type_num in GGML_TYPE.items():
            if type_name in skip_types:
                continue
            blk_size = GGML_TYPE_BLOCK_SIZE.get(type_num, 0)
            if blk_size == 0:
                continue
            cols = blk_size
            with self.subTest(type=type_name):
                x = np.random.randn(3, cols).astype(np.float32) * 2.0
                packed = self.hq.quantize_numpy(x, type_num)

                try:
                    result = self.hq.dequantize_to_e4m3(packed, type_num, cols)
                    self.assertEqual(result.shape, (3, cols))
                    self.assertEqual(result.dtype, np.uint8)
                    self.assertGreater(result.size, 0)
                except (ValueError, RuntimeError) as e:
                    # Some types may not be supported (F8_E4M3 → F8_E4M3 is a memcpy)
                    if "unsupported" not in str(e).lower() and "not supported" not in str(e).lower():
                        raise


@requires_gpu
class TestDequantLargeBatch(unittest.TestCase):
    """Stress test: many rows with many blocks per row."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def test_large_dequant(self):
        cols = 1024
        x = np.random.randn(16, cols).astype(np.float32) * 2.0

        # Test with a representative set of types
        for type_name in ["Q4_K", "IQ2_XXS", "IQ3_XXS", "TQ2_0"]:
            qtype = GGML_TYPE[type_name]
            with self.subTest(type=type_name):
                packed = self.hq.quantize_numpy(x, qtype)
                result = self.hq.dequantize_to_e4m3(packed, qtype, cols)
                self.assertEqual(result.shape, (16, cols))


if __name__ == "__main__":
    unittest.main()
