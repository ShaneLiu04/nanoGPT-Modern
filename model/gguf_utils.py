"""Minimal dependency-free GGUF writer/reader for nanoGPT-Modern.

Supports the container metadata plus ``F32``, ``F16`` and ``Q8_0`` tensor
storage.  This intentionally avoids the external ``gguf`` package so the
export path works on Windows and in lightweight environments.
"""

from __future__ import annotations

import io
import struct
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

GGUF_MAGIC = int.from_bytes(b"GGUF", "little")
GGUF_VERSION = 3


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


_QUANT_BLOCK = {GGMLQuantizationType.F32: (1, 4), GGMLQuantizationType.F16: (1, 2)}


def _align(offset: int, alignment: int = 32) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


def quantize_q8_0(tensor: Union[torch.Tensor, np.ndarray]) -> bytes:
    """Quantize a tensor to GGUF Q8_0 block format.

    Block layout: ``int8[32] qs`` preceded by a ``float16`` scale ``d``.
    The tensor is flattened in C order, padded to a multiple of 32, and
    quantized symmetrically per block.
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().contiguous().view(-1).float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32).ravel()

    n = arr.size
    n_blocks = (n + 31) // 32
    pad = n_blocks * 32 - n
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.float32)])
    blocks = arr.reshape(n_blocks, 32)

    amax = np.abs(blocks).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, np.ones_like(amax))
    quants = np.rint(blocks / scale).clip(-127, 127).astype(np.int8)
    scale = scale.astype(np.float16)

    interleaved = np.empty((n_blocks, 34), dtype=np.uint8)
    interleaved[:, :2] = scale.view(np.uint8).reshape(n_blocks, 2)
    interleaved[:, 2:] = quants.view(np.uint8)
    return interleaved.tobytes()


def dequantize_q8_0(data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantize Q8_0 bytes back to a float32 array of ``shape``."""
    block_size = 34
    n_total = int(np.prod(shape))
    n_blocks = (n_total + 31) // 32
    if len(data) < n_blocks * block_size:
        raise ValueError("Q8_0 data too short for the requested shape")

    arr = np.frombuffer(data[: n_blocks * block_size], dtype=np.uint8).copy()
    arr = arr.reshape(n_blocks, block_size)
    scale = arr[:, :2].view(np.float16).reshape(n_blocks, 1)
    quants = arr[:, 2:].view(np.int8).reshape(n_blocks, 32)
    out = quants.astype(np.float32) * scale.astype(np.float32)
    return out.ravel()[:n_total].reshape(shape)


def _tensor_raw_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """GGUF stores tensor dimensions in reverse order."""
    return tuple(reversed(shape))


def _dtype_np(quant: GGMLQuantizationType) -> Optional[np.dtype]:
    if quant == GGMLQuantizationType.F32:
        return np.dtype("float32")
    if quant == GGMLQuantizationType.F16:
        return np.dtype("float16")
    return None


class GGUFWriter:
    """Minimal GGUF writer for F32/F16/Q8_0 tensors."""

    def __init__(self, path: Union[str, Path], architecture: str = "nanogpt-modern"):
        self.path = Path(path)
        self.architecture = architecture
        self._metadata: List[Tuple[str, GGUFValueType, Any]] = []
        self._tensors: List[
            Tuple[str, GGMLQuantizationType, Tuple[int, ...], bytes]
        ] = []
        self.add_architecture(architecture)

    def add_key_value(self, key: str, value: Any, value_type: GGUFValueType) -> None:
        self._metadata.append((key, value_type, value))

    def add_architecture(self, value: str) -> None:
        self.add_key_value("general.architecture", value, GGUFValueType.STRING)

    def add_name(self, value: str) -> None:
        self.add_key_value("general.name", value, GGUFValueType.STRING)

    def add_context_length(self, value: int) -> None:
        self.add_key_value(
            f"{self.architecture}.context_length", value, GGUFValueType.UINT32
        )

    def add_embedding_length(self, value: int) -> None:
        self.add_key_value(
            f"{self.architecture}.embedding_length", value, GGUFValueType.UINT32
        )

    def add_block_count(self, value: int) -> None:
        self.add_key_value(
            f"{self.architecture}.block_count", value, GGUFValueType.UINT32
        )

    def add_feed_forward_length(self, value: int) -> None:
        self.add_key_value(
            f"{self.architecture}.feed_forward_length", value, GGUFValueType.UINT32
        )

    def add_head_count(self, value: int) -> None:
        self.add_key_value(
            f"{self.architecture}.attention.head_count", value, GGUFValueType.UINT32
        )

    def add_head_count_kv(self, value: int) -> None:
        self.add_key_value(
            f"{self.architecture}.attention.head_count_kv", value, GGUFValueType.UINT32
        )

    def add_tensor(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray],
        quant_type: GGMLQuantizationType = GGMLQuantizationType.F32,
    ) -> None:
        """Add a tensor to the GGUF file.

        ``data`` must be in C-order.  Its shape is stored reversed, matching the
        GGUF convention.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().contiguous()
            shape = tuple(data.shape)
        else:
            data = np.asarray(data, order="C")
            shape = tuple(data.shape)

        if quant_type == GGMLQuantizationType.Q8_0:
            raw = quantize_q8_0(data)
        else:
            np_dtype = _dtype_np(quant_type)
            if np_dtype is None:
                raise ValueError(f"Unsupported GGUF tensor type: {quant_type}")
            raw = np.asarray(data, dtype=np_dtype).tobytes()

        self._tensors.append((name, quant_type, shape, raw))

    @staticmethod
    def _string_size(value: str) -> int:
        encoded = value.encode("utf-8")
        return 8 + len(encoded)

    @staticmethod
    def _write_string(f: io.BufferedWriter, value: str) -> None:
        encoded = value.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    @classmethod
    def _value_size(cls, value_type: GGUFValueType, value: Any) -> int:
        if value_type == GGUFValueType.UINT8:
            return 1
        if value_type == GGUFValueType.INT8:
            return 1
        if value_type == GGUFValueType.UINT16:
            return 2
        if value_type == GGUFValueType.INT16:
            return 2
        if value_type == GGUFValueType.UINT32:
            return 4
        if value_type == GGUFValueType.INT32:
            return 4
        if value_type == GGUFValueType.FLOAT32:
            return 4
        if value_type == GGUFValueType.BOOL:
            return 1
        if value_type == GGUFValueType.STRING:
            return cls._string_size(value)
        if value_type == GGUFValueType.ARRAY:
            item_type, items = value
            size = 4 + 8  # item type + array length
            for item in items:
                size += cls._value_size(item_type, item)
            return size
        if value_type == GGUFValueType.UINT64:
            return 8
        if value_type == GGUFValueType.INT64:
            return 8
        if value_type == GGUFValueType.FLOAT64:
            return 8
        raise ValueError(f"Unsupported value type: {value_type}")

    @classmethod
    def _write_value(
        cls, f: io.BufferedWriter, value_type: GGUFValueType, value: Any
    ) -> None:
        f.write(struct.pack("<I", int(value_type)))
        if value_type == GGUFValueType.UINT8:
            f.write(struct.pack("<B", value))
        elif value_type == GGUFValueType.INT8:
            f.write(struct.pack("<b", value))
        elif value_type == GGUFValueType.UINT16:
            f.write(struct.pack("<H", value))
        elif value_type == GGUFValueType.INT16:
            f.write(struct.pack("<h", value))
        elif value_type == GGUFValueType.UINT32:
            f.write(struct.pack("<I", value))
        elif value_type == GGUFValueType.INT32:
            f.write(struct.pack("<i", value))
        elif value_type == GGUFValueType.FLOAT32:
            f.write(struct.pack("<f", value))
        elif value_type == GGUFValueType.BOOL:
            f.write(struct.pack("<B", 1 if value else 0))
        elif value_type == GGUFValueType.STRING:
            cls._write_string(f, value)
        elif value_type == GGUFValueType.ARRAY:
            item_type, items = value
            f.write(struct.pack("<I", int(item_type)))
            f.write(struct.pack("<Q", len(items)))
            for item in items:
                cls._write_value(f, item_type, item)
        elif value_type == GGUFValueType.UINT64:
            f.write(struct.pack("<Q", value))
        elif value_type == GGUFValueType.INT64:
            f.write(struct.pack("<q", value))
        elif value_type == GGUFValueType.FLOAT64:
            f.write(struct.pack("<d", value))
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    @classmethod
    def _tensor_info_size(cls, name: str, shape: Tuple[int, ...]) -> int:
        # name string + n_dims uint32 + dims uint64 * ndim + dtype uint32 + offset uint64
        return cls._string_size(name) + 4 + 8 * len(shape) + 4 + 8

    @classmethod
    def _write_tensor_info(
        cls,
        f: io.BufferedWriter,
        name: str,
        shape: Tuple[int, ...],
        quant_type: GGMLQuantizationType,
        offset: int,
    ) -> None:
        cls._write_string(f, name)
        raw_shape = _tensor_raw_shape(shape)
        f.write(struct.pack("<I", len(raw_shape)))
        for dim in raw_shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", int(quant_type)))
        f.write(struct.pack("<Q", offset))

    def _metadata_size(self) -> int:
        size = 0
        for key, vtype, value in self._metadata:
            size += self._string_size(key)
            size += self._value_size(vtype, value)
        return size

    def write(self) -> None:
        """Write the complete GGUF file to disk."""
        header_size = 24
        metadata_size = self._metadata_size()
        info_size = sum(
            self._tensor_info_size(name, shape) for name, _, shape, _ in self._tensors
        )
        data_start = _align(header_size + metadata_size + info_size)

        offsets: List[int] = []
        offset = data_start
        for _, _, _, raw in self._tensors:
            offsets.append(offset)
            offset += len(raw)
            offset = _align(offset)

        with open(self.path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self._tensors)))
            f.write(struct.pack("<Q", len(self._metadata)))

            for key, vtype, value in self._metadata:
                self._write_string(f, key)
                self._write_value(f, vtype, value)

            for (name, quant_type, shape, _), off in zip(self._tensors, offsets):
                self._write_tensor_info(f, name, shape, quant_type, off)

            pad = data_start - f.tell()
            if pad > 0:
                f.write(b"\x00" * pad)

            for _, _, _, raw in self._tensors:
                f.write(raw)
                pad = _align(f.tell()) - f.tell()
                if pad > 0:
                    f.write(b"\x00" * pad)


# ---------------------------------------------------------------------------
# Minimal reader helpers used by tests / inspection.
# ---------------------------------------------------------------------------


def _read_string(mm: np.memmap, offset: int) -> Tuple[str, int]:
    length = int(np.frombuffer(mm[offset : offset + 8].tobytes(), dtype=np.uint64)[0])
    offset += 8
    value = bytes(mm[offset : offset + length].tobytes()).decode("utf-8")
    return value, 8 + length


def _read_value(
    mm: np.memmap, offset: int, value_type: GGUFValueType
) -> Tuple[Any, int]:
    if value_type == GGUFValueType.UINT8:
        return int(mm[offset]), 1
    if value_type == GGUFValueType.INT8:
        return int(mm[offset].view(np.int8)), 1
    if value_type == GGUFValueType.UINT16:
        return int(np.frombuffer(mm[offset : offset + 2].tobytes(), np.uint16)[0]), 2
    if value_type == GGUFValueType.INT16:
        return int(np.frombuffer(mm[offset : offset + 2].tobytes(), np.int16)[0]), 2
    if value_type == GGUFValueType.UINT32:
        return int(np.frombuffer(mm[offset : offset + 4].tobytes(), np.uint32)[0]), 4
    if value_type == GGUFValueType.INT32:
        return int(np.frombuffer(mm[offset : offset + 4].tobytes(), np.int32)[0]), 4
    if value_type == GGUFValueType.FLOAT32:
        return float(np.frombuffer(mm[offset : offset + 4].tobytes(), np.float32)[0]), 4
    if value_type == GGUFValueType.BOOL:
        return bool(mm[offset]), 1
    if value_type == GGUFValueType.STRING:
        return _read_string(mm, offset)
    if value_type == GGUFValueType.ARRAY:
        item_type = GGUFValueType(
            int(np.frombuffer(mm[offset : offset + 4].tobytes(), np.uint32)[0])
        )
        length = int(
            np.frombuffer(mm[offset + 4 : offset + 12].tobytes(), np.uint64)[0]
        )
        off = 12
        items = []
        for _ in range(length):
            item, item_len = _read_value(mm, offset + off, item_type)
            items.append(item)
            off += item_len
        return items, off
    if value_type == GGUFValueType.UINT64:
        return int(np.frombuffer(mm[offset : offset + 8].tobytes(), np.uint64)[0]), 8
    if value_type == GGUFValueType.INT64:
        return int(np.frombuffer(mm[offset : offset + 8].tobytes(), np.int64)[0]), 8
    if value_type == GGUFValueType.FLOAT64:
        return float(np.frombuffer(mm[offset : offset + 8].tobytes(), np.float64)[0]), 8
    raise ValueError(f"Unsupported value type: {value_type}")


def read_gguf_header(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse the GGUF header and metadata."""
    mm = np.memmap(path, mode="r")
    magic = int(np.frombuffer(mm[:4].tobytes(), np.uint32)[0])
    if magic != GGUF_MAGIC:
        raise ValueError(f"Invalid GGUF magic: {magic:#x}")
    version = int(np.frombuffer(mm[4:8].tobytes(), np.uint32)[0])
    tensor_count = int(np.frombuffer(mm[8:16].tobytes(), np.uint64)[0])
    kv_count = int(np.frombuffer(mm[16:24].tobytes(), np.uint64)[0])
    offset = 24

    metadata: Dict[str, Any] = {}
    for _ in range(kv_count):
        key, klen = _read_string(mm, offset)
        offset += klen
        value_type = GGUFValueType(
            int(np.frombuffer(mm[offset : offset + 4].tobytes(), np.uint32)[0])
        )
        value, vlen = _read_value(mm, offset + 4, value_type)
        offset += 4 + vlen
        metadata[key] = value

    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata": metadata,
        "metadata_end_offset": offset,
    }


def read_gguf_tensor_info(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Return a list of tensor metadata entries (name, shape, dtype, offset)."""
    header = read_gguf_header(path)
    mm = np.memmap(path, mode="r")
    offset = header["metadata_end_offset"]
    tensor_count = header["tensor_count"]
    tensors = []
    for _ in range(tensor_count):
        name, nlen = _read_string(mm, offset)
        offset += nlen
        n_dims = int(np.frombuffer(mm[offset : offset + 4].tobytes(), np.uint32)[0])
        offset += 4
        dims = tuple(
            int(
                np.frombuffer(
                    mm[offset + i * 8 : offset + (i + 1) * 8].tobytes(), np.uint64
                )[0]
            )
            for i in range(n_dims)
        )
        offset += 8 * n_dims
        dtype = GGMLQuantizationType(
            int(np.frombuffer(mm[offset : offset + 4].tobytes(), np.uint32)[0])
        )
        offset += 4
        data_offset = int(
            np.frombuffer(mm[offset : offset + 8].tobytes(), np.uint64)[0]
        )
        offset += 8
        # GGUF stores dims reversed; expose original shape.
        shape = tuple(reversed(dims))
        tensors.append(
            {"name": name, "shape": shape, "dtype": dtype, "offset": data_offset}
        )
    return tensors


def read_gguf_tensor_data(path: Union[str, Path], info: Dict[str, Any]) -> np.ndarray:
    """Read and dequantize a single tensor described by ``info``."""
    mm = np.memmap(path, mode="r")
    if info["dtype"] == GGMLQuantizationType.F32:
        n = int(np.prod(info["shape"]))
        return np.frombuffer(
            mm[info["offset"] : info["offset"] + n * 4].tobytes(), dtype=np.float32
        ).reshape(info["shape"])
    if info["dtype"] == GGMLQuantizationType.F16:
        n = int(np.prod(info["shape"]))
        return (
            np.frombuffer(
                mm[info["offset"] : info["offset"] + n * 2].tobytes(), dtype=np.float16
            )
            .astype(np.float32)
            .reshape(info["shape"])
        )
    if info["dtype"] == GGMLQuantizationType.Q8_0:
        return dequantize_q8_0(bytes(mm[info["offset"] :].tobytes()), info["shape"])
    raise ValueError(f"Unsupported dtype for reading: {info['dtype']}")
