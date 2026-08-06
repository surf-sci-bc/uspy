"""
Minimal, dependency-free HDF4 reader for SMART/BESSY `.hdf` files.

Used by `uspy.leem.readers.smart` to read HDF4 files without any native HDF4 library. Not a
general-purpose HDF4 library -- only the sliver SMART files actually use:

* the file-description/file-label annotations (`DFTAG_FD`/`DFTAG_FID`), storing metadata as
  free-text "KEY = VALUE;" pairs, and
* the single Scientific Data Set (`DFTAG_NDG` group "Range0", int16, shape
  ``(n_frames, height, width)``), stored contiguously or -- as in every real SMART file --
  as an HDF4 *linked-block* special element.

Compressed/chunked storage is not supported and raises; SMART never writes those.

An HDF4 file is a magic number followed by a linked list of *data descriptor* (DD) blocks;
each DD is ``tag(2) ref(2) offset(4) length(4)`` (big-endian), pointing at a raw byte range
elsewhere in the file. See the HDF4 specification for tag meanings referenced below.
"""
# pylint: disable=invalid-name

from __future__ import annotations
import os
import struct
from typing import Union

import numpy as np

_MAGIC = b"\x0e\x03\x13\x01"

# HDF4 tags we care about.
_DFTAG_NULL = 1     # unused DD slot
_DFTAG_LINKED = 20  # linked-block table *and* the data blocks it points at
_DFTAG_FID = 100    # file label (original acquisition path)
_DFTAG_FD = 101     # file description (the "KEY = VALUE;" metadata)
_DFTAG_NT = 106     # number type
_DFTAG_SDD = 701    # SD dimension record (rank + dim sizes)
_DFTAG_SD = 702     # SD data
_DFTAG_NDG = 720    # numeric data group (ties SD/SDD/NT together)

# A data element is a "special" element (linked block / external / compressed / chunked) when
# its DD tag has this bit set; the element then starts with a 2-byte special code.
_SPECIAL_TAG_BIT = 0x4000
_SPECIAL_LINKED = 1
_SPECIAL_EXT = 2
_SPECIAL_COMP = 3
_SPECIAL_CHUNKED = 4

# HDF4 number type (DFNT_*) -> numpy base type string (byte order added from the NT class byte).
_NT_TO_NUMPY = {
    3: "u1", 4: "i1",        # DFNT_UCHAR8, DFNT_CHAR8
    5: "f4", 6: "f8",        # DFNT_FLOAT32, DFNT_FLOAT64
    20: "i1", 21: "u1",      # DFNT_INT8, DFNT_UINT8
    22: "i2", 23: "u2",      # DFNT_INT16, DFNT_UINT16
    24: "i4", 25: "u4",      # DFNT_INT32, DFNT_UINT32
}
# NT class byte: Intel/PC is little-endian; everything else (IEEE/Motorola, the HDF default) is
# big-endian.
_NT_CLASS_PC = 4


class Hdf4Error(RuntimeError):
    """Raised for HDF4 structures this minimal reader does not support (e.g. compressed data)."""


class Hdf4File:
    """Read SMART's metadata and image frames from an HDF4 file. File structure (DD table,
    annotations, dimensions/type, block layout) is parsed once up front; pixel data isn't
    read until `read_frame`, which touches only the blocks for the requested frame."""

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self._path = os.fspath(path)
        with open(self._path, "rb") as f:
            if f.read(4) != _MAGIC:
                raise Hdf4Error(f"{self._path!r} is not an HDF4 file (bad magic number).")
            dd = self._walk_dds(f)

            self._description = self._read_annotation(f, dd, _DFTAG_FD)
            self._label = self._read_annotation(f, dd, _DFTAG_FID)

            sd_ref, sdd_ref, nt_ref = self._find_sds(f, dd)
            self._dtype = self._read_dtype(f, dd, nt_ref)
            _sdd_frames, height, width = self._read_dims(f, dd, sdd_ref)
            self._blocks, data_length = self._read_sd_layout(f, dd, sd_ref)

        # Frame count comes from the actual data length, not the SDD record: HDF4 records
        # an unlimited/appendable first dimension as 1 regardless of frames written.
        frame_nbytes = height * width * self._dtype.itemsize
        if frame_nbytes == 0 or data_length % frame_nbytes != 0:
            raise Hdf4Error(
                f"data length {data_length} is not a whole number of "
                f"{height}x{width} {self._dtype} frames."
            )
        self._shape = (data_length // frame_nbytes, height, width)

    # -- construction helpers ---------------------------------------------------------------

    @staticmethod
    def _walk_dds(f) -> dict:
        """Return ``{(tag, ref): (offset, length)}`` for every non-empty DD in the file."""
        dd = {}
        f.seek(4)  # past the magic number, at the first DD block header
        while True:
            n_dd, next_block = struct.unpack(">hi", f.read(6))
            raw = f.read(12 * n_dd)
            for i in range(n_dd):
                tag, ref, offset, length = struct.unpack_from(">HHii", raw, 12 * i)
                if tag not in (0, _DFTAG_NULL):
                    dd[(tag, ref)] = (offset, length)
            if next_block <= 0:
                return dd
            f.seek(next_block)

    @staticmethod
    def _read_annotation(f, dd: dict, tag: int) -> str:
        """Text of the first annotation with `tag` (file description or label), or ``""``."""
        refs = sorted(ref for (t, ref) in dd if t == tag)
        if not refs:
            return ""
        offset, length = dd[(tag, refs[0])]
        f.seek(offset)
        return f.read(length).decode("latin-1").rstrip("\x00")

    @staticmethod
    def _find_sds(f, dd: dict) -> tuple:
        """Return ``(sd_ref, sdd_ref, nt_ref)`` for the single Scientific Data Set.

        SMART files contain exactly one numeric data group (`DFTAG_NDG`), so we locate it
        rather than resolving the "Range0" name through vgroups. The NDG element is a flat list
        of ``(member_tag, member_ref)`` pairs pointing at the data, dimension and number-type
        records.
        """
        ndgs = [ref for (t, ref) in dd if t == _DFTAG_NDG]
        if len(ndgs) != 1:
            raise Hdf4Error(
                f"expected exactly one Scientific Data Set (DFTAG_NDG), found {len(ndgs)} "
                f"-- not a SMART file this fallback understands."
            )
        offset, length = dd[(_DFTAG_NDG, ndgs[0])]
        f.seek(offset)
        raw = f.read(length)
        members = {tag: ref for tag, ref in struct.iter_unpack(">HH", raw)}
        try:
            return members[_DFTAG_SD], members[_DFTAG_SDD], members[_DFTAG_NT]
        except KeyError as exc:
            raise Hdf4Error(f"NDG group is missing member {exc} (members: {members}).") from exc

    @staticmethod
    def _read_dims(f, dd: dict, sdd_ref: int) -> tuple:
        """``(n_frames, height, width)`` from the SD dimension record (`DFTAG_SDD`)."""
        offset, _length = dd[(_DFTAG_SDD, sdd_ref)]
        f.seek(offset)
        (rank,) = struct.unpack(">h", f.read(2))
        dims = struct.unpack(f">{rank}i", f.read(4 * rank))
        if rank != 3:
            raise Hdf4Error(f"expected a 3-D SMART dataset, got rank {rank} dims {dims}.")
        return dims

    @staticmethod
    def _read_dtype(f, dd: dict, nt_ref: int) -> np.dtype:
        """numpy dtype (with byte order) from the number-type record (`DFTAG_NT`)."""
        offset, _length = dd[(_DFTAG_NT, nt_ref)]
        f.seek(offset)
        _version, nt_type, _width_bits, nt_class = struct.unpack(">BBBB", f.read(4))
        base = _NT_TO_NUMPY.get(nt_type)
        if base is None:
            raise Hdf4Error(f"unsupported HDF4 number type {nt_type}.")
        byteorder = "<" if nt_class == _NT_CLASS_PC else ">"
        return np.dtype(byteorder + base)

    def _read_sd_layout(self, f, dd: dict, sd_ref: int) -> tuple:
        """Describe where the pixel bytes live (without reading them) and their total length.

        Returns ``(layout, data_length)`` where `layout` is either ``("contiguous", offset)``
        or a list of ``(file_offset, nbytes)`` blocks in logical order (for linked-block
        storage), and `data_length` is the dataset's true byte length.
        """
        if (_DFTAG_SD, sd_ref) in dd:  # plain contiguous data
            offset, length = dd[(_DFTAG_SD, sd_ref)]
            return ("contiguous", offset), length

        special_key = (_DFTAG_SD | _SPECIAL_TAG_BIT, sd_ref)
        if special_key not in dd:
            raise Hdf4Error("Scientific Data Set has no data element.")
        offset, _length = dd[special_key]
        f.seek(offset)
        (sp_code,) = struct.unpack(">h", f.read(2))
        if sp_code == _SPECIAL_LINKED:
            return self._read_linked_blocks(f, dd)
        if sp_code in (_SPECIAL_COMP, _SPECIAL_CHUNKED, _SPECIAL_EXT):
            raise Hdf4Error(
                "this SMART file uses compressed/chunked/external HDF4 storage, which this "
                "reader does not support -- SMART only ever writes contiguous or linked-block "
                "storage."
            )
        raise Hdf4Error(f"unsupported HDF4 special element code {sp_code}.")

    @staticmethod
    def _read_linked_blocks(f, dd: dict) -> tuple:
        """Follow a linked-block special element into ``(blocks, data_length)``, where `blocks`
        is a list of ``(file_offset, nbytes)`` in logical order.

        The special header (already past its 2-byte code, at the current file position) gives
        the total data length, the per-block length, the number of block references per link
        table, and the reference of the first link table. Link tables (`DFTAG_LINKED`) chain
        via their leading `next_ref`; each holds `num_blk` references to `DFTAG_LINKED` data
        blocks. Trailing zero references mark the end.
        """
        length, block_len, num_blk = struct.unpack(">iii", f.read(12))
        (link_ref,) = struct.unpack(">H", f.read(2))

        block_refs = []
        while link_ref != 0:
            offset, _tlen = dd[(_DFTAG_LINKED, link_ref)]
            f.seek(offset)
            (next_ref,) = struct.unpack(">H", f.read(2))
            for (ref,) in struct.iter_unpack(">H", f.read(2 * num_blk)):
                if ref == 0:
                    next_ref = 0  # a zero reference terminates the block list
                    break
                block_refs.append(ref)
            link_ref = next_ref

        blocks = []
        logical = 0
        for ref in block_refs:
            if logical >= length:
                break
            file_offset, _blen = dd[(_DFTAG_LINKED, ref)]
            nbytes = min(block_len, length - logical)  # last block is only partly filled
            blocks.append((file_offset, nbytes))
            logical += nbytes
        if logical != length:
            raise Hdf4Error(f"linked-block data is short: got {logical} of {length} bytes.")
        return blocks, length

    # -- public API -------------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        """``(n_frames, height, width)`` of the Range0 dataset."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """numpy dtype of the raw pixel data (e.g. big-endian int16)."""
        return self._dtype

    def file_description(self) -> str:
        """The free-text file-description annotation (SMART's "KEY = VALUE;" metadata)."""
        return self._description

    def file_label(self) -> str:
        """The file-label annotation (original acquisition path); unused by the reader."""
        return self._label

    def read_frame(self, index: int) -> np.ndarray:
        """Return frame `index` as a ``(height, width)`` array in the file's native dtype."""
        n_frames, height, width = self._shape
        if not 0 <= index < n_frames:
            raise IndexError(f"frame index {index} out of range for {n_frames} frames.")
        frame_nbytes = height * width * self._dtype.itemsize
        lo = index * frame_nbytes
        hi = lo + frame_nbytes

        with open(self._path, "rb") as f:
            if self._blocks[0] == "contiguous":
                f.seek(self._blocks[1] + lo)
                raw = f.read(frame_nbytes)
            else:
                raw = self._read_byte_range(f, lo, hi)
        return np.frombuffer(raw, dtype=self._dtype).reshape(height, width)

    def _read_byte_range(self, f, lo: int, hi: int) -> bytes:
        """Read logical bytes ``[lo, hi)`` of the linked-block dataset (only needed blocks)."""
        out = bytearray()
        pos = 0
        for file_offset, nbytes in self._blocks:
            blk_lo, blk_hi = pos, pos + nbytes
            pos = blk_hi
            if blk_hi <= lo:
                continue
            if blk_lo >= hi:
                break
            start = max(blk_lo, lo) - blk_lo
            end = min(blk_hi, hi) - blk_lo
            f.seek(file_offset + start)
            out += f.read(end - start)
        return bytes(out)
