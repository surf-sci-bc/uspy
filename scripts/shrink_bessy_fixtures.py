"""
Regenerate small SMART/BESSY HDF4 test fixtures by truncating a real ".hdf" file to its
first N frames.

Not part of the uspy package -- a one-off developer tool to shrink `testdata/bessy/*.hdf`
(full multi-GB SMART scans, untracked/local-only today) down to a size that can actually be
committed to the testdata submodule. The kept frames' pixel bytes are byte-identical to the
original file; only two metadata fields in the file-description ("KEY = VALUE;") annotation
are corrected to match the new frame count: "POINTS_1" (the frame count `SmartReader` itself
reads, see `uspy.leem.readers.smart._n_frames`) and, for a true energy scan (where
"MICRO_START_1" != "MICRO_END_1"), "MICRO_END_1" is recomputed to the real energy of the last
kept frame rather than left describing a range that no longer matches what's in the file.

See `uspy/leem/readers/_hdf4.py`'s module docstring for the HDF4 record kinds this mirrors
(DFTAG_FID/FD/NT/SDD/NDG/SD as a linked-block special element) -- this script only ever
writes those six kinds, matching exactly what `Hdf4File` reads, dropping everything else
(vgroup bookkeeping tags real SMART files also carry but the reader never touches).

Usage:
    python scripts/shrink_bessy_fixtures.py SOURCE.hdf DEST.hdf N_FRAMES

Requires the *original*, full-size ".hdf" file as input. The shrunk output cannot be used to
reconstruct the original -- keep a backup of the full-size file elsewhere if you need it for
anything else.
"""
from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uspy.leem.readers._hdf4 import (  # noqa: E402  pylint: disable=wrong-import-position
    _MAGIC,
    _DFTAG_FID,
    _DFTAG_FD,
    _DFTAG_NT,
    _DFTAG_SDD,
    _DFTAG_SD,
    _DFTAG_NDG,
    _DFTAG_LINKED,
    _SPECIAL_TAG_BIT,
    _SPECIAL_LINKED,
    Hdf4File,
)
from uspy.leem.readers.smart import _parse_fd_text  # noqa: E402


def _read_raw(f, dd: dict, tag: int, ref: int) -> bytes:
    offset, length = dd[(tag, ref)]
    f.seek(offset)
    return f.read(length)


def _annotation_ref(dd: dict, tag: int) -> int:
    refs = sorted(ref for (t, ref) in dd if t == tag)
    if not refs:
        raise ValueError(f"source file has no annotation with tag {tag}")
    return refs[0]


def _special_header(f, dd: dict, sd_ref: int) -> tuple:
    """(block_len, num_blk) from the source file's linked-block special header -- reused
    as-is so the shrunk file keeps genuine multi-block/multi-link-table chaining instead of
    collapsing to a single block, which would stop exercising that code path."""
    offset, _length = dd[(_DFTAG_SD | _SPECIAL_TAG_BIT, sd_ref)]
    f.seek(offset)
    (sp_code,) = struct.unpack(">h", f.read(2))
    if sp_code != _SPECIAL_LINKED:
        raise ValueError(f"source file's SD data isn't linked-block storage (code {sp_code})")
    _length, block_len, num_blk = struct.unpack(">iii", f.read(12))
    return block_len, num_blk


def _new_fd_text(fd_text: str, n_frames: int) -> str:
    fd = _parse_fd_text(fd_text)
    orig_points = int(float(fd["POINTS_1"]))
    new_text = re.sub(r"POINTS_1\s*=\s*[^;]*;", f"POINTS_1 = {n_frames};", fd_text)

    try:
        start = float(fd["MICRO_START_1"])
        end = float(fd["MICRO_END_1"])
    except (KeyError, ValueError):
        return new_text
    if start == end:  # fixed-energy (e.g. a TIME scan) -- nothing to recompute
        return new_text
    step = (end - start) / (orig_points - 1)
    new_end = start + (n_frames - 1) * step
    return re.sub(r"MICRO_END_1\s*=\s*[^;]*;", f"MICRO_END_1 = {new_end};", new_text)


def _write_hdf4(
    dest_path: str,
    label_raw: bytes,
    fd_raw: bytes,
    nt_raw: bytes,
    sdd_raw: bytes,
    pixel_blob: bytes,
    block_len: int,
    num_blk: int,
) -> None:
    ref_counter = [0]

    def next_ref() -> int:
        ref_counter[0] += 1
        return ref_counter[0]

    fid_ref, fd_ref, nt_ref, sdd_ref, ndg_ref, sd_ref = (next_ref() for _ in range(6))

    data_chunks = [pixel_blob[i : i + block_len] for i in range(0, len(pixel_blob), block_len)]
    data_refs = [next_ref() for _ in data_chunks]
    table_ref_groups = [data_refs[i : i + num_blk] for i in range(0, len(data_refs), num_blk)]
    table_refs = [next_ref() for _ in table_ref_groups]

    ndg_raw = struct.pack(">HHHHHH", _DFTAG_SD, sd_ref, _DFTAG_SDD, sdd_ref, _DFTAG_NT, nt_ref)
    special_raw = struct.pack(
        ">hiiiH", _SPECIAL_LINKED, len(pixel_blob), block_len, num_blk, table_refs[0]
    )

    table_raws = []
    for i, group in enumerate(table_ref_groups):
        next_table_ref = table_refs[i + 1] if i + 1 < len(table_refs) else 0
        slots = list(group) + [0] * (num_blk - len(group))  # zero-pad terminates the table
        table_raws.append(
            struct.pack(">H", next_table_ref) + b"".join(struct.pack(">H", r) for r in slots)
        )

    entries = [
        (_DFTAG_FID, fid_ref, label_raw),
        (_DFTAG_FD, fd_ref, fd_raw),
        (_DFTAG_NT, nt_ref, nt_raw),
        (_DFTAG_SDD, sdd_ref, sdd_raw),
        (_DFTAG_NDG, ndg_ref, ndg_raw),
        (_DFTAG_SD | _SPECIAL_TAG_BIT, sd_ref, special_raw),
    ]
    entries += list(zip([_DFTAG_LINKED] * len(table_refs), table_refs, table_raws))
    entries += list(zip([_DFTAG_LINKED] * len(data_refs), data_refs, data_chunks))

    dd_header_size = 6  # n_dd(int16) + next_block(int32)
    offset = 4 + dd_header_size + 12 * len(entries)  # magic + DD header + DD entries
    dd_bytes = bytearray(struct.pack(">hi", len(entries), 0))
    data_bytes = bytearray()
    for tag, ref, raw in entries:
        dd_bytes += struct.pack(">HHii", tag, ref, offset, len(raw))
        data_bytes += raw
        offset += len(raw)

    with open(dest_path, "wb") as out:
        out.write(_MAGIC)
        out.write(dd_bytes)
        out.write(data_bytes)


def _verify(source_path: str, dest_path: str, n_frames: int) -> None:
    source = Hdf4File(source_path)
    shrunk = Hdf4File(dest_path)
    if shrunk.shape != (n_frames, *source.shape[1:]):
        raise AssertionError(f"shape mismatch: {shrunk.shape} != {(n_frames, *source.shape[1:])}")
    if shrunk.dtype != source.dtype:
        raise AssertionError(f"dtype mismatch: {shrunk.dtype} != {source.dtype}")
    new_fd = _parse_fd_text(shrunk.file_description())
    if int(float(new_fd["POINTS_1"])) != n_frames:
        raise AssertionError(f"POINTS_1 wasn't updated: {new_fd['POINTS_1']!r}")
    for i in range(n_frames):
        if not np.array_equal(source.read_frame(i), shrunk.read_frame(i)):
            raise AssertionError(f"frame {i} isn't byte-identical to the source")


def shrink(source_path: str, dest_path: str, n_frames: int) -> None:
    source = Hdf4File(source_path)
    if not 1 <= n_frames <= source.shape[0]:
        raise ValueError(f"n_frames must be in [1, {source.shape[0]}], got {n_frames}")

    with open(source_path, "rb") as f:
        if f.read(4) != _MAGIC:
            raise ValueError(f"{source_path!r} is not an HDF4 file")
        dd = Hdf4File._walk_dds(f)  # pylint: disable=protected-access
        sd_ref, sdd_ref, nt_ref = Hdf4File._find_sds(f, dd)  # pylint: disable=protected-access
        nt_raw = _read_raw(f, dd, _DFTAG_NT, nt_ref)
        sdd_raw = _read_raw(f, dd, _DFTAG_SDD, sdd_ref)
        label_raw = _read_raw(f, dd, _DFTAG_FID, _annotation_ref(dd, _DFTAG_FID))
        block_len, num_blk = _special_header(f, dd, sd_ref)

    fd_raw = _new_fd_text(source.file_description(), n_frames).encode("latin-1")
    pixel_blob = b"".join(source.read_frame(i).tobytes() for i in range(n_frames))

    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    _write_hdf4(dest_path, label_raw, fd_raw, nt_raw, sdd_raw, pixel_blob, block_len, num_blk)
    _verify(source_path, dest_path, n_frames)
    print(f"{dest_path}: {n_frames} frames, byte-identical to {source_path}, verified OK")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="original, full-size SMART .hdf file")
    parser.add_argument("dest", help="output path for the shrunk .hdf file")
    parser.add_argument("n_frames", type=int, help="number of leading frames to keep")
    args = parser.parse_args(argv)
    shrink(args.source, args.dest, args.n_frames)


if __name__ == "__main__":
    main()
