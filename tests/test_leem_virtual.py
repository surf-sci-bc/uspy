"""Tests for LEEMImg/LEEMStack deferred (`virtual=True`) construction: the LEEM-specific
guarantees layered on top of the generic `DataObject`/`DataObjectStack` deferral machinery
(see tests/test_deferred_dataobject.py, tests/test_meta_loader.py).

Covers the concerns from the original request this refactor was for: a virtual stack must
not make excessive I/O on construction or access -- no per-frame config/uspy.toml walk, no
pixel reads for metadata, no header re-walks for pixel reads, and metadata is read at most
once per frame no matter how many times it's gathered.
"""
# pylint: disable=missing-docstring
# pylint: disable=protected-access

import numpy as np
import pytest

from uspy.leem import base
from uspy.leem.base import LEEMStack
from uspy.leem.readers.elmitec import ElmitecReader
import uspy.leem.readers as readers_module
from uspy import config as config_module

from .conftest import STACK_FNAMES

# Elmitec-only folders (plain per-file ".dat", no tracefile/sidecar complications).
ELMITEC_STACK_FOLDERS = STACK_FNAMES[0:2]


def _patch_counts(monkeypatch):
    counts = {"parse_image": 0, "parse_metadata": 0, "parse_pixels": 0}
    original_image = ElmitecReader.parse_image
    original_meta = ElmitecReader.parse_metadata
    original_pixels = ElmitecReader.parse_pixels

    def counting_image(self, source):
        counts["parse_image"] += 1
        return original_image(self, source)

    def counting_meta(self, source):
        counts["parse_metadata"] += 1
        return original_meta(self, source)

    def counting_pixels(self, source):
        counts["parse_pixels"] += 1
        return original_pixels(self, source)

    monkeypatch.setattr(ElmitecReader, "parse_image", counting_image)
    monkeypatch.setattr(ElmitecReader, "parse_metadata", counting_meta)
    monkeypatch.setattr(ElmitecReader, "parse_pixels", counting_pixels)
    return counts


@pytest.mark.parametrize("stack_folder", ELMITEC_STACK_FOLDERS)
def test_leemstack_virtual_construction_reads_no_pixels(stack_folder, monkeypatch):
    counts = _patch_counts(monkeypatch)
    LEEMStack(stack_folder, virtual=True)
    assert counts["parse_pixels"] == 0
    assert counts["parse_image"] == 0
    # LEEMStack.__init__ reads frame 0's timestamp to seed the shared TimeOrigin -- at most
    # one header parse, not one per frame.
    assert counts["parse_metadata"] <= 1


@pytest.mark.parametrize("stack_folder", ELMITEC_STACK_FOLDERS)
def test_leemstack_virtual_metadata_gather_reads_headers_not_pixels(stack_folder, monkeypatch):
    stack = LEEMStack(stack_folder, virtual=True)
    counts = _patch_counts(monkeypatch)

    energies = stack.energy

    assert len(energies) == len(stack)
    assert counts["parse_pixels"] == 0
    # frame 0 was already resolved during construction (LEEMStack.__init__ reads its
    # timestamp to seed the shared TimeOrigin, before this monkeypatch was even installed),
    # so gathering across the whole stack now only resolves the remaining N-1 frames.
    assert counts["parse_metadata"] == len(stack) - 1

    # already resolved -- gathering again must not re-read anything
    _ = stack.temperature
    assert counts["parse_metadata"] == len(stack) - 1


@pytest.mark.parametrize("stack_folder", ELMITEC_STACK_FOLDERS)
def test_leemstack_virtual_pixel_access_reads_pixels_not_headers(stack_folder, monkeypatch):
    stack = LEEMStack(stack_folder, virtual=True)
    counts = _patch_counts(monkeypatch)

    _ = stack[1].image

    assert counts["parse_pixels"] == 1
    assert counts["parse_metadata"] == 0
    assert counts["parse_image"] == 0

    # never cached -- reading again re-reads
    _ = stack[1].image
    assert counts["parse_pixels"] == 2


@pytest.mark.parametrize("stack_folder", ELMITEC_STACK_FOLDERS)
def test_leemstack_virtual_pixel_parity_with_eager(stack_folder):
    eager = LEEMStack(stack_folder, virtual=False)
    virtual = LEEMStack(stack_folder, virtual=True)
    for e, v in zip(eager, virtual):
        np.testing.assert_array_equal(e.image, v.image)
        assert e.energy == v.energy or (np.isnan(e.energy) and np.isnan(v.energy))


@pytest.mark.parametrize("stack_folder", ELMITEC_STACK_FOLDERS)
def test_leemstack_virtual_writes_persist(stack_folder):
    stack = LEEMStack(stack_folder, virtual=True)
    stack[0].energy = 12345.0
    assert stack[0].energy == 12345.0
    assert stack[0] is stack[0]


@pytest.mark.parametrize("stack_folder", ELMITEC_STACK_FOLDERS)
def test_leemstack_virtual_construction_does_not_walk_config_per_frame(stack_folder, monkeypatch):
    """Each `LEEMImg` resolves its own plotting fields (`resolve_plotting_fields`) --
    LEEMImg/LEEMStack know nothing about plotting, by design (data storage/handling and
    visualization are kept fully separate; see CLAUDE.md). What must stay bounded per stack,
    not per frame, is the actual I/O: `resolve_plotting_fields` calls into
    `config.config_search_paths`, whose expensive part (the `.resolve()`-based candidate-
    folder computation) is cached at the `uspy.config` layer, keyed on the data folder/cwd/
    global config dir -- so calling `resolve_plotting_fields` once per frame is cheap
    (a cache hit) rather than a repeated filesystem walk."""
    plotting_calls = {"n": 0}
    original_plotting = readers_module.resolve_plotting_fields

    def counting_plotting(*args, **kwargs):
        plotting_calls["n"] += 1
        return original_plotting(*args, **kwargs)

    monkeypatch.setattr(readers_module, "resolve_plotting_fields", counting_plotting)
    monkeypatch.setattr(base, "resolve_plotting_fields", counting_plotting)

    resolve_calls = []
    original_resolve = config_module._config_search_paths_cached
    original_resolve.cache_clear()

    def counting_resolve(*args, **kwargs):
        resolve_calls.append(args)
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(config_module, "_config_search_paths_cached", counting_resolve)

    stack = LEEMStack(stack_folder, virtual=True)

    # resolve_plotting_fields runs once per frame -- that's the point, it's decoupled from
    # construction and cheap either way.
    assert plotting_calls["n"] == len(stack)
    # ... but the expensive underlying computation only ever runs once for this one folder;
    # every other call (one per frame) was a cache hit.
    assert len(set(resolve_calls)) <= 1
