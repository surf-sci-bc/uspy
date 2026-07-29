# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`uspy` (µSPY) is a Python library for analyzing surface-science microscopy/spectroscopy data: LEEM
(Low-Energy Electron Microscopy) images and image stacks and related plotting/processing utilities. It's a research-lab codebase, not a package with a public API contract — expect some dead/commented-out code and in-progress modules. `uspy/legacy/` (and its mirror `tests/legacy/`) holds unmaintained, non-functional code (old XPS/STM support, the pre-`dataobject` LEEM API, deprecated drift-normalization/RSM helpers) that was quarantined out of the active tree on 2026-07-28 — **never read, grep, edit, or otherwise consider anything under `uspy/legacy/` or `tests/legacy/` for any task.**

uspy is intended for use in Jupyter notebooks. Users call the methods directly to analyze data. A function that is never called in the codebase can still be important for analysis.

Priorities:
1. Never modify source data.
2. The code/data processing must always behave as expected by the user. 
3. Never silently change processing or visualization of data. If you do raise a Warning. 
4. Never cherry-pick inputs for passing tests. Write tests against possible unexpected behavior and look into edge cases. 

## Setup

```sh
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python3 -m pip install --upgrade pip   # avoids broken opencv installs
python3 -m pip install -e .
```

`uspy` is installed in editable mode; `import uspy` should then work from anywhere in the venv.

Versioning is handled by `setuptools_scm` from git tags (`MAJOR.MINOR.PATCH`, see `uspy/version.py`).
There's no separate build step beyond the editable install.

## Tests

Tests live in `tests/` and use real binary fixture files from the `testdata` git submodule (a
separate repo, since binary data doesn't belong in the main history). If you need to run tests:

```sh
git submodule init
git submodule update
```

Then, from within the venv:

```sh
pytest                        # skips tests marked "slow"
pytest --runslow               # include slow tests
pytest tests/test_leembase.py  # single file
pytest tests/test_leembase.py::test_name  # single test
```

Test fixtures (fixture image/stack filenames, `pytest_addoption` for `--runslow`) are centralized in
`tests/conftest.py`. Test modules roughly mirror source modules (`test_leembase.py` ↔
`uspy/leem/base.py`, etc.); `tests/conftest.py` sets `collect_ignore = ["legacy"]` since
`tests/legacy/` targets `uspy/legacy/`.

Linting config exists at `.pylintrc` (pylint) but there's no enforced CI for it currently.

## Architecture

### Core abstraction: `uspy/dataobject/`

Nearly everything in this codebase is built on a small set of base classes defined here:

- **`Loadable`**: mixin providing `dump`/`dumps`/`save`/`load` — objects can be pickled (optionally
  bz2-compressed, `.pickle`/`.pickle.bz2`) or serialized with `json_tricks` (`.json`).
- **`DataObject`**: base class for a single piece of data (an image, a line/spectrum). Holds three
  dicts under the hood: `_data` (numeric arrays), `_meta` (metadata like temperature, energy,
  timestamps), `_units` (unit strings per meta/data key). Subclasses declare `_data_keys`,
  `_meta_defaults`, `_unit_defaults` as class attributes; `parse(source)` (implemented per subclass)
  turns a raw source (file path, ndarray, tuple of upstream objects) into a dict that populates those
  three stores. `__getattr__`/`__setattr__` are overridden so `obj.temperature` transparently reads
  from `_meta`, `obj.image`/`obj.x`/`obj.y` from `_data`, etc. — **this dynamic attribute dispatch is
  the trickiest part of the codebase**; when adding fields, add them to `_meta_defaults`/`_data_keys`
  rather than assuming a plain instance attribute will be picked up.
- **`DataObjectStack`**: an ordered collection of `DataObject`s of one `_type` (e.g. `ImageStack` of
  `Image`s). Supports slicing (`__getitem__`/`__setitem__` return more stacks or single elements),
  arithmetic (`+`, `-`, `*`, `/` broadcast elementwise across the stack), and a **virtual mode**
  (`virtual=True`): elements are stored as un-loaded sources (file paths) and only constructed into
  full objects on access, via `_split_source`/`_construct`/`_deconstruct`/`_single_construct`. This
  matters for large stacks of images that don't all fit in memory.
- Concrete `DataObject` types: **`Image`** (2D array, arithmetic ops, `.filter()`, `.contrast()`),
  **`ImageStack`** (stack of `Image`, `.average()`), **`Line`** (x/y data, `.interpolate()`,
  `.integral()`, `.derivative()`, `.smooth()`, CSV/TSF-file save/load), plus derived line types
  (`IntensityLine` — mean intensity inside a ROI per stack image; `StitchedLine` — stitches several
  `IntensityLine`s together with overlap-based rescaling; `ProfileLine`, `Waterfall`).

Domain-specific subpackages build on these bases rather than replacing them:

- **`uspy/config.py`**: shared `uspy.toml` machinery — locating config files (data folder → parent
  → cwd → global `$XDG_CONFIG_HOME/uspy` dir), parsing them (`tomllib`/`tomli`, cached on
  path+mtime), and resolving a `[uspy.paths]` table (`datadir`, `leemdir`, and arbitrary
  site-specific keys via `get_path(name)`) with `~`/env-var expansion and a session-override API
  (`set_datadir`/`set_leemdir`/`set_path`/`reset_paths`). Backs `uspy.DATADIR`/`LEEMDIR` (see "Data
  location convention" below) and is reused by `uspy/leem/readers/` for `[uspy] reader`/
  `[uspy.options]`/`[uspy.plotting]` resolution.
- **`uspy/leem/base.py`**: `LEEMImg(Image)` and `LEEMStack(ImageStack)` — the Elmitec `.dat` binary
  format parser (`parse_dat`) lives at the bottom of this file. `LEEMStack` loads a directory/glob of
  `.dat` files, tracks a `TimeOrigin` shared across images, and provides `.align()` for image
  registration (ECC-based via `do_ecc_align`, with outlier sanity-checking/spline interpolation of
  shift values) and `.normalize()` (dark-count and MCP flat-field correction). `IVCurve(StitchedLine)`
  stitches I(V) curves from multiple stacks/ROIs. `uspy/leem/processing.py`, `cluster.py`,
  `elettra.py`: further LEEM processing (clustering analysis, beamline-specific parsing).
  `uspy/leem/plotting.py` (thin) plus `uspy/plotting.py` (the bulk of plotting code) hold
  matplotlib-based visualization: image/movie plotting, metadata plots, video export (`make_video`,
  uses CRF/level video-encoding options).
- **`uspy/casaxps.py`**: reads CasaXPS `.tsf` files for XPEEM/analyzer data (including analyzer
  dispersion correction).
- **`uspy/roi.py`**: `ROI` (boolean-mask regions on images: circle/ellipse/square/rectangle/point/
  polygon, built from geometric params or an explicit array) and `Profile`/line-profile classes,
  used by `IntensityLine`, `Waterfall`, `.align()` masking, etc. `StyledObject` is the shared base for
  matplotlib style dicts (auto-cycles through the color cycle when no color is given).

### Memory: large stacks and the lazy backend

The default `ImageStack`/`LEEMStack` holds every frame in memory as a list of independent objects.
Out-of-place operations (arithmetic without `+=`/`-=`, `.copy()`, `.align()` with the default
`inplace=False`) deep-copy the whole stack, so peak memory briefly doubles; some `processing.py`
helpers additionally materialize a full `(N, H, W)` array via `stack.image`
(`DataObjectStack.__getattr__` rebuilds this on every access — never cache it yourself without
being sure it's still fresh).

For a stack near or over available RAM, use the out-of-core backend instead:
`ImageStack(path, lazy=True)` / `LEEMStack(path, lazy=True)`, or `.to_lazy()` on an existing stack —
see `uspy/dataobject/lazy.py` (`LazyImageStack`/`LazyLEEMStack`). It's backed by a chunked
`dask.array` and keeps loading, slicing, `+ - * /`, `.average()`, and `.normalize()` streamed
instead of materializing every frame at once; call `.materialize()` to get back a regular in-memory
stack once it's small enough (e.g. after slicing). `.align()` and anything touching `stack.image`
still need the eager backend.

### Data flow pattern

The typical construction path across all subpackages is: raw file(s) or an upstream `DataObject` →
`parse()` returns a flat dict → base `__init__` sorts dict keys into `_data`/`_meta`/`_units` by
name convention (keys ending in `_unit` become units; keys in `_data_keys` become data; everything
else becomes meta). Derived types like `IntensityLine`/`Waterfall`/`ProfileLine` take *tuples of
upstream objects* (stack, ROI, axis name, etc.) as their `source`, so `parse()` there is really doing
extraction/aggregation rather than file I/O — check `parse()` before assuming a class only reads
from disk.

### Data location convention

`uspy/__init__.py` exposes `DATADIR`/`LEEMDIR`, defaulting to `~/data/...` — real experimental
data is conventionally kept under the user's home directory, separate from the `testdata`
submodule used for automated tests. They're resolved lazily on every access (module `__getattr__`,
not frozen at import time) from a `[uspy.paths]` table in a `uspy.toml` (search order: current
working directory, then `$XDG_CONFIG_HOME/uspy`/`~/.config/uspy`), or from a session override —
see `uspy/config.py`, which also holds the `uspy.toml` search/parse machinery shared with
`uspy/leem/readers/`'s reader-selection config. This makes the data root configurable per
site/project instead of hardcoded, while keeping the historical `~/data/...` default and the
trailing-slash `str` type notebooks rely on (`LEEMDIR + "file.dat"`).

### Notebooks

`examples/*.ipynb` are outdated usage demos (one per topic: base usage, dose calculation, cluster analysis,
norm/align, plotting, ROIs, video export) and double as informal documentation for how the classes
above are meant to be used together. `dev/` contains scratch/experimental notebooks and scripts, not maintained
code.
