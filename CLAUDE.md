# CLAUDE.md

## What this is

`uspy` (µSPY) is a Python library for analyzing surface-science microscopy/spectroscopy data: LEEM
(Low-Energy Electron Microscopy) images and image stacks and related plotting/processing utilities. It's a research-lab codebase, not a package with a public API contract — expect some dead/commented-out code and in-progress modules. `uspy/legacy/` (and its mirror `tests/legacy/`) holds unmaintained, non-functional code (old XPS/STM support, the pre-`dataobject` LEEM API, deprecated drift-normalization/RSM helpers) that was quarantined out of the active tree on 2026-07-28 — **never read, grep, edit, or otherwise consider anything under `uspy/legacy/` or `tests/legacy/` for any task.**

uspy is intended for use in Jupyter notebooks. Users call the methods directly to analyze data. A function that is never called in the codebase can still be important for analysis.

Priorities:
0. Be Brief.
1. Never modify source data.
2. The code/data processing must always behave as expected by the user. 
3. Never silently change processing or visualization of data. If you do raise a Warning.
4. Keep the code focused. Dont mix data storage classes and visualization.
5. Keep code easy to maintain. Reuse the codebase as much as possible. If you have written 200 Lines that could be 50, write 50.
6. Write tests first then implement. Never cherry-pick inputs for passing tests.
7. Write focused numpydoc-style docstrings instead of essays, stories or anecdotes.

## Setup

Editable install (`pip install -e .`); upgrade pip first — avoids broken opencv installs.
Versioning is via `setuptools_scm` from git tags, see `uspy/version.py`.

## Tests

Tests use real binary fixture files from the `testdata` git submodule — run `git submodule init &&
git submodule update` before testing. By default `pytest` skips tests marked "slow"; use
`--runslow` to include them.

## Architecture

### Core abstraction: `uspy/dataobject/`

Nearly everything is built on `Loadable` (pickle/json_tricks dump/load), `DataObject` (single image/
line: `_data`/`_meta`/`_units` dicts, populated from `parse(source)`), and `DataObjectStack` (ordered
collection of one `DataObject` type, with `virtual=True` mode — see below). `__getattr__`/
`__setattr__` are overridden so `obj.temperature` reads `_meta`, `obj.image` reads `_data`, etc. —
**this dynamic attribute dispatch is the trickiest part of the codebase**; when adding fields, add
them to `_meta_defaults`/`_data_keys` rather than assuming a plain instance attribute is picked up.

Domain-specific subpackages build on these bases rather than replacing them. Note the layout is
inverted from what the directory names suggest: `uspy/plotting.py` holds the bulk of matplotlib-based
visualization code, while `uspy/leem/plotting.py` is a thin wrapper.

### Memory: large stacks and `virtual=True`

The default `ImageStack`/`LEEMStack` holds every frame in memory as a list of independent objects.
Out-of-place operations (arithmetic without `+=`/`-=`, `.copy()`, `.align()` with the default
`inplace=False`) deep-copy the whole stack, so peak memory briefly doubles; some `processing.py`
helpers additionally materialize a full `(N, H, W)` array via `stack.image`
(`DataObjectStack.__getattr__` rebuilds this on every access, with a `UserWarning` on a virtual
stack since it materializes every element at once — never cache it yourself without being sure
it's still fresh).

For a stack near or over available RAM, use `ImageStack(path, virtual=True)` /
`LEEMStack(path, virtual=True)` (or convert an existing stack — see below) instead. This is
not a separate backend: elements are ordinary, stable `DataObject`s whose data (`_data`, see
`uspy/dataobject/deferred.py`'s `DeferredArray`) is read from `source` on first access instead of
at construction, and is **never cached** — reading `img.image` twice reads the file twice, so
iterate once rather than indexing repeatedly if that matters. Metadata *is* resolved once per
element and cached permanently, so gathers like `stack.energy` cost one header read per frame, not
one per field per frame. Every stack operation (indexing, arithmetic, `.normalize()`, `.average()`,
`.filter()`, `.align()`, `uspy.plotting.make_video()`, `.save()`) works the same as on a resident
stack and stays out-of-core where the operation allows it: arithmetic, `.normalize()`, `.align()`
(`LEEMImg.warp`), `.filter()`, and `ROI.apply()` all compose onto the existing `DeferredArray`
(via `DeferredArray.map`, with a module-level function bound through `functools.partial` for
anything parameterized — a lambda/closure isn't picklable and would defeat a pickled virtual
stack's whole point) rather than loading and materializing; `.average()`/`.filter()` then pass
`virtual=self.virtual` through so the *stack* built from those composed elements stays virtual
too. A virtual stack skips the pairwise element-compatibility check at construction to stay
zero-I/O; call `stack.check_compatible()` to fail early instead of at first bulk use.

**Converting between resident and deferred** uses the same four-verb vocabulary as
`DataObject` (`materialize`/`defer`/`materialized`/`deferred`, see `base.py:237-278`), now
mirrored on `DataObjectStack` itself: `stack.materialize()`/`stack.defer()` convert in place and
return `self` (chainable); `stack.materialized()`/`stack.deferred()` return a *new* stack,
leaving `stack` untouched — the non-mutating pair is a no-op (`is stack`) when already in the
target state. `stack.virtual` stays a read-only, *derived* report (`all(element.is_virtual for
element in stack.elements)`, not a stored flag — it can't misreport); assigning to it
(`stack.virtual = False`) still works as a deprecated alias for the verbs, but warns.

Slicing (`stack[a:b]`) is a view: the slice shares the same element objects as the parent, same
as it always has (so does a `LEEMStack`'s shared `TimeOrigin`). Converting either the parent or a
slice never *mutates* those shared elements in place — it rebinds `self._elements` to freshly
materialized/deferred copies (`DataObject.materialized()`/`.deferred()`), leaving whichever stack
you didn't touch (and its elements) untouched. So `sub = stack[:5]; sub.materialize()` no longer
silently pins the first five frames of `stack` in memory behind its back, but it does mean a
reference held from *before* the transition (`img = stack[0]` taken before `stack.materialize()`)
is now a detached, stale copy — re-fetch `stack[0]` after a transition rather than reusing an old
handle.

Assigning a *resident* array onto a still-deferred element's data key (`img.image = some_array`)
still permanently pins that one element in memory (`DataObject.__setattr__` warns, `UserWarning`,
whenever this happens) — but since `virtual` is derived, any stack containing that element now
correctly flips to `virtual=False` as soon as it's checked, rather than silently disagreeing with
its own elements. Treat the warning as a sign an operation needs the compose-via-`.map()`
treatment above instead of a materializing read/modify/write-back.

Re-deferring (`stack.defer()`/`.deferred()`, or a single element's `.defer()`/`.deferred()`)
always re-derives data from `source` from scratch — any in-memory edit *or composed processing
chain* not yet reflected at `source` (a `.normalize()`/`.align()`/`.filter()` result that was
never `.save()`d, for instance) is discarded, not preserved. This is a deliberate, documented
contract, not a bug: don't round-trip `materialize()` → `defer()` expecting composed results to
survive.

### Data flow pattern

Construction always goes through `parse(source)` → a flat dict → sorted into `_data`/`_meta`/`_units`.
Derived types like `IntensityLine`/`Waterfall`/`ProfileLine` take *tuples of upstream objects* (stack,
ROI, axis name, etc.) as `source`, so their `parse()` does extraction/aggregation, not file I/O —
check `parse()` before assuming a class only reads from disk.

### Archival export: `uspy/hdf5.py`

See `.claude/rules/hdf5.md` for the archive format contracts (derived-type export behavior,
uspy-agnostic file layout) — loaded automatically when working on that file.

### Data location convention

`uspy.DATADIR`/`LEEMDIR` default to `~/data/...` — real experimental data lives under the user's
home directory, separate from the `testdata` submodule used for tests. Both are plain trailing-slash
`str`s that notebooks rely on (`LEEMDIR + "file.dat"`) — don't change the type.

### Notebooks

`examples/*.ipynb` are outdated usage demos that double as informal documentation. `dev/` contains
scratch/experimental notebooks and scripts — not maintained code.
