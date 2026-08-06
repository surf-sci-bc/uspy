---
paths: uspy/hdf5.py
---

# Archival export: `uspy/hdf5.py`

`obj.save("deposit.h5")` / `.hdf5` / `.nxs` (any `DataObject`/`DataObjectStack`, via the
`Loadable.save`/`load` dispatch in `uspy/serialize.py`) writes/reads a NeXus-flavoured HDF5 file —
the only *self-contained and archivable* format uspy has (`.pickle` is self-contained but
Python-version-fragile; `.json`/`.tif`/`.zarr` are either lossy or re-read from the original
source on load). See the `uspy/hdf5.py` module docstring for the full file layout and
type-coercion rules; the two things worth knowing before touching it:

- **Derived types export as their parent class.** `IntensityLine`/`StitchedLine`/`ProfileLine`/
  `IVCurve` all archive as a plain `Line`; `Waterfall` archives as a plain `Image`; the lazy
  stacks archive as their eager equivalent. An archive holds *data*, not the analysis object
  graph — an upstream `ImageStack`/`ROI`/`Profile` reference can't be represented and is simply
  not written (a `DroppedFieldWarning` at `save()` time says so).
- **The archive is deliberately uspy-agnostic**, not just uspy-loadable: two content groups
  (`/entry/data`, `/entry/metadata`) plus standard NeXus scaffolding, and exactly three
  `uspy_*` attributes in the whole file. There's no per-field type tag — the Python type of a
  metadata value is inferred from its HDF5 dtype on read, and even the one class-identity
  attribute (`uspy_class`) is optional (missing/unrecognized falls back to structural
  inference: 2-D signal → `Image`, 3-D → `ImageStack`, `y`-over-`x` → `Line`). This is a design
  choice, not an accident — resist the urge to add bookkeeping attributes/groups back in.
