# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # A Tutorial for LEEM data base classes
#
# `LEEMImg` and `LEEMStack` are the two base classes that encapsulate and parse LEEM data. Nearly
# everything else in uspy builds on them, so this notebook covers construction, metadata,
# arithmetic, slicing, and saving/loading -- the things every other notebook assumes you already
# know. It ends with a handful of worked examples.
#
# Related notebooks: `LEEM_plotting.ipynb` for real plotting (this one only uses bare
# `plt.imshow`), `LEEM_readers-and-config.ipynb` for non-Elmitec data and configuring data paths,
# `LEEM_norm-and-align.ipynb` for `.normalize()`/`.align()` in depth, `LEEM_rois.ipynb` for
# region-of-interest and intensity-profile analysis.

# %%
from uspy.leem.base import LEEMImg, LEEMStack
import uspy.leem.plotting as plot

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

TESTDATA_DIR = "../testdata/"

# %% [markdown]
# ## LEEMImg
#
# Give the constructor a file name to load an Elmitec ".dat" image. Pixel data is parsed
# immediately -- this is not lazy loading. (Lazy/out-of-core loading exists, but is opt-in; see
# the "Virtual (out-of-core) stacks" section below.)

# %%
img = LEEMImg(TESTDATA_DIR + "bremen.dat")

# %% [markdown]
# `LEEMImg` also accepts a plain 2D `np.ndarray` (for synthetic/processed data with no source
# file), a `.tif`/`.tiff` path, or another `LEEMImg`/`Image` (which copies it). Data from other
# endstations needs a `reader=` argument -- see `LEEM_readers-and-config.ipynb`.

# %%
synthetic = LEEMImg(np.random.rand(64, 64).astype("float32"))
synthetic.energy = 5.0  # metadata can be set freely on an array-backed image too
print(synthetic.image.shape, synthetic.energy, synthetic.source)

# %% [markdown]
# ## Metadata
#
# Metadata is exposed as attributes. Missing values return `np.nan` (or a sensible string
# default, e.g. `fov`). Attributes can be set too -- for example if a value is missing, or wrong,
# you can fill in the correct one. Note the `fov` example below: it changes when set.

# %%
for field in ("energy", "temperature", "pressure1", "pressure2", "fov", "exposure", "width"):
    print(f"{field}: {getattr(img, field)}")

img.fov = "10 µm"
print("fov after setting:", img.fov)

# %% [markdown]
# Internally, `img.meta` *is* the live metadata dictionary -- `img.meta["some_key"] = value`
# works, and so does the attribute form above (`img.attr = value` writes into `_meta` for any
# name that isn't a data key or class property). What you can't do is rebind the whole dict
# (`img.meta = {...}` is a silent no-op). `img.unit` is the matching dict of unit strings.

# %%
print(img.meta["temperature"])
print(img.unit["temperature"])

# %% [markdown]
# `str(img)` pretty-prints source, metadata and data shapes -- without loading pixel data, so
# it's safe to use even on a not-yet-materialized (deferred) image:

# %%
print(img)

# %% [markdown]
# ### Field reference
#
# | field | unit | meaning |
# |---|---|---|
# | `energy` | eV | kinetic energy of the imaging electrons (start voltage) |
# | `kinetic_energy` | eV | alias of `energy` |
# | `photon_energy` | eV | synchrotron photon energy; `nan` unless a beamline reader set it |
# | `binding_energy` | eV | `photon_energy - energy - energy_offset` -- read-only; adjust `energy_offset`/`photon_energy` instead |
# | `temperature` | °C | sample temperature |
# | `pressure1`, `pressure2` | Torr | the two gauge readings actually stored |
# | `pressure` | Torr | convenience property: first non-`nan` of `pressure`/`pressure1`/`pressure2`/`MCH`/`PCH` |
# | `objective` | mA | objective lens current |
# | `emission` | µA | emission current |
# | `exposure` | s | exposure time |
# | `averaging` | -- | `0` = sliding average, `1` = no averaging, otherwise the number of averaged frames |
# | `fov` | -- | a *string*: e.g. `'10µm'`, `'LEED'`, or `'Unknown FoV'` if absent -- not numeric |
# | `x_position`, `y_position` | µm | sample stage position |
# | `width`, `height` | px | image dimensions |
# | `timestamp` | -- | UNIX timestamp (no unit string is attached by the Elmitec reader) |
# | `rel_time` | s | `timestamp` minus the stack's `time_origin` (see below); `nan` on a standalone image |
# | `isotime` | -- | `timestamp` as a human-readable `"%Y-%m-%d %H:%M:%S"` string |
# | `mcp`, `dark_counts`, `warp_matrix` | -- | bookkeeping set by `.normalize()`/`.align()` |
#
# Some files also carry self-labelled "UView" fields (e.g. `'Bomb. Voltage'`) that aren't fixed
# across files -- these are only reachable via `img.meta['Bomb. Voltage']`, not attribute syntax.

# %% [markdown]
# There are also functions for prettier printing of individual fields, with field-appropriate
# formatting and units:

# %%
print(img.get_field_string("fov"))
print(img.get_field_string("objective"))
print(img.get_field_string("energy"))
print(img.get_field_string("timestamp"))  # no unit -- the Elmitec reader doesn't set one
print(img.get_field_string("rel_time"))   # nan on a standalone image, but "s" as its unit

# %% [markdown]
# ## Image data and simple operations
#
# `img.image` is a numpy array of shape `(height, width)`. For quick sanity-checking, plain
# matplotlib is fine; for labelled/annotated plots use `uspy.leem.plotting.plot_img` instead (see
# `LEEM_plotting.ipynb`) -- keep data handling and visualization separate.

# %%
print(img.image.shape)
plt.imshow(img.image, cmap="gray")
plt.show()

# %%
plt.imshow(np.log(img.image), cmap="gray")
plt.show()

# %% [markdown]
# Images support arithmetic (`+ - * /`, including in-place `+= -= *= /=`) against another image,
# a number, or an array. `.filter(method=...)` (`"gaussian"`, `"blur"`, `"median"`, `"kernel"`)
# and `.contrast(method=...)` (`"equal"`, `"clahe"`) return a processed copy. **Note:** an
# unrecognized `contrast` method doesn't raise -- it silently returns a plain copy.

# %%
img2 = LEEMImg(TESTDATA_DIR + "bremen2.dat")
diff = img - img2
print(diff.image.mean())

smoothed = img.filter(method="gaussian", sigma=2)
equalized = img.contrast(method="equal")

# %% [markdown]
# ## Saving and loading
#
# Images (and stacks) can be saved under several extensions, which trade off self-containedness,
# disk usage, and long-term archivability:
#
# | extension | contains | notes |
# |---|---|---|
# | `.pickle` | full binary data | self-contained, but tied to the Python/library versions used to write it |
# | `.pickle.bz2` | same, bz2-compressed | smaller on disk, same caveats |
# | `.json` | only *references* (source path, `mcp`, `warp_matrix`, `dark_counts`, `reader`) | tiny, but the original source file(s) must still exist and be reachable at the same (relative) path to reload |
# | `.h5` / `.hdf5` / `.nxs` | full data, NeXus-flavoured HDF5 | self-contained *and* archive-safe -- the only format intended for publication/deposition; a derived type (e.g. `IntensityLine`) reloads as its plain parent class |
# | `.tif` / `.tiff` | pixel data only | metadata is lost; not reloadable via `.load()` |

# %%
img.save("test_img.json")
img_from_json = LEEMImg.load("test_img.json")
print((img.image == img_from_json.image).all())

# %%
img.save("test_img.pickle")
img_from_pickle = LEEMImg.load("test_img.pickle")
print((img.image == img_from_pickle.image).all())

# %%
img.save("test_img.pickle.bz2")
print(Path("test_img.pickle.bz2").stat().st_size, "bytes (compressed)")
print(Path("test_img.pickle").stat().st_size, "bytes (uncompressed)")

# %%
img.save("test_img.h5")
img_from_hdf5 = LEEMImg.load("test_img.h5")
print((img.image == img_from_hdf5.image).all())

# %% [markdown]
# ## LEEMStack
#
# A `LEEMStack` holds an ordered list of `LEEMImg`s, loaded from a directory, a glob, an explicit
# list of file names, or a list of already-built `LEEMImg`s.

# %%
stack = LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu")
print(len(stack))

# %% [markdown]
# It's iterable like a Python list:

# %%
for frame in stack[:5]:
    print(frame.energy)

print(stack[-8].timestamp)

# %% [markdown]
# ### Slicing and attribute gathers
#
# `stack[i]` returns a single `LEEMImg`. `stack[a:b]` returns a new `LEEMStack` that *shares* its
# element objects with the parent -- it's a view, not a copy. A boolean mask also works, and is
# the idiomatic way to select by metadata (e.g. an energy or time window):

# %%
sub = stack[(stack.energy > 20) & (stack.energy < 25)]
print(len(sub))
print(sub.energy)

# %% [markdown]
# **Note:** an empty slice or an all-`False` mask raises `ValueError` rather than returning an
# empty stack.
#
# Any attribute shared by every image can be gathered across the whole stack as an array -- this
# is equivalent to (but much less typing than) a list comprehension:

# %%
print(stack.energy)                          # numpy array
print([frame.energy for frame in stack])     # equivalent list comprehension

# %% [markdown]
# The reverse also works: `stack.attr = value` broadcasts a scalar to every element (handy for
# fixing metadata that's missing or wrong on a whole stack at once), while assigning a sequence of
# matching length sets one value per element:

# %%
sub = stack[:4].copy()
sub.fov = "10µm"
print([frame.fov for frame in sub])

# %% [markdown]
# ### Time origin
#
# A stack (and every slice taken from it) shares one mutable `time_origin`. `rel_time` is always
# `timestamp - time_origin`, so setting `stack.time_origin` retroactively changes `rel_time` for
# every image derived from that stack. `isotime` gives a human-readable timestamp regardless of
# the origin. Appending another stack/image with `.extend()` warns if its time origin differs.

# %%
print(stack.time_origin)
print(stack.rel_time[:5])
print(stack[0].isotime)

# %% [markdown]
# ### average and filter on stacks
#
# `.average(n)` collapses every `n` consecutive frames into their mean (the stack length must be
# divisible by `n`); `.filter(...)` applies `Image.filter` to every element. Both preserve
# `virtual` if the stack was virtual.

# %%
averaged = stack[:4].average(2)
print(len(averaged))

# %% [markdown]
# ## Virtual (out-of-core) stacks
#
# For a stack too large to fit in RAM at once, use `LEEMStack(path, virtual=True)`. Elements are
# ordinary, stable `LEEMImg`s, but their pixel data is read from disk on first access instead of
# at construction, and is **never cached** -- reading `img.image` twice reads the file twice
# (metadata *is* cached after the first read). Every stack operation (indexing, arithmetic,
# `.normalize()`, `.align()`, `.filter()`, gathers) works the same as on a resident stack and
# stays out-of-core where possible.

# %%
virt_stack = LEEMStack(TESTDATA_DIR + "test_stack_IV_g-Cu", virtual=True)
print(virt_stack.virtual, len(virt_stack))

# %% [markdown]
# `stack.virtual` is *derived* from its elements, not a stored flag, so it can't misreport.
# Convert with `materialize()`/`defer()` (in place, return `self`) or `materialized()`/
# `deferred()` (return a *new* stack, leaving the original untouched) -- the same four verbs
# `LEEMImg` already has. `small_stack = virt_stack[:5].materialized()` rebinds the new stack's
# elements to fresh, resident copies, leaving the parent stack (and any handle to its old
# elements) untouched. A **materialized substack no longer stays connected to its parent** once
# converted this way:

# %%
small_stack = virt_stack[:5].materialized()   # a new, resident stack
print(small_stack.virtual, len(small_stack))   # materialized
print(virt_stack.virtual, len(virt_stack))     # parent is still virtual

# %%
for frame in virt_stack[:5]:
    print(frame.is_virtual)
for frame in small_stack:
    print(frame.is_virtual)

# %% [markdown]
# **Note:** re-deferring a stack (`stack.defer()`/`.deferred()`) always re-derives data from the
# source file from scratch -- any in-memory edit or composed processing chain (e.g. `.normalize()`
# or `.align()`) that was never `.save()`d is discarded, not preserved. Gathering a data key
# (`virt_stack.image`) across a virtual stack materializes every element at once and raises a
# `UserWarning`, since that defeats the point of staying virtual. A virtual stack also skips the
# pairwise shape-compatibility check at construction (to stay zero-I/O); call
# `stack.check_compatible()` to fail early instead of at first bulk use.
#
# (`stack.virtual = False`/`True` still works as a deprecated alias for `materialize()`/`defer()`,
# with a `DeprecationWarning` pointing at the verbs.)

# %% [markdown]
# ## Stack save/load
#
# Stacks save and load the same way images do (`.json`, `.pickle`, `.pickle.bz2`, or the
# self-contained `.h5`/`.hdf5`/`.nxs` archive format described above):

# %%
stack.save("test_stack.json")
stack_from_json = LEEMStack.load("test_stack.json")
print((stack.energy == stack_from_json.energy).all())

stack.save("test_stack.h5")
stack_from_hdf5 = LEEMStack.load("test_stack.h5")
print((stack.energy == stack_from_hdf5.energy).all())

# %% [markdown]
# ## Worked examples
#
# A few things a LEEM user actually does with these classes.

# %% [markdown]
# **Pick the frame nearest a given energy:**

# %%
target_energy = 22.0
idx = int(np.argmin(np.abs(stack.energy - target_energy)))
print(idx, stack[idx].energy)

# %% [markdown]
# **Difference image between two energies:**

# %%
diff = stack[idx] - stack[0]
plot.plot_img(diff)

# %% [markdown]
# **Select a time or energy window:**

# %%
window = stack[(stack.rel_time > 10) & (stack.rel_time < 40)]
print(len(window))

# %% [markdown]
# **Export a metadata table** (e.g. for external plotting/fitting):

# %%
table = np.column_stack([stack.rel_time, stack.energy, stack.pressure1])
print(table.shape)
np.savetxt("test_stack_meta.csv", table, header="rel_time,energy,pressure1", delimiter=",")

# %% [markdown]
# **Memory-safe pass over a stack too large for RAM:** open it `virtual=True` and iterate exactly
# once, since deferred pixel data is re-read from disk on every access -- don't index the same
# element twice if that matters.

# %%
means = [frame.image.mean() for frame in virt_stack]
print(len(means), means[0])

# %% [markdown]
# **Gotcha: `stack_a + stack_b` concatenates, it does not add elementwise.** `-`, `*`, `/` (and
# subtracting/dividing by a single image or number) *are* elementwise. This asymmetry is easy to
# trip over:

# %%
a, b = stack[:3].copy(), stack[3:6]
print(len(a + b))          # 6 -- concatenation
print(len(a - stack[0]))   # 3 -- elementwise, same length as `a`

# %% [markdown]
# **Teaser: MCP flat-field and dark-count correction.** `LEEMImg.normalize()` divides out the MCP
# response and subtracts dark counts; see `LEEM_norm-and-align.ipynb` for the full treatment
# (including `LEEMStack.normalize()` and `.align()`). Note this raises a `RuntimeWarning` for any
# zero-valued MCP pixels -- uspy maps the resulting `inf`/`nan` to `0`, but the warning is
# expected here, not a bug.

# %%
mcp = LEEMImg(TESTDATA_DIR + "channelplate.dat")
dark_counts = LEEMImg(TESTDATA_DIR + "dark_counts_bremen.dat")
normalized = img.normalize(mcp=mcp, dark_counts=dark_counts)
print(normalized.mcp)

# %% [markdown]
# ## Where to go next
#
# - `LEEM_plotting.ipynb` -- annotated image/metadata plots and video export, instead of bare `plt.imshow`.
# - `LEEM_norm-and-align.ipynb` -- `LEEMStack.normalize()` and `.align()` (image registration) in depth.
# - `LEEM_rois.ipynb` -- regions of interest and intensity profiles.
# - `LEEM_readers-and-config.ipynb` -- non-Elmitec data (`reader=`) and configuring data paths (`uspy.toml`, `uspy.DATADIR`/`LEEMDIR`).
