# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LEEM - Readers and configuration
#
# For the basics of loading images and stacks, see `LEEM_base.ipynb` first. This notebook is about
# data that isn't Elmitec ".dat" (e.g. from a synchrotron beamline), and about not having to repeat
# the same paths/settings in every notebook you open.

# %%
import uspy
import uspy.leem.base as leem
import uspy.leem.plotting as plot
from uspy.leem.readers import set_reader, reset_reader

LEEMDIR = "../testdata/"

# %% [markdown]
# ## Loading data from another endstation
#
# By default, `LEEMImg`/`LEEMStack` expect Elmitec UKSOFT2001 ".dat" files. For data from a
# different endstation, tell it which reader to use with the `reader` argument:

# %%
img = leem.LEEMImg(
    LEEMDIR + "elettra/2024_09_26_004_XPS/2024_09_26_004_000.dat", reader="elettra"
)
print(img.energy)
print(img.photon_energy)

# %% [markdown]
# The built-in reader names are `"elmitec"` (the default), `"elettra"` (Nanospectroscopy beamline,
# Elettra) and `"smart"` (SMART microscope, BESSY). The same `reader` argument works on `LEEMStack`:

# %%
stack = leem.LEEMStack(LEEMDIR + "elettra/2024_09_26_004_XPS", reader="elettra")
print(len(stack))
print(stack.photon_energy)

# %% [markdown]
# **uspy does not guess the format from the file content.** Elettra files also end in ".dat", so
# without `reader="elettra"` they would be read as Elmitec data and the metadata would come out
# wrong, silently.

# %% [markdown]
# ## Setting the reader once for a session
#
# If you're working with one endstation's data for a whole notebook, you don't have to pass
# `reader=` every time -- set it once instead:

# %%
set_reader("smart")
img_smart = leem.LEEMImg(LEEMDIR + "bessy/0409a001.hdf")
print(img_smart.energy)
print(img_smart.photon_energy)

# %% [markdown]
# Every load for the rest of the session now uses this reader. Clear it again with:

# %%
reset_reader()

# %% [markdown]
# ## Making it permanent with `uspy.toml`
#
# For data you'll come back to, the most convenient option is a small text file named `uspy.toml`
# placed in the folder with your data:
#
# ```toml
# [uspy]
# reader = "elettra"
# ```
#
# With that file in place, `leem.LEEMImg(LEEMDIR + "elettra/2024_09_26_004_XPS/...")` picks the
# right reader by itself -- no `reader=` argument, no `set_reader` needed.
#
# Ready-to-copy files for each supported endstation are in `examples/uspy_toml/` -- just copy the
# one for your setup next to your data (named exactly `uspy.toml`).
#
# uspy looks for this file in a few places, most specific first: the folder with your data, that
# folder's parent, the folder you started the notebook/analysis from, and finally
# `~/.config/uspy/uspy.toml` -- handy on a machine that only ever produces one file format, so you
# never need a per-folder file at all.
#
# Some readers accept extra settings via `[uspy.options]`, e.g. a manual energy calibration offset:
#
# ```toml
# [uspy.options]
# energy_offset = 0.0
# ```

# %% [markdown]
# ## Nicer default fields for your endstation
#
# `plot_img`/`plot_meta`/`print_meta` show a handful of metadata fields by default (pressure1,
# pressure2, objective, ...), which are Elmitec-specific and don't exist on SMART data at all --
# calling `plot.print_meta(img_smart)` with no arguments would raise an `AttributeError`. Pick
# fields that actually exist for this endstation instead:

# %%
plot.print_meta(img_smart, fields=("energy", "photon_energy", "exposure", "scan_mode"))

# %% [markdown]
# To make that the default for a whole folder instead of repeating `fields=(...)` everywhere, add a
# `[uspy.plotting]` table to that folder's `uspy.toml`:
#
# ```toml
# [uspy.plotting]
# default_fields = ["temperature", "photon_energy", "energy", "objective"]  # plot_img/make_video
# meta_fields = ["photon_energy"]                                          # plot_meta
# print_fields = ["energy", "photon_energy", "binding_energy", "temperature"]  # print_meta
# ```
#
# The baseline `uspy.toml` files in `examples/uspy_toml/` already include sensible plotting fields
# for each endstation, so copying one of those usually covers this too.

# %% [markdown]
# ## Default mask: `[uspy.mask]`
#
# Methods that should only see real data -- `LEEMStack.align()`, `plot_img()`, `make_video()` --
# accept `mask=True` to use the instrument's default mask (e.g. a channelplate aperture that would
# otherwise throw off drift correction). Every built-in reader defaults to a
# `x0="50%", y0="50%", radius="45%"` circle:

# %%
print(img.default_mask)

# %% [markdown]
# Override it per folder/instrument with a `[uspy.mask]` table in `uspy.toml`:
#
# ```toml
# [uspy.mask]
# shape  = "circle"   # circle | ellipse | square | rectangle | polygon | point | "none"
# x0     = "50%"      # percent strings scale with frame width  (x0, width, radius)
# y0     = "50%"      # percent strings scale with frame height (y0, height)
# radius = "30%"      # plain numbers are absolute pixels; may be mixed with percent strings
# ```
#
# `shape = "none"` disables the default mask for that instrument. A session override,
# mirroring `set_reader`, works the same way:
#
# ```python
# from uspy.leem.readers import set_default_mask, reset_default_mask
#
# set_default_mask(False)   # disable for the rest of the session
# reset_default_mask()      # clear the override again
# ```
#
# `mask=True` is never applied automatically -- `align()`/`plot_img()`/`make_video()` all default
# to no masking, so existing plots/videos/alignments don't change unless you opt in.

# %% [markdown]
# ## Where your data lives: `uspy.DATADIR`/`LEEMDIR`
#
# Instead of hardcoding paths, notebooks typically build them from `uspy.LEEMDIR`:

# %%
print(uspy.DATADIR)
print(uspy.LEEMDIR)

# %% [markdown]
# These default to `~/data/` and `~/data/LEEM/`. For a one-off session (e.g. working from a mounted
# network drive), you can override them without writing a config file:

# %%
from uspy.config import set_datadir, reset_paths

set_datadir("/mnt/surfer/data")
print(uspy.LEEMDIR)  # follows along automatically

# %%
reset_paths()
print(uspy.LEEMDIR)  # back to the default

# %% [markdown]
# To make this permanent for a project or a machine, add it to the same `uspy.toml`:
#
# ```toml
# [uspy.paths]
# datadir = "/mnt/surfer/data"
# ```
#
# `leemdir` then defaults to `<datadir>/LEEM` automatically; `~` and `$VAR` in the path are
# expanded. This table is only looked for in the folder you started the notebook from and in
# `~/.config/uspy/uspy.toml` -- unlike the reader settings above, there's no data folder to look
# relative to, since that's exactly what's being resolved.

# %% [markdown]
# ## Summary
#
# When several ways of choosing a reader apply at once, the `reader=` argument wins, then a
# `set_reader(...)` session override, then a folder's `uspy.toml`, and finally the Elmitec default.
# The default mask follows the same order, with `set_default_mask(...)`/`[uspy.mask]`/the reader's
# built-in default in place of `set_reader`/`[uspy]`/the Elmitec default.
#
# If uspy doesn't yet support your endstation's file format, see `examples/uspy_toml/README.md` and
# ask a maintainer about adding a reader for it.
