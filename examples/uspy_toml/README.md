# Baseline `uspy.toml` files

Each subfolder here holds a ready-to-copy baseline `uspy.toml`. `elmitec`/`elettra`/`smart`
are per-reader, with plotting fields tailored to that endstation's metadata -- copy one of
those (unchanged or edited to taste) into the folder that contains your data, named exactly
`uspy.toml`. `site/uspy.toml` is different: a combined reader+data-root baseline meant for
the *global* config location (`~/.config/uspy/uspy.toml`) instead of a data folder -- see
"Data directories" below.

See the module docstring of `uspy.leem.readers` for the full reader-selection order.
Two things worth knowing that aren't obvious from a single example file:

## Config search path

A `uspy.toml` doesn't have to sit right next to the data. It's looked for at several
locations, most specific first, and the results are layered together (a more specific
file's `[uspy.plotting]` keys override a less specific file's, but both apply):

1. the data folder
2. that folder's parent
3. the current working directory
4. a global config directory: `$XDG_CONFIG_HOME/uspy/uspy.toml`, or
   `~/.config/uspy/uspy.toml` if `XDG_CONFIG_HOME` is unset

The global location is useful for a measurement machine that only ever produces one file
format: drop one `uspy.toml` there (e.g. the `smart/uspy.toml` baseline) and every load
on that machine uses the right reader without needing a per-folder file at all.

## Session override from Python/a notebook

For a one-off session (e.g. exploring data from an endstation you don't want to leave a
config file for), set the reader programmatically instead:

```python
from uspy.leem.readers import set_reader

set_reader("elettra")          # every load for the rest of the session uses this reader
# ... load images/stacks without needing reader=... or a uspy.toml ...
set_reader(None)                # or reset_reader() -- clears the override again
```

This ranks below an explicit `reader=` argument to `LEEMImg`/`LEEMStack`, but above every
`uspy.toml`.

## Default mask (`[uspy.mask]`)

The same `uspy.toml` files can also declare the instrument's default mask -- e.g. the
LEEM channelplate aperture -- used by `LEEMStack.align()`, `plot_img()`, and
`make_video()` when called with `mask=True` (never automatically otherwise):

```toml
[uspy.mask]
shape  = "circle"   # circle | ellipse | square | rectangle | polygon | point | "none"
x0     = "50%"      # percent strings scale with frame width  (x0, width, radius)
y0     = "50%"      # percent strings scale with frame height (y0, height)
radius = "45%"      # plain numbers are absolute pixels; may be mixed with percent strings
```

Without a `[uspy.mask]` table anywhere, each reader has its own built-in default (a
channelplate circle, `x0="50%", y0="50%", radius="45%"`, for every built-in reader --
see `LEEMReader.default_mask`). Set `shape = "none"` to disable the default mask for an
instrument that doesn't need one. Like `reader`/`[uspy.options]`, `[uspy.mask]` is taken
as a unit from the first file on the search path that declares it, not merged per-key
like `[uspy.plotting]`.

A session override, mirroring `set_reader`, ranks above every `uspy.toml`:

```python
from uspy.leem.readers import set_default_mask, reset_default_mask

set_default_mask(False)          # disable the default mask for the rest of the session
reset_default_mask()             # clears the override again
```

See `LEEMImg.default_mask`/`uspy.leem.readers.resolve_default_mask` for the full
resolution order.

## Data directories (`[uspy.paths]`)

The same `uspy.toml` files can also declare where real experimental data lives, via a
`[uspy.paths]` table:

```toml
[uspy.paths]
datadir = "/mnt/surfer/data"
leemdir = "/mnt/surfer/data/LEEM"   # optional; defaults to "<datadir>/LEEM"
xps     = "/mnt/surfer/data/XPS"    # arbitrary extra keys are allowed
```

`uspy.DATADIR`/`uspy.LEEMDIR` (each a `str` ending in a path separator, e.g.
`"/mnt/surfer/data/"`) resolve these lazily on every access -- with no `uspy.toml`
anywhere, they default to `~/data/` and `~/data/LEEM/`. Other keys (`xps` above) are
read with `uspy.config.get_path("xps")`.

**The search path here is shorter than the reader one above:** there's no data folder
to look relative to `[uspy.paths]` -- the data folder is exactly what's being resolved
-- so only two locations are checked, most specific first:

1. the current working directory
2. the global config directory: `$XDG_CONFIG_HOME/uspy/uspy.toml`, or
   `~/.config/uspy/uspy.toml` if `XDG_CONFIG_HOME` is unset

As with the reader tables, multiple files layer together, most specific first -- but
**a full `datadir`/`leemdir` pair is taken as a unit from the most specific file that
mentions either one**, rather than merged key-by-key: if a project-local `uspy.toml`
overrides only `datadir`, `leemdir` moves with it (defaulting to `<that datadir>/LEEM`)
even if a less-specific file set `leemdir` explicitly. Otherwise `LEEMDIR` could
silently end up pointing into an unrelated tree.

Values may use `~` and `$VAR`/`${VAR}` environment-variable references; a relative
value is anchored to the folder of the `uspy.toml` that declared it.

A session override, mirroring `set_reader`, ranks above every `uspy.toml`:

```python
from uspy.config import set_datadir, reset_paths

set_datadir("/mnt/surfer/data")   # every uspy.DATADIR/LEEMDIR access uses this now
reset_paths()                     # clears it again
```
