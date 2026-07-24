# Baseline `uspy.toml` files

Each subfolder here holds a ready-to-copy baseline `uspy.toml` for one built-in reader,
with plotting fields tailored to that endstation's metadata. Copy the file (unchanged or
edited to taste) into the folder that contains your data, named exactly `uspy.toml`.

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
