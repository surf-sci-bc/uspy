"""
Generic serialization mixin, independent of the data model.
"""

from __future__ import annotations
from pathlib import Path
import bz2
import pickle

import json_tricks


class Loadable:
    """
    Base class for loadable objects. Contains methods for serializing
    and deserializing.
    """

    _pickle_extension = ".pickle"

    @classmethod
    def _is_serialize_path(cls, fname: str) -> bool:
        """True for extensions `save`/`load` handle themselves (pickle/bz2/json), as opposed
        to a format-specific extension a subclass dispatches elsewhere (e.g. `.tif`/`.png` in
        `Image.save`)."""
        return fname.endswith(
            (cls._pickle_extension + ".bz2", cls._pickle_extension, ".json")
        )

    def _dump_pickle(self, fileobj) -> None:
        """Pickle straight into an already-open file object, instead of building the
        whole pickled bytestring in memory first (as `dumps()` does) and then writing
        it -- avoids holding both the live object graph and its full byte serialization
        in RAM at once, which matters for large stacks."""
        try:
            pickle.dump(self, fileobj, protocol=4)
        except RecursionError:
            print("Did not pickle due to recursion error.")
            raise

    def dump(self, fname: str, use_pickle: bool = True) -> None:
        """Dumps the object into a JSON/pickle file."""
        with Path(fname).open("wb") as pfile:
            if use_pickle:
                self._dump_pickle(pfile)
            else:
                pfile.write(self.dumps(use_pickle=False))

    def dumps(self, use_pickle: bool = True) -> bytes:
        """Dumps the object with pickle and returns it as a bytestring"""
        if use_pickle:
            try:
                return pickle.dumps(self, protocol=4)
            except RecursionError:
                print("Did not pickle due to recursion error.")
                raise
        else:
            return json_tricks.dumps(self, allow_nan=True).encode("utf-8")

    def save(self, fname: str, **kwargs) -> Loadable:
        """Save the object to a file depending on the filename extension.

        `**kwargs` are only meaningful for an HDF5 target (`.h5`/`.hdf5`/`.nxs`; see
        `uspy.hdf5.write` for the accepted options) and raise `TypeError` for any other
        extension -- silently swallowing e.g. a typo'd `compresion=9` on a `.pickle` save
        would otherwise be exactly the kind of silent behavior change this project avoids.
        """
        # Deferred import: uspy.hdf5 is data-model-aware (it inspects _data/_meta/_units), so
        # importing it here rather than at module level keeps this module -- explicitly
        # "independent of the data model" per its own docstring -- free of that dependency,
        # and means a stale environment missing h5py only breaks on an actual .h5 save/load.
        from uspy.hdf5 import is_hdf5_path

        if is_hdf5_path(fname):
            from uspy.hdf5 import write

            write(self, fname, **kwargs)
            return self

        if kwargs:
            raise TypeError(
                f"save() got unexpected keyword argument(s) {list(kwargs)!r}: these are only "
                "accepted for an HDF5 target (.h5/.hdf5/.nxs)."
            )

        compress = pickl = False

        if fname.endswith(self._pickle_extension + ".bz2"):
            compress = True
            pickl = True
        elif fname.endswith(self._pickle_extension):
            pickl = True
        elif not fname.endswith(".json"):
            raise ValueError(f"No meaningful fileextension recognized for {fname!r}.")

        if compress:
            print("Compressing data...")
            with bz2.BZ2File(fname, "w") as bzfile:
                self._dump_pickle(bzfile)
            print("Done.")
        elif pickl:
            with Path(fname).open("wb") as pfile:
                self._dump_pickle(pfile)
        else:
            with Path(fname).open("wb") as pfile:
                pfile.write(self.dumps(use_pickle=False))

        return self

    @classmethod
    def load(cls, fname: str) -> Loadable:
        """Returns an object retrieved from an HDF5/JSON/pickle file."""
        from uspy.hdf5 import is_hdf5_path

        if is_hdf5_path(fname):
            from uspy.hdf5 import read

            return read(fname)

        if fname.endswith(cls._pickle_extension + ".bz2"):
            print("Uncompressing data...")
            file = bz2.BZ2File(fname, "rb")
            obj = pickle.load(file)
            print("Done.")
        elif fname.endswith(cls._pickle_extension):
            with Path(fname).open("rb") as pfile:
                obj = pickle.load(pfile)
        elif fname.endswith(".json"):
            obj = json_tricks.load(fname)
        else:
            raise ValueError(f"No meaningful fileextension recognized for {fname!r}.")

        return obj
