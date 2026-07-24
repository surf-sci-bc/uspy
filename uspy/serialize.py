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

    def dump(self, fname: str, use_pickle: bool = True) -> None:
        """Dumps the object into a JSON/pickle file."""
        with Path(fname).open("wb") as pfile:
            pfile.write(self.dumps(use_pickle))

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

    def save(self, fname: str) -> Loadable:
        """Save the object to a file depending on the filename extension."""
        compress = pickl = False

        if fname.endswith(self._pickle_extension + ".bz2"):
            compress = True
            pickl = True
        elif fname.endswith(self._pickle_extension):
            pickl = True
        elif not fname.endswith(".json"):
            raise ValueError("No meaningful fileextension recognized.")

        if compress:
            print("Compressing data...")
            with bz2.BZ2File(fname, "w") as bzfile:
                bzfile.write(self.dumps(use_pickle=pickl))
            print("Done.")
        else:
            with Path(fname).open("wb") as pfile:
                pfile.write(self.dumps(use_pickle=pickl))

        return self

    @classmethod
    def load(cls, fname: str) -> Loadable:
        """Returns an object retrieved from a JSON/pickle file."""

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

        return obj
