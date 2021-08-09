"""
Basic data containers.
"""
from __future__ import annotations
from typing import Any



class Loadable:
    """
    Base class for loadable objects. Contains methods for serializing
    and deserializing.
    """
    def dump(self, fname: str) -> None:
        """Dumps the object into a JSON/pickle file."""

    def load(self, fname: str) -> Loadable:
        """Returns an object retrieved from a JSON/pickle file."""

    def parametrize(self) -> dict[str, Any]:
        """Reduce the object to a dictionary from which contents the
        whole object can be reconstructed, without saving huge raw data."""
        raise NotImplementedError

    def reconstruct(self, params: dict[str, Any]) -> Loadable:
        """Reconstruct an object from a dictionary made by parametrize"""
        raise NotImplementedError


class DataObject:
    """
    Base class for data objects like images, lines, points, ...
    """
    _data_keys = ()
    _meta_keys = ()

    def __init__(self) -> None:
        self._data = {}
        self._meta = {}
        for k, value in self.parse():
            if k in self._data_keys:
                self._data[k] = value
            elif k in self._meta_keys:
                self._meta[k] = value
            else:
                raise ValueError(f"Key {k} not valid for class {type(self)}")

    def parse(self) -> dict[str, Any]:
        """Read in data."""
        raise NotImplementedError

    @property
    def data(self) -> dict[str, Any]:
        """Immutable data container."""
        return self._data.copy()

    @property
    def meta(self) -> dict[str, Any]:
        """Mutable metadata dictionary."""
        return self._meta.copy()

    def __getattr__(self, attr: str) -> Any:
        if attr in self._data:
            return self._data[attr]
        if attr in self._meta:
            return self._meta[attr]
        raise AttributeError(f"No attribute named {attr}")

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in self._meta:
            self._meta[attr] = value
        elif attr in self._data:
            raise ValueError(f"Data attribute {attr} is immutable")
        else:
            super().__setattr__(attr, value)


class Image(DataObject):
    """
    Base class for all spatial 2D data.
    """
    _data_keys = ("image")
