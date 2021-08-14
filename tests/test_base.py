from __future__ import annotations

import numpy as np
import pytest
from agfalta.base import DataObject
from deepdiff.diff import DeepDiff

"""Tests the agfalta.base module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

### Test DataObject


class MinimalObject(DataObject):
    """A Class used for Tests. Source is converted to a numpy array"""

    _data_keys = ("matrix",)
    _meta_defaults = {"some_meta": "nothing"}

    def parse(self, source: str) -> dict[str, Any]:
        data = {"matrix": np.array(source), "string": "TestMeta"}
        return data

    def is_compatible(self, other: DataObject) -> bool:
        return self.matrix.shape == other.matrix.shape


@pytest.mark.parametrize("source", [[[1, 2], [3, 4]], 1])
def test_dataobject_generation(source):
    ### DataObjects cannot be created itself
    with pytest.raises(NotImplementedError):
        DataObject("nosource")

    obj = MinimalObject(source)

    assert hasattr(obj, "matrix")
    assert hasattr(obj, "string")
    assert hasattr(obj, "some_meta")
    assert not hasattr(obj, "bogus")

    assert (obj.matrix == np.array(source)).all()
    assert obj.string is "TestMeta"
    assert obj.some_meta is "nothing"

    assert not DeepDiff(
        obj.meta, {"string": "TestMeta", "some_meta": "nothing"}, ignore_order=True
    )
    assert not DeepDiff(obj.data, {"matrix": np.array(source)})


@pytest.mark.parametrize("val", [1, "bogus", np.ones(2)])
@pytest.mark.parametrize("source", [[[1, 2], [3, 4]], 1])
def test_dataobj_setattr(source, val):
    obj = MinimalObject(source)
    with pytest.raises(AttributeError):
        print(obj.attr)
    obj.attr = val
    assert obj.attr is val


@pytest.mark.parametrize(
    ("source", "other_source"), [([[1, 2], [3, 4]], [[1, 0], [0, 1]])]
)
def test_datobject_comparision(source, other_source):
    obj = MinimalObject(source)
    same_obj = MinimalObject(source)
    other_obj = MinimalObject(other_source)

    assert other_obj is not obj
    assert same_obj is not obj
    assert same_obj == obj
    assert same_obj != other_obj
