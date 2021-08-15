from __future__ import annotations
import numbers

import numpy as np
import pytest
from agfalta.base import DataObject, DataObjectStack
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
        if not isinstance(source, (np.ndarray, numbers.Number)):
            print(source, type(source))
            raise TypeError
        data = {"matrix": source, "string": "TestMeta"}
        self._source = source
        return data

    def is_compatible(self, other: DataObject) -> bool:
        if isinstance(self.matrix, np.ndarray):
            return self.matrix.shape == other.matrix.shape
        if isinstance(self.matrix, numbers.Number) and isinstance(
            other.matrix, numbers.Number
        ):
            return True
        return False


### Test DataObjectStack

class MinimalObjectStack(DataObjectStack):
    _type = MinimalObject


@pytest.mark.parametrize("source", [np.array([[1, 2], [3, 4]])])
def test_dataobject_generation(source):
    ### DataObjects cannot be created itself
    with pytest.raises(NotImplementedError):
        DataObject("nosource")

    obj = MinimalObject(source)

    assert hasattr(obj, "matrix")
    assert hasattr(obj, "string")
    assert hasattr(obj, "some_meta")
    assert not hasattr(obj, "bogus")

    assert (obj.matrix == source).all()
    assert obj.string is "TestMeta"
    assert obj.some_meta is "nothing"

    assert not DeepDiff(
        obj.meta, {"string": "TestMeta", "some_meta": "nothing"}, ignore_order=True
    )
    assert not DeepDiff(obj.data, {"matrix": np.array(source)})


@pytest.mark.parametrize("val", [1, "bogus"])
@pytest.mark.parametrize("source", [np.array([[1, 2], [3, 4]])])
def test_dataobj_setattr(source, val):
    obj = MinimalObject(source)
    with pytest.raises(AttributeError):
        print(obj.attr)
    obj.attr = val
    assert obj.attr == val


@pytest.mark.parametrize(
    ("source", "other_source"), [(np.array([[1, 2], [3, 4]]), np.ones(2))]
)
def test_datobject_comparision(source, other_source):
    obj = MinimalObject(source)
    same_obj = MinimalObject(source)
    other_obj = MinimalObject(other_source)

    assert other_obj is not obj
    assert same_obj is not obj
    assert same_obj == obj
    assert same_obj != other_obj


@pytest.mark.parametrize("source", [np.array([[1, 2], [3, 4]])])
def test_dataobject_data_mutability(source):
    obj = MinimalObject(source)

    # data and meta should be immutable
    with pytest.raises(AttributeError):
        obj.data = None
        obj.meta = None

    # _data and _meta are not immutable
    obj._data = None
    obj._meta = None

    assert obj._data is None
    assert obj._meta is None


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_sources(virtual):
    sources = np.linspace(0, 100, 10)

    stack = MinimalObjectStack(sources, virtual=virtual)
    assert len(stack) == len(sources)

    data = np.array([element.matrix for element in stack])
    assert (data == sources).all()

    objs = [MinimalObject(source) for source in sources]

    stack = MinimalObjectStack(objs)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_objects(virtual):
    sources = np.linspace(0, 100, 11)

    objs = [MinimalObject(source) for source in sources]

    stack = MinimalObjectStack(objs, virtual=virtual)

    assert len(stack) == len(sources)
    for index, element in enumerate(stack):
        assert element is objs[index]  # Elements should be the same not equal


def test_stack_virtual():
    sources = np.linspace(0, 100, 11)
    virt_stack = MinimalObjectStack(sources, virtual=True)
    real_stack = MinimalObjectStack(sources, virtual=False)

    # For Virtual stacks the elements are sources
    assert (virt_stack.elements == np.asarray(sources)).all()
    for elv, elr in zip(virt_stack, real_stack):
        assert elv == elr
    # reifiy
    virt_stack.virtual = False
    assert not virt_stack.virtual
    assert virt_stack.elements == real_stack.elements

    # virtualization
    virt_stack.virtual = True
    assert virt_stack.virtual
    assert (virt_stack.elements == np.asarray(sources)).all()


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_setattr(virtual):
    sources = np.linspace(0, 100, 11)
    stack = MinimalObjectStack(sources, virtual=virtual)

    # setting of per element attributes

    energy = np.linspace(0, 20, len(sources))
    if virtual:
        with pytest.raises(ValueError):
            stack.energy = energy
            stack.bogus = "Bogus"
            stack.array = [1, 2]
            stack.number = 1
    else:
        stack.energy = energy
        assert (stack.energy == energy).all()
        stack.bogus = "Bogus"
        stack.array = [1, 2]
        stack.number = 1
        assert stack.bogus is "Bogus"
        assert stack.array == [1, 2]
        assert stack.number == 1


@pytest.mark.parametrize("virtual", [False])
def test_stack_extend(virtual):
    sources = np.linspace(0, 100, 11)
    sources_ext = np.linspace(100, 200, 11)
    stack = MinimalObjectStack(sources, virtual=virtual)
    stack_ext = MinimalObjectStack(sources_ext, virtual=virtual)
    objs = [element for element in stack]
    objs.extend([element for element in stack_ext])

    stack.extend(stack_ext)

    for ii, jj in zip(stack.elements, objs):
        assert ii is jj

@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem(virtual):
    sources = np.linspace(0, 100, 11)
    stack = MinimalObjectStack(sources, virtual=virtual)

    assert isinstance(stack[:3], MinimalObjectStack)
    assert isinstance(stack[0], MinimalObject)
