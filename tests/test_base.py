"""Tests the uspy.base module."""

from __future__ import annotations
import numbers

import cv2 as cv
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
from uspy.dataobject import DataObject, DataObjectStack, Image, ImageStack, Line
from uspy.dataobject.deferred import DeferredArray
from deepdiff.diff import DeepDiff
import os

# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=attribute-defined-outside-init

### Test DataObject

from .conftest import (
    TESTDATA_DIR,
    TESTIMAGE,
    TESTIMAGE_ENDING,
    TESTIMAGE_NAME,
    ARRAY_1D,
    ARRAY_2D,
    quad_line,
)


def array2d():
    np.random.rand(2, 2)


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
            print(self.matrix.shape, other.matrix.shape)
            return self.matrix.shape == other.matrix.shape
        if isinstance(self.matrix, numbers.Number) and isinstance(
            other.matrix, numbers.Number
        ):
            return True
        return False

    ### Property to set Source to None when needed
    # @property
    # def source(self) -> Optional[Any]:
    #     return super().source

    # @source.setter
    # def source(self, val):
    #     self._source = val


### Test DataObjectStack


class MinimalObjectStack(DataObjectStack):
    _type = MinimalObject


class ObjectWithProperties(MinimalObject):
    """Adds a read-only and a writable class-level property, to test that
    DataObject.__setattr__ (and DataObjectStack.__setattr__, via broadcast) actually run a
    property's setter instead of silently writing _meta underneath it."""

    @property
    def readonly_prop(self) -> str:
        return "derived"

    @property
    def writable_prop(self):
        return self._meta.get("writable_prop")

    @writable_prop.setter
    def writable_prop(self, value) -> None:
        # Doubles the value, so a test can tell whether this setter actually ran (10) or
        # whether the assignment instead silently landed as a raw _meta write (5).
        self._meta["writable_prop"] = value * 2


class ObjectWithPropertiesStack(DataObjectStack):
    _type = ObjectWithProperties


@pytest.mark.parametrize("source", ARRAY_2D + ARRAY_1D)
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
    assert (obj.source == source).all()

    assert not DeepDiff(
        obj.meta, {"string": "TestMeta", "some_meta": "nothing"}, ignore_order=True
    )
    assert not DeepDiff(obj.data, {"matrix": np.array(source)})


@pytest.mark.parametrize("val", [1, "bogus"])
@pytest.mark.parametrize("source", ARRAY_2D)
def test_dataobj_setattr(source, val):
    obj = MinimalObject(source)
    with pytest.raises(AttributeError):
        print(obj.attr)
    obj.attr = val
    assert obj.attr == val
    obj.attr = 1
    assert obj.attr == 1


@pytest.mark.parametrize("source", ARRAY_2D)
def test_dataobj_setattr_gets_default_unit(source):
    """Arbitrary metadata must always have a `_units` entry -- plotting/derivation code
    (IntensityLine.parse, waterfall(), ...) indexes `_units` directly by field name and would
    otherwise KeyError the first time a freshly-added field is used as an axis."""
    obj = MinimalObject(source)
    assert "attr" not in obj.unit

    obj.attr = 5
    assert obj.unit["attr"] == ""

    # setting a unit, then re-assigning the value, must not clobber the unit
    obj._units["attr"] = "eV"
    obj.attr = 6
    assert obj.unit["attr"] == "eV"


@pytest.mark.parametrize("source", ARRAY_2D)
def test_dataobject_every_meta_key_has_a_unit(source):
    """The guarantee the whole change is about: right after construction, every key in
    `.meta` has a matching entry in `.unit` (default "" if none was ever set)."""
    obj = MinimalObject(source)
    assert set(obj.meta) <= set(obj.unit)


def test_dataobject_construction_does_not_mutate_unit_defaults():
    """Regression test: DataObject.__init__ used to gap-fill the *class-level*
    `_unit_defaults` dict instead of the instance's `_units`, permanently polluting a dict
    shared by every sibling subclass that doesn't declare its own `_unit_defaults` (e.g.
    Image and Waterfall both inherit DataObject._unit_defaults directly)."""
    before = dict(Image._unit_defaults)
    Image(np.zeros((4, 5)))
    assert Image._unit_defaults == before

    before_do = dict(DataObject._unit_defaults)
    MinimalObject(np.eye(2))  # MinimalObject doesn't declare its own _unit_defaults either
    assert DataObject._unit_defaults == before_do


@pytest.mark.parametrize(("source", "other_source"), [(ARRAY_2D[0], ARRAY_2D[1])])
def test_datobject_comparision(source, other_source):
    obj = MinimalObject(source)
    same_obj = MinimalObject(source)
    other_obj = MinimalObject(other_source)

    assert other_obj is not obj
    assert same_obj is not obj
    assert same_obj == obj
    assert same_obj != other_obj


@pytest.mark.parametrize("source", ARRAY_2D)
def test_dataobject_datadir_not_overridable(source):
    """`data`/`meta` are read-only properties (backed by `_data`/`_meta`): assigning to
    them now runs the property's setter, i.e. there isn't one, so it raises -- it no longer
    silently no-ops. `_data`/`_meta` themselves (the underlying storage) still can't be
    rebound once set, by design (see `DataObject.__setattr__`)."""
    obj = MinimalObject(source)

    with pytest.raises(AttributeError):
        obj.data = None
    with pytest.raises(AttributeError):
        obj.meta = None
    assert obj.data is not None
    assert obj.meta is not None

    obj._data = None
    obj._meta = None

    assert obj._data is not None
    assert obj._meta is not None


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_sources(list2darrays, virtual):
    sources = list2darrays

    stack = MinimalObjectStack(sources, virtual=virtual)
    assert len(stack) == len(sources)

    data = np.array([element.matrix for element in stack])
    assert (data == sources).all()

    objs = [MinimalObject(source) for source in sources]

    stack = MinimalObjectStack(objs)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_from_objects(list1darrays, virtual):
    sources = list1darrays

    objs = [MinimalObject(source) for source in sources]

    stack = MinimalObjectStack(objs, virtual=virtual)

    assert len(stack) == len(sources)
    for index, element in enumerate(stack):
        assert element is objs[index]  # Elements should be the same not equal


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_generation_with_not_compatible_objs(virtual):
    sources = [np.eye(2), np.eye(3)]

    objs = [MinimalObject(source) for source in sources]
    with pytest.raises(TypeError):
        MinimalObjectStack(objs, virtual=virtual)


def test_stack_virtual(list1darrays):
    sources = list1darrays
    virt_stack = MinimalObjectStack(sources, virtual=True)
    real_stack = MinimalObjectStack(sources, virtual=False)

    # A virtual stack's elements are real, stable DataObjects -- not bare sources. `.sources`
    # is where the underlying source list lives (see test_stack_sources_vs_elements below).
    assert all(isinstance(element, MinimalObject) for element in virt_stack.elements)
    assert all(element.is_virtual for element in virt_stack.elements)
    # stable identity: repeated access returns the *same* element, not a fresh re-parse
    assert virt_stack[0] is virt_stack[0]
    for elv, elr in zip(virt_stack, real_stack):
        assert elv == elr

    # reify
    virt_stack.virtual = False
    assert not virt_stack.virtual
    assert not any(element.is_virtual for element in virt_stack.elements)
    for elv, elr in zip(virt_stack, real_stack):
        assert elv == elr

    # virtualization
    virt_stack.virtual = True
    assert virt_stack.virtual
    assert all(element.is_virtual for element in virt_stack.elements)
    for elv, elr in zip(virt_stack, real_stack):
        assert elv == elr


def test_stack_sources_vs_elements(list1darrays):
    sources = list1darrays
    virt_stack = MinimalObjectStack(sources, virtual=True)
    assert list(virt_stack.sources) == list(sources)
    assert all(isinstance(element, MinimalObject) for element in virt_stack.elements)


def test_stack_copy_of_virtual_stack_does_not_share_metadata_dicts(list1darrays):
    """Regression guard for the bug the old lazy backend had: copying (or slicing) a virtual
    stack must give each copy independent metadata storage, not dicts shared by reference --
    mutating one copy's metadata must not leak into the original or into other copies."""
    stack = MinimalObjectStack(list1darrays, virtual=True)
    other = stack.copy()

    other[0].some_meta = "changed-on-copy"

    assert stack[0].some_meta != "changed-on-copy"


def test_stack_slice_virtual_transition_does_not_affect_parent(list1darrays):
    """Regression guard: a slice shares element objects with its parent (Python list-slice
    semantics), but materializing the *slice* must not mutate -- and thus must not pin into
    memory -- the parent's (still meant to be virtual) elements. Before this fix,
    `_materialize_elements`/`_deconstruct` mutated shared elements in place, so a materialized
    slice silently un-virtualized the parent's frames while `parent.virtual` kept reporting
    `True`."""
    parent = MinimalObjectStack(list1darrays, virtual=True)
    sub = parent[:2]

    assert sub[0] is parent[0]  # slicing is still a view, sharing elements
    assert sub.virtual is True

    sub.virtual = False

    # The parent must stay fully virtual: neither its `virtual` flag nor its elements were
    # touched by materializing the slice.
    assert parent.virtual is True
    assert all(element.is_virtual for element in parent.elements)
    # The slice, in turn, no longer shares identity with the parent -- the transition rebuilt
    # its elements rather than mutating the shared ones.
    assert sub[0] is not parent[0]
    assert not sub.virtual
    assert not any(element.is_virtual for element in sub.elements)


def test_stack_slice_defer_transition_does_not_affect_parent(list1darrays):
    """Mirror of the above in the other direction: virtualizing a slice of an eager stack must
    not re-defer (and thereby discard the resident data of) the parent's shared elements."""
    parent = MinimalObjectStack(list1darrays, virtual=False)
    sub = parent[:2]

    assert sub[0] is parent[0]
    assert sub.virtual is False

    sub.virtual = True

    assert parent.virtual is False
    assert not any(element.is_virtual for element in parent.elements)
    assert sub[0] is not parent[0]
    assert sub.virtual is True
    assert all(element.is_virtual for element in sub.elements)


def test_stack_virtual_flag_is_derived_and_honest(list1darrays):
    """`stack.virtual` is derived from the elements rather than a stored flag that could drift:
    pinning a single element resident (via a direct `_data` write, which warns) must be
    reflected immediately, without any explicit `stack.virtual = ...` call."""
    stack = MinimalObjectStack(list1darrays, virtual=True)
    assert stack.virtual is True

    with pytest.warns(UserWarning, match="deferred data with a resident"):
        stack[0].matrix = np.asarray(stack[0].matrix)

    assert stack.virtual is False
    assert not stack[0].is_virtual
    assert all(element.is_virtual for element in stack.elements[1:])


def test_stack_constructor_does_not_materialize_caller_objects(list1darrays):
    """Building an eager stack directly from another (virtual) stack's elements must not
    materialize those elements in place -- that would silently pin the source stack's data in
    memory even though the source stack itself was never touched."""
    virt = MinimalObjectStack(list1darrays, virtual=True)

    eager = MinimalObjectStack(list(virt.elements))  # virtual=False (default)

    assert eager.virtual is False
    assert not any(element.is_virtual for element in eager.elements)
    # The source stack's own elements must be untouched.
    assert virt.virtual is True
    assert all(element.is_virtual for element in virt.elements)
    assert eager[0] is not virt[0]


def test_stack_constructor_mixed_elements_end_fully_resident(list1darrays):
    """A stack built directly from a mix of virtual and resident objects used to leave that
    mix in place silently (no materialization, and -- if `virtual=True` was requested -- no
    warning either, for a *partially* virtual input). Both requested-virtuality cases must now
    end up fully resident, since the objects were "directly given" either way."""
    objs = [MinimalObject(source) for source in list1darrays]
    objs[0].defer()  # one element deferred, the rest resident: a genuine mix

    with pytest.warns(UserWarning, match="won't be virtual"):
        stack_requesting_virtual = MinimalObjectStack(objs, virtual=True)
    assert stack_requesting_virtual.virtual is False
    assert not any(element.is_virtual for element in stack_requesting_virtual.elements)

    objs2 = [MinimalObject(source) for source in list1darrays]
    objs2[0].defer()
    stack_requesting_eager = MinimalObjectStack(objs2, virtual=False)
    assert stack_requesting_eager.virtual is False
    assert not any(element.is_virtual for element in stack_requesting_eager.elements)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_setattr(list1darrays, virtual):
    """A stack has no metadata of its own: a scalar set on the stack means "this value, for
    every element" (whether or not the name already exists as a per-element field); only a
    length-matched sequence distributes one value per element; anything else (a mismatched-
    length sequence for a genuinely new name) attaches to the stack object itself. Elements of
    a virtual stack are real, stable DataObjects now (not re-parsed per access), so a write
    persists exactly like it does on an eager stack."""
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=virtual)

    energy = np.linspace(0, 20, len(sources))
    stack.energy = energy
    assert (stack.energy == energy).all()

    # A string scalar is never split character-wise (regardless of whether len(string)
    # happens to equal len(stack)) -- it broadcasts like any other scalar.
    stack.bogus = "Bogus"
    assert (stack.bogus == "Bogus").all()
    assert len(stack.bogus) == len(sources)

    stack.number = 1
    assert (stack.number == 1).all()
    assert len(stack.number) == len(sources)

    # A mismatched-length sequence for a name that collides with nothing on the elements
    # attaches to the stack itself.
    stack.array = [1, 2]
    assert stack.array == [1, 2]

    if virtual:
        assert all(element.is_virtual for element in stack.elements)


def test_stack_setattr_scalar_broadcasts_to_elements(list1darrays):
    """A stack has no metadata of its own: a scalar assignment -- whether or not `attr`
    already names a real per-element field -- is set as metadata on every element, not
    attached to the stack object. `stack.some_meta = 5.0` used to raise (or, before that,
    silently become a plain instance attribute that permanently shadowed the per-element
    gather in __getattr__); now it's simply equivalent to writing that meta key on every
    element."""
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=False)

    assert len(stack.some_meta) == len(sources)  # baseline: per-element gather works

    stack.some_meta = 5.0

    assert (stack.some_meta == 5.0).all()
    assert len(stack.some_meta) == len(sources)
    assert all(element._meta["some_meta"] == 5.0 for element in stack)
    # not shadowed by a plain instance attribute
    assert "some_meta" not in stack.__dict__


def test_stack_setattr_mismatched_iterable_collision_raises(list1darrays):
    """A mismatched-length *iterable* assignment must be refused when it collides with a
    real per-element field -- it can't be distributed (wrong length) and can't attach to the
    stack either without shadowing that field's per-element gather in __getattr__.
    (Originally added to accommodate uspy/leem/readers/elettra.py's post_process_stack
    attaching a mismatched-length tracefile column to a sliced stack -- that turned out to be
    a bug, not a real use case, and is fixed at its source; see uspy/leem/base.py's
    LEEMStack.__init__.)"""
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=False)

    assert len(stack.some_meta) == len(sources)  # baseline: per-element gather works

    mismatched = ["x"] * (len(sources) + 1)  # deliberately wrong length
    with pytest.raises(ValueError):
        stack.some_meta = mismatched

    # the rejected assignment must not have taken effect
    assert len(stack.some_meta) == len(sources)

    # a genuinely non-colliding mismatched-length iterable is still allowed (existing
    # behavior, e.g. test_stack_setattr's `stack.array = [1, 2]` above)
    stack.totally_new_field = mismatched
    assert stack.totally_new_field == mismatched

    # a genuinely new, non-colliding scalar broadcasts to every element just like a
    # colliding one does -- there's nothing special about "new" vs. "existing" names for a
    # scalar assignment.
    stack.totally_new_label = "fine"
    assert (stack.totally_new_label == "fine").all()
    assert len(stack.totally_new_label) == len(sources)


### Empty-stack construction ######################################################################

# A stack is a container for one or more objects -- an empty stack is deliberately not a
# supported thing to *construct* (DataObjectStack.__init__(source=[]) must raise a clear
# ValueError, not the opaque `IndexError: list index out of range` it used to raise via
# `isinstance(source[0], self._type)` with an unchecked source[0]). An empty stack can still
# *arise* from manipulating an existing one (e.g. `del stack[:]`), and that route must keep
# working -- the safeguards elsewhere in this module (e.g. the scalar-setattr guard below) exist
# specifically to handle that already-reachable case sanely.


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_empty_list_raises_valueerror_not_indexerror(virtual):
    with pytest.raises(ValueError):
        MinimalObjectStack([], virtual=virtual)


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_zero_width_slice_raises(list1darrays, virtual):
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=virtual)
    with pytest.raises(ValueError):
        stack[0:0]
    with pytest.raises(ValueError):
        stack[len(stack) : len(stack)]
    with pytest.raises(ValueError):
        stack[5:5]


@pytest.mark.parametrize("source", ["", "no/such/path", "no/such/*.tif"])
def test_imagestack_bad_string_source_raises_valueerror_not_indexerror(source):
    # A string source that resolves to no files is a typo, not an empty stack -- it must keep
    # raising its own clear ValueError (from _split_source), distinct from the empty-list
    # ValueError above, and never IndexError. "" is the tricky case: source[0] on an empty
    # string is what used to raise IndexError directly inside __init__, before _split_source
    # was ever reached.
    with pytest.raises(ValueError) as excinfo:
        ImageStack(source)
    assert not isinstance(excinfo.value, IndexError)


def test_stack_del_all_still_yields_working_empty_stack(list1darrays):
    """Emptying a stack via manipulation (as opposed to constructing one empty) is not being
    taken away by this fix -- __delitem__ doesn't go through __init__, so it must keep working,
    and the result must still behave like a normal (if empty) stack."""
    stack = MinimalObjectStack(list1darrays, virtual=False)
    del stack[:]
    assert len(stack) == 0
    assert list(stack) == []
    assert stack.elements == []


def test_stack_setattr_mismatched_iterable_on_empty_stack_raises(list1darrays):
    """Regression for a silent hole found while fixing empty-stack construction: the collision
    check in __setattr__ (`collides = len(self) and hasattr(self[0], attr)`) short-circuits to
    falsy for an empty stack, so a mismatched-length iterable used to sail through to the
    print-and-attach path -- silently creating a stack-level attribute that permanently shadows
    the per-element gather in __getattr__, exactly what the non-empty collision check exists to
    prevent. Must now raise instead, mirroring the scalar branch's existing empty-stack guard.
    """
    stack = MinimalObjectStack(list1darrays, virtual=False)
    del stack[:]
    assert len(stack) == 0
    with pytest.raises(ValueError):
        stack.some_meta = [1, 2, 3]
    # the rejected assignment must not have taken effect as a shadowing instance attribute
    assert "some_meta" not in stack.__dict__


def test_stack_setattr_scalar_broadcast_survives_indexing(list1darrays):
    """A scalar assigned to the stack must really be written onto every element, not just
    kept somewhere that only the stack's own __getattr__ gather can see -- so it also has to
    survive going through a fresh single-element access."""
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=False)

    stack.some_meta = "shared"

    for i in range(len(sources)):
        assert stack[i].some_meta == "shared"


def test_dataobject_setattr_runs_property_setter():
    """Assigning to a name that is a class-level property must run its setter, not silently
    write _meta underneath it (which the getter, a data descriptor, would then never see)."""
    obj = ObjectWithProperties(np.eye(2))

    obj.writable_prop = 5
    assert obj.writable_prop == 10  # setter doubled it -- proves the setter really ran

    with pytest.raises(AttributeError):
        obj.readonly_prop = "x"


def test_stack_setattr_scalar_runs_element_property_setter():
    """A scalar broadcast onto a stack must run each element's property setter, same as
    setting it directly on one element -- and a read-only element property can't be
    broadcast at all."""
    stack = ObjectWithPropertiesStack([np.eye(2), np.eye(2)])

    stack.writable_prop = 5
    assert (stack.writable_prop == 10).all()
    assert all(element.writable_prop == 10 for element in stack)

    with pytest.raises(AttributeError):
        stack.readonly_prop = "x"


def test_stack_setattr_scalar_gets_default_unit(list1darrays):
    """A scalar broadcast onto a stack (see test_stack_setattr_scalar_broadcasts_to_elements)
    must give every element a default unit too, same as setting it directly."""
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=False)

    stack.brand_new_field = 5.0

    assert all(element.unit["brand_new_field"] == "" for element in stack)


def test_stack_set_unit(list1darrays):
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=False)
    stack.some_meta = "shared"

    stack.set_unit("some_meta", "eV")

    assert all(element.unit["some_meta"] == "eV" for element in stack)


def test_stack_set_unit_virtual_persists(list1darrays):
    """A virtual stack's elements are real, stable DataObjects, so `set_unit` persists on them
    exactly like it does for an eager stack (it used to raise, back when iterating a virtual
    stack yielded freshly re-parsed throwaway elements that any write would be lost on)."""
    stack = MinimalObjectStack(list1darrays, virtual=True)
    stack.set_unit("some_meta", "eV")
    assert all(element.unit["some_meta"] == "eV" for element in stack)


@pytest.mark.parametrize("virtual", [False])
@pytest.mark.parametrize(("source", "source_ext"), [(ARRAY_1D[0], ARRAY_1D[1])])
def test_stack_extend(source, source_ext, virtual):
    sources = source.tolist()
    sources_ext = source_ext.tolist()
    stack = MinimalObjectStack(sources, virtual=virtual)
    stack_ext = MinimalObjectStack(sources_ext, virtual=virtual)
    objs = [element for element in stack]
    objs.extend([element for element in stack_ext])

    stack.extend(stack_ext)

    for ii, jj in zip(stack.elements, objs):
        assert ii is jj


@pytest.mark.parametrize("virtual", [False, True])
def test_stack_getitem(list1darrays, virtual):
    sources = list1darrays
    stack = MinimalObjectStack(sources, virtual=virtual)

    assert isinstance(stack[:3], MinimalObjectStack)
    assert isinstance(stack[0], MinimalObject)


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize(("source", "source_add"), [(ARRAY_1D[0], ARRAY_1D[1])])
def test_stack_setitem(source, source_add, virtual):
    sources = source.tolist()
    sources_add = source_add.tolist()
    add_obj = MinimalObject(-10)
    add_stack = MinimalObjectStack(sources_add, virtual=virtual)
    load_stack = MinimalObjectStack(sources, virtual=virtual)

    # set one element to element
    stack = load_stack.copy()
    stack[-1] = add_obj
    if virtual:
        assert stack[-1].source == add_obj.source
        with pytest.raises(ValueError):
            add_obj._source = None
            assert add_obj.source is None
            stack[-1] = add_obj
    else:
        assert stack[-1] is add_obj

    add_obj._source = -10

    # set one element to list
    stack = load_stack.copy()
    with pytest.raises(ValueError):
        stack[0] = add_stack

    # set list to element
    stack = load_stack.copy()
    print(add_obj.source)
    with pytest.raises(TypeError):
        stack[:] = add_obj

    # set list to list

    stack = load_stack.copy()
    stack[0:1] = add_stack
    assert len(stack) == len(sources) + len(sources_add) - 1
    for index, obj in enumerate(stack[: len(add_stack)]):
        assert obj == add_stack[index]
    for index, obj in enumerate(stack[len(add_stack) :]):
        assert obj == load_stack[index + 1]

    stack = load_stack.copy()
    stack[:] = add_stack
    for obj1, obj2 in zip(stack, add_stack):
        assert obj1 == obj2

    # set element to integer

    with pytest.raises(ValueError):
        stack[0] = 1


### Image Class Tests


@pytest.mark.parametrize("fileending", TESTIMAGE_ENDING)
def test_image_generation_from_common_filetypes(fileending):
    img = Image(TESTDATA_DIR + TESTIMAGE_NAME + fileending)
    # pylint: disable = no-member
    assert (img.image == TESTIMAGE).all()
    assert (img.width, img.height) == TESTIMAGE.shape


@pytest.mark.parametrize("source", ARRAY_2D)
def test_image_generation_from_numpy_array(source):
    array = source
    img = Image(array)
    assert (img.image == array).all()
    assert img.width, img.height == source.shape
    assert img.source is None


### Image.shape/dtype without materializing pixel data


class _RaisingLoader:
    """Loader that fails the test if it's ever called -- used to prove `Image.shape`/`dtype`
    don't force a pixel read when the shape/dtype are already knowable some cheaper way."""

    def __call__(self):
        raise AssertionError("pixel loader was called -- shape/dtype should not materialize")


def test_image_shape_dtype_resident_array():
    img = Image(np.zeros((5, 7), dtype=np.float32))
    assert img.shape == (5, 7)
    assert img._dtype == np.float32


def test_image_shape_dtype_from_deferred_handle_without_loading():
    img = Image(np.zeros((2, 2), dtype=np.float32))
    img._data["image"] = DeferredArray(_RaisingLoader(), shape=(5, 7), dtype=np.float32)
    assert img.shape == (5, 7)
    assert img._dtype == np.float32  # must not raise -- loader is never called


def test_image_shape_falls_back_to_meta_without_loading():
    """When the DeferredArray doesn't know its own shape yet, but height/width metadata is
    already available (e.g. from a reader's cheap header-only parse), shape must come from
    metadata rather than forcing a pixel load."""
    img = Image(np.zeros((2, 2), dtype=np.float32))
    img._data["image"] = DeferredArray(_RaisingLoader())
    img._meta["height"] = 5
    img._meta["width"] = 7
    assert img.shape == (5, 7)


def test_image_shape_resident_wins_over_stale_meta():
    """After `img.image = smaller_array`, width/height metadata may be stale (still
    describing the old array) -- the resident array must win."""
    img = Image(np.zeros((5, 7), dtype=np.float32))
    img.image = np.zeros((2, 3), dtype=np.float32)
    assert img.shape == (2, 3)


### DataObject.__setattr__: warns when a resident write pins a still-deferred data key


def test_setattr_warns_when_replacing_deferred_data_with_resident_array():
    """Assigning a resident array onto a still-deferred `_data` key (`img.image = array`)
    silently discards the DeferredArray handle and pins that element in memory for good, even
    though a containing stack's `.virtual` flag won't reflect it -- see the "Memory: large
    stacks" section of CLAUDE.md. This is the single chokepoint every such write funnels
    through, regardless of which operation (warp/mask/filter/...) caused it."""
    img = Image(np.zeros((2, 2), dtype=np.float32))
    img._data["image"] = DeferredArray(lambda: np.ones((2, 2), dtype=np.float32))

    with pytest.warns(UserWarning, match="deferred"):
        img.image = np.zeros((3, 3), dtype=np.float32)

    assert not img.is_virtual


def test_setattr_does_not_warn_for_resident_to_resident_write(recwarn):
    img = Image(np.zeros((2, 2), dtype=np.float32))
    img.image = np.ones((2, 2), dtype=np.float32)
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_setattr_does_not_warn_when_assigning_a_deferred_value(recwarn):
    """Composing onto a deferred element (`img._data["image"] = current.map(fn)`, the pattern
    `LEEMImg.warp`/`ROI.apply`/`Image.filter` all use) must not itself warn -- only a
    deferred-to-resident write should."""
    img = Image(np.zeros((2, 2), dtype=np.float32))
    img._data["image"] = DeferredArray(lambda: np.ones((2, 2), dtype=np.float32))

    img._data["image"] = img._data["image"].map(lambda arr: arr * 2)
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert img.is_virtual


def test_image_incompatibility_reason_names_both_shapes_without_loading():
    a = Image(np.zeros((2, 2), dtype=np.float32))
    a._data["image"] = DeferredArray(_RaisingLoader(), shape=(5, 7), dtype=np.float32)
    b = Image(np.zeros((2, 2), dtype=np.float32))
    b._data["image"] = DeferredArray(_RaisingLoader(), shape=(3, 3), dtype=np.float32)
    assert not a.is_compatible(b)
    reason = a.incompatibility_reason(b)
    assert "(3, 3)" in reason
    assert "(5, 7)" in reason


# Integer Calculation Tests


@pytest.mark.parametrize("x", [10, 2.0, -5.0])
def test_image_add_integer(x, testimgpng):
    img2 = testimgpng + x
    assert (img2.image == TESTIMAGE + x).all()
    assert img2 is not testimgpng

    img2 = x + testimgpng
    assert (img2.image == TESTIMAGE + x).all()
    assert img2 is not testimgpng

    img2 = testimgpng - x
    assert (img2.image == TESTIMAGE - x).all()
    assert img2 is not testimgpng

    img2 = testimgpng * x
    assert (img2.image == TESTIMAGE * x).all()
    assert img2 is not testimgpng

    img2 = x * testimgpng
    assert (img2.image == TESTIMAGE * x).all()
    assert img2 is not testimgpng

    img2 = testimgpng / x
    assert (img2.image == TESTIMAGE / x).all()
    assert img2 is not testimgpng

    with pytest.raises(TypeError):
        _ = x / testimgpng


def test_image_iadd_integer(testimgpng):
    img = testimgpng
    testimgpng += 100
    assert (testimgpng.image == TESTIMAGE + 100).all()
    assert img is testimgpng


def test_image_imul_integer(testimgpng):
    img = testimgpng
    testimgpng *= 0.5
    assert (testimgpng.image == TESTIMAGE * 0.5).all()
    assert img is testimgpng


# Image Calculation Tests


def test_image_add_image(testimgpng):
    img2 = testimgpng + testimgpng / 2
    assert (img2.image == TESTIMAGE * 1.5).all()
    assert img2 is not testimgpng


def test_image_sub_integer(testimgpng):
    img2 = testimgpng - testimgpng / 2
    assert (img2.image == TESTIMAGE * 0.5).all()
    assert img2 is not testimgpng


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_image_div_image(testimgpng):
    img2 = testimgpng / testimgpng
    # 0/0 gives NaN so we have to get rid of it before testing
    assert (np.nan_to_num(img2.image) == np.eye(10)).all()
    assert img2 is not testimgpng


### Image Filter/Contrast Tests
#
# Regression coverage for `DataObject._with_data`, the helper `filter`/`contrast` build their
# result with: a first version silently no-op'd `new._data = ...` because of the pre-existing
# `__setattr__` guard against replacing `_data`/`_meta` once already set on an instance, so
# filtered/contrasted images silently kept their *original* pixels. These tests assert on
# exactly what that bug violated: the result differs from the input as expected, the input is
# untouched, and the result's `_data`/`_meta` are independent dict objects (not the same ones
# the original still holds) -- not just an independent wrapper pointing at shared containers.

FILTER_METHODS = [
    ("gaussian", {}),
    ("blur", {}),
    ("median", {}),
    ("kernel", {"kernel": np.ones((3, 3), dtype=np.float32) / 9}),
]


@pytest.mark.parametrize(
    "method, kwargs", FILTER_METHODS, ids=[case[0] for case in FILTER_METHODS]
)
def test_image_filter(method, kwargs, testimgpng):
    orig = testimgpng.image.copy()

    result = testimgpng.filter(method, **kwargs)

    if method == "gaussian":
        expected = cv.GaussianBlur(orig, (3, 3), -1)
    elif method == "blur":
        expected = cv.blur(orig, (3, 3))
    elif method == "median":
        expected = cv.medianBlur(orig, 3)
    else:
        expected = cv.filter2D(orig, ddepth=-1, kernel=kwargs["kernel"])

    assert np.allclose(result.image, expected)
    assert result is not testimgpng
    assert result.image is not testimgpng.image
    assert np.array_equal(testimgpng.image, orig)  # original untouched
    assert result.data is not testimgpng.data
    assert result.meta is not testimgpng.meta


def test_image_filter_unknown_method_raises(testimgpng):
    with pytest.raises(ValueError):
        testimgpng.filter("nonsense")


def test_image_contrast_equal(testimgpng):
    orig = testimgpng.image.copy()

    result = testimgpng.contrast("equal")
    expected = cv.equalizeHist(orig.astype(np.uint8))

    assert np.allclose(result.image, expected)
    assert result is not testimgpng
    assert result.image is not testimgpng.image
    assert np.array_equal(testimgpng.image, orig)
    assert result.data is not testimgpng.data
    assert result.meta is not testimgpng.meta


def test_image_contrast_clahe():
    # CLAHE requires an 8-/16-bit single-channel source -- unlike the float32 `testimgpng`
    # fixture, which .contrast("clahe") can't currently handle (a separate, pre-existing
    # limitation of the "clahe" branch, not something this test is about); use a directly
    # constructed uint8 image so this actually exercises the _with_data-based code path.
    array = np.random.default_rng(0).integers(0, 256, size=(12, 12), dtype=np.uint8)
    img = Image(array)
    orig = img.image.copy()

    result = img.contrast("clahe")
    expected = cv.createCLAHE().apply(orig)

    assert np.allclose(result.image, expected)
    assert result is not img
    assert result.image is not img.image
    assert np.array_equal(img.image, orig)
    assert result.data is not img.data
    assert result.meta is not img.meta


def test_image_contrast_fallback_returns_independent_copy(testimgpng):
    # "auto" (and any other unrecognized method) hits contrast()'s fallback, which mirrors
    # the pre-_with_data behavior of returning an unmodified but independent copy.
    orig = testimgpng.image.copy()

    result = testimgpng.contrast("auto")

    assert result is not testimgpng
    assert result.image is not testimgpng.image
    assert np.array_equal(result.image, orig)
    assert np.array_equal(testimgpng.image, orig)
    assert result.data is not testimgpng.data
    assert result.meta is not testimgpng.meta


def test_dataobject_with_data_independent_of_original():
    """Direct unit test for `DataObject._with_data`, independent of any specific caller: the
    exact invariant a prior bug broke (a plain `new._data = ...` was silently swallowed by
    `__setattr__`'s guard against replacing `_data`/`_meta` once already set, so the
    substituted array never actually took effect)."""
    original = Image(np.zeros((3, 3)))
    new_array = np.ones((3, 3))

    result = original._with_data(image=new_array)

    assert result.image is new_array
    assert result is not original
    assert result.data is not original.data
    assert result.meta is not original.meta
    assert np.array_equal(original.image, np.zeros((3, 3)))  # original untouched

    # mutating one's meta afterward must not affect the other
    result.meta["probe"] = "result"
    original.meta["probe"] = "original"
    assert result.meta["probe"] == "result"
    assert original.meta["probe"] == "original"


### Stack Arithmetic Tests


def test_stack_addsub_stack():
    sources = list(range(0, 10))
    sources_add = list(range(9, 12))

    stack = MinimalObjectStack(sources)
    stack_add = MinimalObjectStack(sources_add)

    comb_stack = stack + stack_add
    comb_sources = sources + sources_add

    for obj, source in zip(comb_stack, comb_sources):
        assert obj == MinimalObject(source)

    with pytest.raises(TypeError):
        comb_stack = stack - stack_add


# Further Tests have to be done with a Stack of a Class that implements calc itself


# @pytest.mark.parametrize("x", [Image(TESTDATA_DIR + TESTIMAGE_NAME + ".png"), 10, 10.0])
# @pytest.mark.parametrize(
#    "virtual", [False, pytest.param(True, marks=pytest.mark.xfail(raises=ValueError))]
# )
@pytest.mark.parametrize("x", [Image(TESTDATA_DIR + TESTIMAGE_NAME + ".png"), 10, 10.0])
@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_stack_calc_imgNumber(x, virtual, testimgpng):
    sources = [testimgpng * i for i in range(1, 3)]
    if virtual:
        with pytest.warns(UserWarning):
            stack = ImageStack(sources, virtual=virtual)
    else:
        stack = ImageStack(sources, virtual=virtual)
    stack_calc = stack + x

    for index, img in enumerate(stack_calc):
        assert img == sources[index] + x

    stack_calc = stack - x

    for index, img in enumerate(stack_calc):
        assert img == sources[index] - x

    stack_calc = stack / x
    for index, img in enumerate(stack_calc):
        assert img == sources[index] / x

    if isinstance(x, numbers.Number):
        stack_calc = stack * x
        for index, img in enumerate(stack_calc):
            assert img == sources[index] * x


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_stack_img_meansum(testimgpng, virtual):
    sources = [testimgpng * i for i in range(1, 4)]
    if virtual:
        with pytest.warns(UserWarning):
            stack = ImageStack(sources, virtual=virtual)
    else:
        stack = ImageStack(sources, virtual=virtual)

    # pylint: disable = no-member
    assert (np.mean(stack).image == TESTIMAGE * 2).all()
    assert (np.sum(stack).image == TESTIMAGE * 6).all()


### Test data saving and loading


def test_image_pickling(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.pickle")
    img = Image.load(TESTDATA_DIR + "test_image.pickle")
    os.remove(TESTDATA_DIR + "test_image.pickle")
    assert (img.image == TESTIMAGE).all()


def test_image_compression(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.pickle.bz2")
    img = Image.load(TESTDATA_DIR + "test_image.pickle.bz2")
    os.remove(TESTDATA_DIR + "test_image.pickle.bz2")
    assert (img.image == TESTIMAGE).all()


def test_image_save_json(testimgpng):
    testimgpng.save(TESTDATA_DIR + "test_image.json")
    img = Image.load(TESTDATA_DIR + "test_image.json")
    os.remove(TESTDATA_DIR + "test_image.json")
    assert (img.image == TESTIMAGE).all()


# @pytest.mark.parametrize(
#     "fileext",
#     [
#         ".thin.pickle",
#         pytest.param(".pickle", marks=pytest.mark.xfail(raises=AttributeError, strict = True)),
#     ],
# )
# def test_thin_obj_is_newly_generated(fileext, testimgpng):
#     testimgpng.save(TESTDATA_DIR + "test_image" + fileext)
#     Image.test = 1
#     img = Image.load(TESTDATA_DIR + "test_image" + fileext)
#     os.remove(TESTDATA_DIR + "test_image" + fileext)
#     assert img.test == 1


@pytest.mark.parametrize("fileext", [".pickle", ".pickle.bz2", ".json"])
def test_imagestack_save_full(fileext, testimgpng):
    stack = ImageStack([testimgpng, testimgpng])
    stack.save(TESTDATA_DIR + "test_image" + fileext)
    load_stack = Image.load(TESTDATA_DIR + "test_image" + fileext)
    os.remove(TESTDATA_DIR + "test_image" + fileext)
    assert len(stack) == len(load_stack)
    assert all(
        [(img1.image == img2.image).all for img1, img2 in zip(stack, load_stack)]
    )


@pytest.mark.parametrize("fileext", [".png", ".tiff"])
def test_image_save(testimgpng, fileext):
    testimgpng.save(TESTDATA_DIR + "test_image" + fileext)
    img = Image(TESTDATA_DIR + "test_image" + fileext)
    os.remove(TESTDATA_DIR + "test_image" + fileext)
    assert (img.image == testimgpng.image).all()


def test_stack_save():
    # array = [np.random.rand(10,10) for i in np.linspace(0,10,10)]
    array = [np.eye(10) * i for i in np.linspace(0, 65000, 10)]
    stack = ImageStack(array)
    stack.save(TESTDATA_DIR + "test_image.tiff")
    load_stack = ImageStack(TESTDATA_DIR + "test_image.tiff")
    os.remove(TESTDATA_DIR + "test_image.tiff")
    assert all([(img1.image == img2).all() for img1, img2 in zip(load_stack, array)])


def test_imagestack_average_nonpositive_avg_raises():
    """`avg <= 0` must raise a clear error rather than a misleading one. Before the empty-stack
    constructor fix, `stack.average(-2)` on a 10-frame stack passed `10 % -2 == 0`, iterated
    `range(0, -5)` zero times, and crashed building `type(self)([])` with a bare IndexError;
    after that fix it would instead report "cannot create an empty stack", which points at the
    wrong thing -- the input average, not the (non-empty) stack, is what's actually bad. `avg=0`
    hit a bare ZeroDivisionError before; both must now raise ValueError."""
    stack = ImageStack([np.eye(4) * i for i in range(10)])
    with pytest.raises(ValueError):
        stack.average(0)
    with pytest.raises(ValueError):
        stack.average(-2)


def test_imagestack_save_tiff_empty_stack_raises(tmp_path):
    stack = ImageStack([np.eye(4) * i for i in range(3)])
    del stack[:]
    assert len(stack) == 0
    target = tmp_path / "empty.tiff"
    with pytest.raises(ValueError):
        stack.save(str(target))
    assert not target.exists()


### Loading a stack from per-frame ".tiff" files (glob / folder), as opposed to a single 3D
### TIFF -- see uspy.dataobject.image.list_image_files. Reproduces a reported bug where
### globbing per-frame TIFFs silently misread frames (wrong pixels ended up at several
### indices) via tifffile's FileSequence machinery, and lost each element's `source`.


def test_imagestack_glob_loads_one_frame_per_file_in_natural_order(tiff_frame_stack):
    stack = ImageStack(tiff_frame_stack["glob"])
    assert len(stack) == len(tiff_frame_stack["paths"])
    assert stack.sources == tiff_frame_stack["paths"]
    for element, expected in zip(stack, tiff_frame_stack["arrays"]):
        assert element.image.shape == expected.shape
        np.testing.assert_array_equal(element.image, expected)


def test_imagestack_glob_folder_and_explicit_list_agree(tiff_frame_stack):
    from_glob = ImageStack(tiff_frame_stack["glob"])
    from_dir = ImageStack(tiff_frame_stack["dir"])
    from_dir_trailing_slash = ImageStack(tiff_frame_stack["dir"] + "/")
    from_list = ImageStack(tiff_frame_stack["paths"])

    assert from_glob.sources == from_dir.sources == from_dir_trailing_slash.sources
    assert from_glob.sources == from_list.sources
    for a, b in zip(from_glob, from_list):
        np.testing.assert_array_equal(a.image, b.image)


def test_imagestack_glob_can_virtualize(tiff_frame_stack):
    # Before the fix, a glob-built stack lost every element's `source` (parsed straight out
    # of a 3D array returned by tifffile), so virtualizing raised "Can't virtualize stack
    # without source information."
    stack = ImageStack(tiff_frame_stack["glob"])
    stack.virtual = True
    assert stack.virtual
    assert len(stack) == len(tiff_frame_stack["paths"])
    stack.virtual = False
    assert not stack.virtual


@pytest.mark.parametrize("source", ARRAY_2D)
def test_line_generation(source):
    line = Line(source)
    assert_array_equal(line.dataframe.to_numpy(), source.T)
    assert line.xdim == "x"
    assert line.ydim == "y"


# @pytest.mark.xfail(raises = ValueError, strict = True)
@pytest.mark.parametrize("source", ARRAY_1D)
def test_line_incompatible_generation(source):
    with pytest.raises(ValueError):
        Line(source)


@pytest.mark.parametrize("source", ARRAY_2D)
def test_line_xy_alias(source):
    line = Line(source)
    line.xdim = "a"
    line.ydim = "b"
    # assert (line.a == source[0,:]).all()
    # assert (line.b == source[1,:]).all()
    assert_array_equal(line.a, source[0, :])
    assert_array_equal(line.b, source[1, :])


@pytest.mark.parametrize("source", ARRAY_2D)
def test_line_saveload_csv(source):
    line = Line(source)
    line.xdim = "a"
    line.ydim = "b"
    line.save(TESTDATA_DIR + "test.csv")
    line2 = Line(TESTDATA_DIR + "test.csv")
    # assert line.dataframe.equals(line2.dataframe)
    pd.testing.assert_frame_equal(line.dataframe, line2.dataframe)


def test_line_interpolate(quad_line):
    line = quad_line
    np.testing.assert_almost_equal(line.interpolate(line.x), line.y)
    assert line.interpolate(1.0).size == 1


def test_line_integration(quad_line):
    integral = quad_line.integral()
    assert 10 ** 3 / 3 - 10 ** -5 < integral.interpolate(10) < 10 ** 3 / 3 + 10 ** -5


def test_line_area(quad_line):
    assert quad_line.area == 1000 / 3


def test_line_derivative(quad_line):
    diff = quad_line.derivative()
    assert diff.length == quad_line.length - 1
    assert_array_equal(diff.x, quad_line.x[1:])

    # derivative of x^2 is 2x
    dydx = 2 * quad_line.x[1:]
    assert_almost_equal(dydx, diff.y)


def test_line_calc(quad_line):
    # calc with Numbers
    assert_array_equal((2 * quad_line).y, 2 * quad_line.y)
    assert_array_equal((quad_line / 2).y, quad_line.y / 2)
    assert_array_equal((quad_line + 2).y, quad_line.y + 2)
    assert_array_equal((quad_line - 2).y, quad_line.y - 2)
    # calc with Lines
    assert_array_equal((quad_line + quad_line).y, 2 * quad_line.y)
    assert_array_equal((quad_line - quad_line).y, np.zeros(quad_line.length))
    assert_array_equal((quad_line * quad_line).y, quad_line.y ** 2)
    assert_array_equal(
        np.nan_to_num((quad_line / quad_line).y, nan=1), np.ones(quad_line.length)
    )
