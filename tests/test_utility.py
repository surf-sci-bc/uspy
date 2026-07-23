"""Tests the uspy.leem.utility module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

from uspy.leem import utility
from uspy.leem import base


def test_imgify(img, img_fname):
    assert isinstance(utility.imgify(img), base.LEEMImg)
    assert isinstance(utility.imgify(img_fname), base.LEEMImg)

def test_imgify_returns_same_object_unchanged(img):
    # imgify must be idempotent: passing an already-constructed LEEMImg should return
    # it as-is, not a copy or a re-parsed instance.
    assert utility.imgify(img) is img

def test_stackify(stack, stack_fname):
    assert isinstance(utility.stackify(stack), base.LEEMStack)
    assert isinstance(utility.stackify(stack_fname), base.LEEMStack)

def test_stackify_returns_same_object_unchanged(stack):
    assert utility.stackify(stack) is stack

def test_stackify_wraps_single_image(img):
    wrapped = utility.stackify(img)
    assert isinstance(wrapped, base.LEEMStack)
    assert len(wrapped) == 1
    assert wrapped[0] == img

def test_stackify_honors_virtual(stack_folder):
    virtual_stack = utility.stackify(stack_folder, virtual=True)
    assert virtual_stack.virtual is True
    real_stack = utility.stackify(stack_folder, virtual=False)
    assert real_stack.virtual is False

def test_timing_notification(capfd):
    @utility.timing_notification("test")
    def wrapped(arg):
        out, _err = capfd.readouterr()
        assert "test" in out
        print(arg)
    wrapped(5)
    out, _err = capfd.readouterr()
    assert "Finished" in out

def test_progress_bar_class(capfd):
    progbar = utility.ProgressBar(10, suffix="test_bar", length=10)
    assert not progbar.started
    assert not progbar.finished
    for i in range(1, 6):
        progbar.increment(2)
        assert progbar.iteration == i * 2
        out, _err = capfd.readouterr()
        if i < 5:
            assert progbar.unfill in out
            assert not progbar.finished
        assert progbar.fill in out
        assert progbar.started
    assert progbar.finished
    assert "100" in out

def test_progress_bar_function(capfd):
    """Regression test: uspy.leem.utility.progress_bar used to reference an
    uninitialized `share` variable and raise NameError on the first iteration. It now
    delegates to uspy.utility.progress_bar."""
    assert list(utility.progress_bar(range(3))) == [0, 1, 2]
    out, _err = capfd.readouterr()
    assert "moved to uspy.utility" in out

def test_progress_bar_function_silent(capfd):
    assert list(utility.progress_bar(range(3), silent=True)) == [0, 1, 2]

def test_progress_bar_function_empty():
    assert list(utility.progress_bar([])) == []
