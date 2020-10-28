"""Tests the agfalta.leem.driftnorm module."""
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

from agfalta.leem import utility
from agfalta.leem import base


def test_try_load_img(img, img_fname):
    assert isinstance(utility.try_load_img(img), base.LEEMImg)
    assert isinstance(utility.try_load_img(img_fname), base.LEEMImg)

def test_try_load_stack(stack, stack_fname):
    assert isinstance(utility.try_load_stack(stack), base.LEEMStack)
    assert isinstance(utility.try_load_stack(stack_fname), base.LEEMStack)

def test_timing_notification(capfd):
    @utility.timing_notification("test")
    def wrapped(arg):
        out, _err = capfd.readouterr()
        assert "test" in out
        print(arg)
    wrapped(5)
    out, _err = capfd.readouterr()
    assert "Finished" in out

def test_progress_bar(capfd):
    pass
