# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments
# pylint: disable=line-too-long

from shutil import copyfile

import pytest
import numpy as np

from agfalta.leem import ivdb

# ivbak.sqlite is the test database
# iv.sqlite is working database
# iv.sqlite is replaced with ivbak.sqlite before every test

DB_PATH = '../testdata/iv.sqlite'
DB_PATH_BAK = '../testdata/ivbak.sqlite'

@pytest.fixture(autouse=True)
def run_around_tests():
    copyfile(DB_PATH_BAK, DB_PATH)

    #yield
    # Code that will run after your test, for example:
    #files_after = # ... do something to check the existing files
    #assert files_before == files_after

def test_database():
    #revertDatabase()

    db = ivdb.IVDatabase(DB_PATH)
    ivc = ivdb.IVCurve(db = db, id_ = 1)
    assert ivc.name == "g/S/Ru(0001)"
    assert ivc.material == "Graphene"
    assert ivc.substrate == "Ru(0001)"
    assert len(ivc.tags) > 0

    ivc = ivdb.IVCurve(db = db, id_ = None)
    assert ivc.name is None
    assert ivc.substrate is None
    assert ivc.material is None

def test_readonly():
    db = ivdb.IVDatabase(DB_PATH)
    ivc = ivdb.IVCurve(db = db, id_ = 1, readonly=True)
    with pytest.raises(AttributeError):
        ivc.substrate_id = 2
    with pytest.raises(AttributeError):
        ivc.material_id = 2
    with pytest.raises(AttributeError):
        ivc.data = np.array([[1,2],[3,4]])
    with pytest.raises(AttributeError):
        ivc.name = "Test"
    with pytest.raises(AttributeError):
        ivc.tag_ids = [1,2]
    with pytest.raises(AttributeError):
        ivc.comment = "Test"
    with pytest.raises(AttributeError):
        ivc.source = "Test"

    ## Test Properties that can never be written
    with pytest.raises(AttributeError):
        ivc.substrate = None
    with pytest.raises(AttributeError):
        ivc.material = None
    with pytest.raises(AttributeError):
        ivc.tags = None

    #revertDatabase()

def test_writable():
    db = ivdb.IVDatabase(DB_PATH)
    ivc = ivdb.IVCurve(db = db, id_ = 1, readonly = False)

    ivc.substrate_id = 3
    ivc.material_id = 2
    ivc.data = np.array([[1,2],[3,4]])
    ivc.name = "Test"
    ivc.tag_ids = [1, 2, 3]
    ivc.comment = "Test"
    ivc.source = "Test"

    assert ivc.substrate_id == 3
    assert ivc.material_id == 2
    assert np.array_equiv(ivc.data, np.array([[1,2],[3,4]]))
    assert ivc.name == "Test"
    assert ivc.comment == "Test"
    assert ivc.source == "Test"
    assert ivc.tag_ids == [1, 2, 3]

    # Revert Test
    #revertDatabase()

def test_tags():
    db = ivdb.IVDatabase(DB_PATH)
    ivc = ivdb.IVCurve(db = db, id_ = 1, readonly = False)
    assert ivc.tags == ['Bilayer', 'Multilayer', 'Oxide']
    ivc.tag_ids = [2,4]
    assert ivc.tags == ['Multilayer', 'Intercalation']

def test_new_curve():
    db = ivdb.IVDatabase(DB_PATH)
    ivc = ivdb.IVCurve(db = db, readonly = False)
    ivc.new_curve(name = "Test", substrate_id = 1, material_id = 1, source = "Test", data = np.array([[1,2],[3,4]]), comment = "Insert Test", tag_ids=[1,2])
    assert ivc.name == "Test"
    assert ivc.substrate_id == 1
    assert ivc.material_id == 1
    assert ivc.source == "Test"
    assert np.array_equal(ivc.data, np.array([[1,2],[3,4]]))
    assert ivc.comment == "Insert Test"
    assert ivc.tag_ids == [1,2]
    assert ivc.tags == ['Bilayer', 'Multilayer']

def test_np2json():
    db = ivdb.IVDatabase(DB_PATH)
    ivc = ivdb.IVCurve(db = db, id_ = 1)

    assert np.array_equal(ivc.data, np.array([[1,2,3,4],[9,8,7,6]]))
    assert ivc.np2json(ivc.data) == "[[1, 2, 3, 4], [9, 8, 7, 6]]"
