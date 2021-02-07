import pytest
import sqlite3
from shutil import copyfile
import numpy as np

from agfalta.leem import ivdb

# def revertDatabase():
#     db = ivdb.iv_database('/Users/larsbuss/Projects/adminer/iv.sqlite')
#     ivc = ivdb.iv_curve(db = db, id = 1, readonly = False)
#     ivc.substrateId = 1
#     ivc.materialId = 1
#     ivc.data = "[1,2,3,4,5,6,7,8,9,10]"
#     ivc.name = "g/S/Ru(0001)"
#     ivc.tagIds = ["1","2"]
#     ivc.comment = "This is test data"
#     ivc.source = "20210206_IV01"

db_path = '/Users/larsbuss/Projects/adminer/iv.sqlite'
db_path_bak = '/Users/larsbuss/Projects/adminer/ivbak.sqlite'

@pytest.fixture(autouse=True)
def run_around_tests():
    copyfile(db_path_bak, db_path)

    #yield
    # Code that will run after your test, for example:
    #files_after = # ... do something to check the existing files
    #assert files_before == files_after

def test_database():
    #revertDatabase()

    db = ivdb.iv_database(db_path)
    ivc = ivdb.iv_curve(db = db, id = 1)
    assert ivc.name == "g/S/Ru(0001)"
    assert ivc.material == "Graphene"
    assert ivc.substrate == "Ru(0001)"
    assert len(ivc.tags) > 0

    ivc = ivdb.iv_curve(db = db, id = None)
    assert ivc.name is None
    assert ivc.substrate is None
    assert ivc.material is None

def test_readonly():
    db = ivdb.iv_database(db_path)
    ivc = ivdb.iv_curve(db = db, id = 1, readonly=True)
    with pytest.raises(AttributeError):
        ivc.substrateId = 2
    with pytest.raises(AttributeError):
        ivc.materialId = 2
    with pytest.raises(AttributeError):
        ivc.data = np.array([[1,2],[3,4]])
    with pytest.raises(AttributeError):
        ivc.name = "Test"
    with pytest.raises(AttributeError):
        ivc.tagIds = [1,2]
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
    db = ivdb.iv_database(db_path)
    ivc = ivdb.iv_curve(db = db, id = 1, readonly = False)
    
    ivc.substrateId = 3
    ivc.materialId = 2
    ivc.data = np.array([[1,2],[3,4]])
    ivc.name = "Test"
    ivc.tagIds = [1, 2, 3]
    ivc.comment = "Test"
    ivc.source = "Test"

    assert ivc.substrateId == 3
    assert ivc.materialId == 2
    assert np.array_equiv(ivc.data, np.array([[1,2],[3,4]]))
    assert ivc.name == "Test"
    assert ivc.comment == "Test"
    assert ivc.source == "Test"
    assert ivc.tagIds == [1, 2, 3]

    # Revert Test
    #revertDatabase()

def test_tags():
    db = ivdb.iv_database(db_path)
    ivc = ivdb.iv_curve(db = db, id = 1, readonly = False)
    assert ivc.tags == ['Bilayer', 'Multilayer', 'Oxide']
    ivc.tagIds = [2,4]
    assert ivc.tags == ['Multilayer', 'Intercalation']

def test_newCurve():
    db = ivdb.iv_database(db_path)
    ivc = ivdb.iv_curve(db = db, readonly = False)
    ivc.newCurve(name = "Test", substrateId = 1, materialId = 1, source = "Test", data = np.array([[1,2],[3,4]]), comment = "Insert Test", tagIds=[1,2])
    assert ivc.name == "Test"
    assert ivc.substrateId == 1
    assert ivc.materialId == 1
    assert ivc.source == "Test"
    assert np.array_equal(ivc.data, np.array([[1,2],[3,4]]))
    assert ivc.comment == "Insert Test"
    assert ivc.tagIds == [1,2]
    assert ivc.tags == ['Bilayer', 'Multilayer']

def test_np2json():
    db = ivdb.iv_database(db_path)
    ivc = ivdb.iv_curve(db = db, id = 1)
    print(ivc.data)
    

    assert np.array_equal(ivc.data, np.array([[1,2,3,4],[9,8,7,6]]))
    assert ivc.np2json(ivc.data) == "[[1, 2, 3, 4], [9, 8, 7, 6]]"