import pytest
import sqlite3

from agfalta.leem import ivdb

def test_database():
    db = ivdb.iv_database('/Users/larsbuss/Projects/adminer/iv.sqlite')
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
    db = ivdb.iv_database('/Users/larsbuss/Projects/adminer/iv.sqlite')
    ivc = ivdb.iv_curve(db = db, id = 1, readonly=True)
    with pytest.raises(AttributeError):
        ivc.substrateId = None
        ivc.materialId = None
        ivc.data = None
        ivc.name = None
        ivc.tags = None
        ivc.comment = None
        ivc.source = None
        ## Test Properties that can never be written
        ivc.substrate = None
        ivc.material = None
        ivc.tags = None

def test_writable():
    db = ivdb.iv_database('/Users/larsbuss/Projects/adminer/iv.sqlite')
    ivc = ivdb.iv_curve(db = db, id = 1, readonly=False)
    
    ivc.substrateId = None
    ivc.materialId = None
    ivc.data = None
    ivc.name = None
    ivc.tagIds = None
    ivc.comment = None
    ivc.source = None

    assert ivc.substrateId is None
    assert ivc.materialId is None
    assert ivc.data is None
    assert ivc.name is None
    assert ivc.comment is None
    assert ivc.source is None
    assert ivc.tagIds is None

def test_tags():
    db = ivdb.iv_database('/Users/larsbuss/Projects/adminer/iv.sqlite')
    ivc = ivdb.iv_curve(db = db, id = 1, readonly=True)
    assert ivc.tags == ['Bilayer', 'Multilayer', 'Intercalation']
    