# pylint: disable=attribute-defined-outside-init
# pylint: disable=missing-docstring
# pylint: disable=too-many-arguments
# pylint: disable=line-too-long

from contextlib import closing

import sqlite3
import json
import numpy as np


class IVDatabase():
    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
        except:
            print("Can`t connect to Database")

    def query(self, sql, values):
        with closing(sqlite3.connect(self.db_file)) as con, con,  \
                closing(con.cursor()) as cur:
            con.set_trace_callback(print)
            cur.execute(sql,values)
            result = cur.fetchall()
            if len(result)==0:
                return cur.lastrowid
            else:
                return result

class IVCurve():

    """
    Substrate, Name, Adlayer, Measurement
    """

    def __init__(self, db, id_ = None, readonly = True):
        self.readonly = readonly
        self.db = db
        self._id = id_

    attrs = { #var name : db name
        "substrate_id": "SubstrateId", # Attributes from Measurement
        "material_id": "MaterialId",
        "name": "Name",
        "source": "Source",
        "data": "Data",
        "comment": "Comment",
        "substrate": ["Substrate", "substrate_id"],
        "material": ["Material", "material_id"],
        "tag_ids": "Id",
        "tags": "Name"
    }

    def __getattr__(self, attr):
        if self._id is not None:
            if attr in ["substrate_id", "material_id", "source", "comment", "name"]:
                return self.db.query(
                    f"SELECT {self.attrs[attr]} FROM Measurement WHERE Id=(?)",
                    (self._id, ))[0][0]
            if attr in ["data"]:
                return self.json2np(self.db.query(
                    f"SELECT {self.attrs[attr]} FROM Measurement WHERE Id=(?)",
                    (self._id, ))[0][0])
            if attr in ["substrate", "material"]:
                return self.db.query(
                    f"SELECT Name FROM {self.attrs[attr][0]} WHERE Id=(?)",
                    (getattr(self, self.attrs[attr][1]), ))[0][0]
            if attr in ["tag_ids", "tags"]:
                query = f"SELECT Tags. {self.attrs[attr]} FROM MeasurementTags " \
                + "INNER JOIN Tags ON MeasurementTags.TagsId = Tags.Id WHERE MeasurementId = (?)"""
                current_tags = self.db.query(query, (self._id, ))
                if isinstance(current_tags,int):
                    return []
                return [tag[0] for tag in self.db.query(query, (self._id, ))]

        return None

    def __setattr__(self, attr, value):

        if attr in ("substrate_id", "material_id", "source", "data", "comment", "name"):
            if not self.readonly:
                self.update_db(**{attr: value})
            else:
                raise AttributeError("Readonly")
        elif attr in ("tag_ids",):
            if not self.readonly:
                self.update_db(**{attr: (getattr(self, attr), value)})
            else:
                raise AttributeError("Readonly")
        elif attr in ("substrate", "material", "tags"):
            raise AttributeError("Readonly")
        else:
            self.__dict__[attr] = value


    def update_db(self, name = None, substrate_id = None, material_id = None, source = None, data = None, comment = None, tag_ids = None):

        name = [self.name, name][name is not None]
        substrate_id = [self.substrate_id, substrate_id][substrate_id is not None]
        material_id = [self.material_id, material_id][material_id is not None]
        source = [self.source, source][source is not None]
        data = [self.data, data][data is not None]
        comment = [self.comment, comment][comment is not None]
        if tag_ids is not None:
            new_tag_ids = tag_ids[1]
            old_tag_ids = tag_ids[0]
        else:
            new_tag_ids, old_tag_ids = None, None

        if any([name, substrate_id, material_id, source, self.valid_data(data), comment]):
            query = "UPDATE Measurement SET Name = (?), SubstrateId = (?), MaterialId = (?), Source = (?), Data = (?), Comment = (?) WHERE Id = (?)"
            self.db.query(query, (name, substrate_id, material_id, source, self.np2json(data), comment, self._id))

        if tag_ids is not None:
            for tag_id in new_tag_ids:
                query = "INSERT OR IGNORE INTO MeasurementTags (TagsId, MeasurementId) VALUES (?, ?)"
                self.db.query(query, (tag_id, self._id))
            for tag_id in old_tag_ids:
                if tag_id not in new_tag_ids:
                    query = "DELETE FROM MeasurementTags WHERE TagsId = (?) AND MeasurementId = (?)"
                    self.db.query(query, (tag_id, self._id))

    def new_curve(self, name, substrate_id, material_id, source, data, comment = None, tag_ids = None):
        if None in [name, substrate_id, material_id, source]:
            raise AttributeError("Values must not be None")
        elif not self.valid_data(data):
            raise TypeError(f"Data must be Numpy Array. Data is of type {type(data)}")
        else:
            # Insert Measurement
            query = "INSERT INTO Measurement (Name, SubstrateId, MaterialId, Source, Data, Comment) VALUES (?,?,?,?,?,?)"
            rowid = self.db.query(query, (name, substrate_id, material_id, source, self.np2json(data), comment))
            query = "SELECT Id FROM Measurement WHERE ROWID = (?)"
            self._id = self.db.query(query,(rowid,))[0][0]
            self.tag_ids = tag_ids

    def np2json(self, data):
        return json.dumps(data.tolist())

    def json2np(self, data):
        return np.asarray(json.loads(data))

    def valid_data(self, data):
        return isinstance(data, np.ndarray)
