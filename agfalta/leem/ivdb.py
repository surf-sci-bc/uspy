import sqlite3
from contextlib import closing


class iv_database():
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

class iv_curve():

    """
    Substrate, Name, Adlayer, Measurement
    """

    def __init__(self, db, id = None, readonly = True):
        self.readonly = readonly
        self.db = db
        self._id = id
        #if id is not None:
            #self._id = id
            #self.db = db
            #self.c = self.db.conn.cursor()

            ## Get Information
            # self.c.execute("""SELECT Measurement.Name, Substrate.Name, Material.Name, Measurement.Data, Measurement.Comment 
            #                 FROM Measurement INNER JOIN Substrate ON Measurement.Substrate_Id = Substrate.Id 
            #                 INNER JOIN Material ON Measurement.Material_Id = Material.Id WHERE Measurement.Id=(?)""", (self._id,))
            #query = """SELECT Name, SubstrateId, MaterialId, Data, Comment, Source FROM Measurement WHERE Id = (?) """
            #(self._name, self._substrateId, self._materialId, self._data, self._comment, self._source) = self.db.query(query, (self._id,))[0]

            ## Get Tags
            #query = """SELECT Tags.Name FROM MeasurementTags INNER JOIN Tags ON MeasurementTags.TagsId = Tags.Id WHERE MeasurementId = (?)"""
            #query = """SELECT Tags.Id FROM MeasurementTags INNER JOIN Tags ON MeasurementTags.TagsId = Tags.Id WHERE MeasurementId = (?)"""
            #self._tagIds = self.db.query(query, (self._id,))
        # if id is None:
        #     for arg in kwargs:
        #         print(arg)
            #self.newCurve(kwargs)
            #self._id, self._name, self._substrateId, self._materialId, self._data, self._comment, self._tagIds = None, None, None, None, None, None, None
    
    attrs = { #var name : db name
        "substrateId": "SubstrateId", # Attributes from Measurement
        "materialId": "MaterialId",
        "name": "Name",
        "source": "Source",
        "data": "Data",
        "comment": "Comment",
        "substrate": ["Substrate", "substrateId"],
        "material": ["Material", "materialId"],
        "tagIds": "Id",
        "tags": "Name"
    }
 
    def __getattr__(self, attr):
        if self._id is not None:
            if attr in ["substrateId", "materialId", "source", "data", "comment", "name"]:
                return self.db.query(f"SELECT {self.attrs[attr]} FROM Measurement WHERE Id=(?)", (self._id, ))[0][0]
            if attr in ["substrate", "material"]:
                return self.db.query(f"SELECT Name FROM {self.attrs[attr][0]} WHERE Id=(?)", (getattr(self, self.attrs[attr][1]), ))[0][0]
            if attr in ["tagIds", "tags"]:
                query = f"SELECT Tags. {self.attrs[attr]} FROM MeasurementTags INNER JOIN Tags ON MeasurementTags.TagsId = Tags.Id WHERE MeasurementId = (?)"
                print("----------------------------")
                print([self.db.query(query, (self._id, ))])
                currentTags = self.db.query(query, (self._id, ))
                if isinstance(currentTags,int):
                    return []
                else:
                    return [tag[0] for tag in self.db.query(query, (self._id, ))]
        else:
            return None
    
    def __setattr__(self, attr, value):

        if attr in ("substrateId", "materialId", "source", "data", "comment", "name"):
            if not self.readonly:
                self.updateDB(**{attr: value})
            else:
                raise AttributeError("Readonly")
        elif attr in ("tagIds",):
            if not self.readonly:
                self.updateDB(**{attr: (getattr(self, attr), value)})
            else:
                raise AttributeError("Readonly")
        elif attr in ("substrate", "material", "tags"):
            raise AttributeError("Readonly")
        else:
            self.__dict__[attr] = value
    
    def updateDB(self, name = None, substrateId = None, materialId = None, source = None, data = None, comment = None, tagIds = None):
        
        name = [self.name, name][name is not None]
        substrateId = [self.substrateId, substrateId][substrateId is not None]
        materialId = [self.materialId, materialId][materialId is not None]
        source = [self.source, source][source is not None]
        data = [self.data, data][data is not None]
        comment = [self.comment, comment][comment is not None]
        if tagIds is not None:
            newtagIds = tagIds[1]
            oldtagIds = tagIds[0]
        else:
            newtagIds, oldtagIds = None, None
        print(tagIds)

        print(name,substrateId,materialId,source,data,comment)
        print(f"Id: {self._id}")
        # if self._id is None:
        #     query = "INSERT INTO Measurement (Name, SubstrateId, MaterialId, Source, Data, Comment) VALUES (?,?,?,?,?,?)"
        #     rowid = self.db.query(query, (name, substrateId, materialId, source, data, comment))
        #     query = "SELECT Id FROM Measurement WHERE ROWID = (?)"
        #     self._id = self.db.query(query,(rowid,))[0][0]
        #     print(self._id)
        #else:
        if any([name, substrateId, materialId, source, data, comment]):
            query = "UPDATE Measurement SET Name = (?), SubstrateId = (?), MaterialId = (?), Source = (?), Data = (?), Comment = (?) WHERE Id = (?)"
            self.db.query(query, (name, substrateId, materialId, source, data, comment, self._id))
        
        if tagIds is not None:
            for tagId in newtagIds:
                query = "INSERT OR IGNORE INTO MeasurementTags (TagsId, MeasurementId) VALUES (?, ?)"
                self.db.query(query, (tagId, self._id))
            for tagId in oldtagIds:
                if tagId not in newtagIds:
                    query = "DELETE FROM MeasurementTags WHERE TagsId = (?) AND MeasurementId = (?)"
                    self.db.query(query, (tagId, self._id))

            
    #def newCurve(self, name = None, substrateId = None, materialId = None, source = None, data = None, comment = None):
    def newCurve(self, name, substrateId, materialId, source, data, comment = None, tagIds = None):
        print([name, substrateId, materialId, source, data, comment])
        if None in [name, substrateId, materialId, source, data, comment]:
            raise AttributeError("Values must not be None")
        else:
            # Update Measurement
            query = "INSERT INTO Measurement (Name, SubstrateId, MaterialId, Source, Data, Comment) VALUES (?,?,?,?,?,?)"
            rowid = self.db.query(query, (name, substrateId, materialId, source, data, comment))
            query = "SELECT Id FROM Measurement WHERE ROWID = (?)"
            self._id = self.db.query(query,(rowid,))[0][0]
            print(f"New ID {self._id}")
            
            self.tagIds = tagIds
            

        




    

    
    #@substrate.setter
    #def substrate()


#db = iv_database('/Users/larsbuss/Projects/adminer/iv.sqlite')
# ivc = iv_curve(db = db, id = 1, readonly=False)
# print(ivc.tags)
# ivc.tagIds = [1,2,3]
# print(ivc.tags)

#ivc = iv_curve(db = db, readonly=False)
#ivc.newCurve(name = "Test", substrateId = 1, materialId = 1, source = "Test", data = "[1,2,3,4,5,6,7,8,9,10]")



#ivc.writeToDB(name = 1, data = "[1,2,3,4,5,6,7,8,9,10]")


#print(ivc.tags)
