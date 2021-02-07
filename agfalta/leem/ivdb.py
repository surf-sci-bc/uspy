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

        if id is not None:
            self._id = id
            self.db = db
            #self.c = self.db.conn.cursor()

            ## Get Information
            # self.c.execute("""SELECT Measurement.Name, Substrate.Name, Material.Name, Measurement.Data, Measurement.Comment 
            #                 FROM Measurement INNER JOIN Substrate ON Measurement.Substrate_Id = Substrate.Id 
            #                 INNER JOIN Material ON Measurement.Material_Id = Material.Id WHERE Measurement.Id=(?)""", (self._id,))
            query = """SELECT Name, SubstrateId, MaterialId, Data, Comment, Source FROM Measurement WHERE Id = (?) """
            (self._name, self._substrateId, self._materialId, self._data, self._comment, self._source) = self.db.query(query, (self._id,))[0]

            ## Get Tags
            #query = """SELECT Tags.Name FROM MeasurementTags INNER JOIN Tags ON MeasurementTags.TagsId = Tags.Id WHERE MeasurementId = (?)"""
            query = """SELECT Tags.Id FROM MeasurementTags INNER JOIN Tags ON MeasurementTags.TagsId = Tags.Id WHERE MeasurementId = (?)"""
            self._tagIds = self.db.query(query, (self._id,))
        else:
            self._id, self._name, self._substrateId, self._materialId, self._data, self._comment, self._tagIds = None, None, None, None, None, None, None

    ## Name of the Substrate
    @property
    def substrate(self):
        if self._substrateId is not None:
            return self.db.query("SELECT Name FROM Substrate WHERE Id=(?)", (self._substrateId, ))[0][0]
        else:
            return None

    ## Name of the Material
    @property
    def material(self):
        if self._materialId is not None:
            return self.db.query("SELECT Name FROM Material WHERE Id=(?)", (self._materialId, ))[0][0]
        else:
            return None

    ## Names of Tags
    @property
    def tags(self):
        if self._tagIds is not None:
            query = """SELECT Tags.Name FROM MeasurementTags INNER JOIN Tags ON MeasurementTags.TagsId = Tags.Id WHERE MeasurementId = (?)"""
            tags = [tag[0] for tag in self.db.query(query, (self._id, ))]
            return tags
        else:
            return None

    @property
    def substrateId(self):
        return self._substrateId
    
    @substrateId.setter
    def substrateId(self, newSubstrateId):
        if not self.readonly:
            self._substrateId = newSubstrateId
        else:
            raise AttributeError("Readonly")

    @property
    def materialId(self):
        return self._materialId
    
    @materialId.setter
    def materialId(self, newMaterialId):
        if not self.readonly:
            self._materialId = newMaterialId
        else:
            raise AttributeError("Readonly")
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, newData):
        if not self.readonly:
            self._data = newData
        else:
            raise AttributeError("Readonly")

    @property
    def tagIds(self):
        return self._tagIds
    
    @tagIds.setter
    def tagIds(self, newTagIds):
        if not self.readonly:
            self._tagIds = newTagIds
        else:
            raise AttributeError("Readonly")

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, newName):
        if not self.readonly:
            self._name = newName
        else:
            raise AttributeError("Readonly")
    
    def writeToDB(self, name = None, substrateId = None, materialId = None, source = None, data = None, comment = None):
        
        name = [self._name, name][name is not None]
        substrateId = [self._substrateId, substrateId][substrateId is not None]
        materialId = [self._materialId, materialId][materialId is not None]
        source = [self._source, source][source is not None]
        data = [self._data, data][data is not None]
        comment = [self._comment, comment][comment is not None]

        print(name,substrateId,materialId,source,data,comment)
        if self._id is not None:
            query = "INSERT INTO Measurement (Name, SubstrateId, MaterialId, Source, Data, Comment) VALUES (?,?,?,?,?,?)"
            print(db.query(query, (name, substrateId, materialId, source, data, comment)))
            
        




    

    
    #@substrate.setter
    #def substrate()

db = iv_database('/Users/larsbuss/Projects/adminer/iv.sqlite')
ivc = iv_curve(db = db,id = 1)
ivc.writeToDB(name = 1, data = "[1,2,3,4,5,6,7,8,9,10]")


#print(ivc.tags)
