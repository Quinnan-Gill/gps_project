from datetime import datetime
import os
from requests.api import request
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import BigInteger
from sqlalchemy.sql.visitors import CloningExternalTraversal

basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data/app.db')

# ----- This is related code -----
engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)
Base = declarative_base()
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
Session.configure(bind=engine)
session = Session()
# ----- This is related code -----

class ViresRequests(Base):
    __tablename__ = "vires_requests"

    request_id = Column(Integer, primary_key=True)
    maxfiles = Column(Integer)
    position = Column(Integer)
    satelite = Column(String)
    measurement = Column(String)
    url_used = Column(String)
    url_hash = Column(String)
    overall_processed = Column(Boolean)

class ViresMetaData(Base):
    __tablename__ = "vires_meta_data"

    meta_id = Column(Integer, primary_key=True)
    zip_file = Column(String)
    zip_hash = Column(String)
    measurement = Column(String)
    processed = Column(Boolean)

    start_time = Column(DateTime)
    end_time = Column(DateTime)

    request_id = Column(Integer, ForeignKey('vires_requests.request_id'))

class DataMeasurement(Base):
    __tablename__ = "data_measurement"
    
    measurement_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, primary_key=True)
    
    """
    TEC Data
    """
    gps_position1 = Column(Float)
    gps_position2 = Column(Float)
    gps_position3 = Column(Float)

    leo_position1 = Column(Float)
    leo_position2 = Column(Float)
    leo_position3 = Column(Float)

    prn = Column(Float)
    l1 = Column(Float)
    l2 = Column(Float)

    """
    IBI Data
    """

    bubble_index = Column(Integer)

    """
    Relationships
    """
    meta_id = Column(Integer, ForeignKey("vires_meta_data.meta_id"))

# Create the tables
Base.metadata.create_all(engine)
