from sqlalchemy.sql.functions import count
from sql_models import (
    ViresRequests,
    ViresMetaData,
    DataMeasurement,
    DataPoint,
    session,
    engine
)
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.sql.sqltypes import BigInteger
from sqlalchemy.sql.visitors import CloningExternalTraversal

OUTPUTS = "./graphs/"


for c in list(point_list.keys()):
    
    if c == 'timestamp':
        continue
    y = measurement_points[c]

    fig = plt.figure(figsize=(12,8))
    plt.plot(x, y)
    plt.ylabel(c)
    plt.savefig(OUTPUTS + f'{c}.png')
