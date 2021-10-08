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

column_headers = [
    str(c).split('.')[-1]
    for c in DataMeasurement.__table__.columns
]

column_headers = [
    c
    for c in column_headers
    if not(c in ["measurement_id", "meta_id"])
]

value_subquery = (
    session.query(DataMeasurement)
    .filter(DataMeasurement.bubble_index != -1)
    .order_by(DataMeasurement.timestamp)
)

os.mkdir("./graphs/")

step_size = 1000

print("Get count of measurements...")
count_measures = value_subquery.count()

count_measures = count_measures - (count_measures % step_size)

print(f"Count Measures {count_measures}")

index = 0
measurement_points = pd.DataFrame(columns=column_headers)
point_list = {}
while index < count_measures:
    # print(index)
    point = value_subquery.offset(index).limit(1).first()
    point_list = {
        "timestamp": point.timestamp,
        "gps_position1": point.gps_position1,
        "gps_position2": point.gps_position2,
        "gps_position3": point.gps_position3,
        "leo_position1": point.leo_position1,
        "leo_position2": point.leo_position2,
        "leo_position3": point.leo_position3,
        "prn": point.prn,
        "l1": point.l1,
        "l2": point.l2,
        "bubble_index": point.bubble_index,
    }
    measurement_points = measurement_points.append(point_list, ignore_index=True)

    index += step_size

session.close()

x = measurement_points['timestamp']

for c in list(point_list.keys()):
    Session = sessionmaker(bind=engine)
    Session.configure(bind=engine)
    session = Session()
    if c == 'timestamp':
        continue
    y = measurement_points[c]

    fig = plt.figure(figsize=(12,8))
    plt.plot(x, y)
    plt.ylabel(c)
    plt.savefig(OUTPUTS + f'{c}.png')
    session.close()
