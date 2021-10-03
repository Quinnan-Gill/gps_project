from sqlalchemy.inspection import inspect

from sql_models import (
    ViresRequests,
    ViresMetaData,
    DataMeasurement,
    DataPoint,
    session,
    engine
)
from tqdm import tqdm

wanted_cols = [
    col for col in inspect(DataMeasurement).c
    if col.name != "measurement_id"
]

value_subquery = (
    session.query(DataMeasurement)
    .filter(DataMeasurement.bubble_index != -1)
    .order_by(DataMeasurement.timestamp)
)

count_measures = value_subquery.count()

scale = 30000
index = 0
count = 0

session.query(DataPoint).delete()
session.commit()

while count < count_measures:
    bulk_list = value_subquery.offset(index * scale).limit(scale).all()

    new_data = []
    progress_bar = tqdm(enumerate(bulk_list))
    for step, measure in progress_bar:
        point = DataPoint(
            measurement_id=count,
            timestamp=measure.timestamp,
            gps_position1=measure.gps_position1,
            gps_position2=measure.gps_position2,
            gps_position3=measure.gps_position3,
            leo_position1=measure.leo_position1,
            leo_position2=measure.leo_position2,
            leo_position3=measure.leo_position3,
            prn=measure.prn,
            l1=measure.l1,
            l2=measure.l2,
        )
        # new_data.append(point)
        session.add(point)
        count += 1
        progress_bar.set_description(
            'Step: %d/%d, Completion: %d/%d: %d%%' % (step, scale, count, count_measures, (count * 100) // count_measures)
        )
    session.commit()
    index += 1
