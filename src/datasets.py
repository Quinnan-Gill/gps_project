import collections
import os
import hashlib
import io
from typing import List
import requests
import cdflib
import tempfile
import json
import zipfile
import shutil
import zarr

import numpy as np
from sqlalchemy.sql.functions import func
from torch.utils.data import Dataset
from viresclient import SwarmRequest, ReturnedData
from datetime import datetime, timedelta
from sqlalchemy import and_
from sqlalchemy.inspection import inspect
import pandas as pd

from sql_models import (
    DataPoint,
    ViresRequests,
    ViresMetaData,
    DataMeasurement,
    session,
    engine
)

DATA_DIR = "data"
TEC_COLLECTION = "SW_OPER_TECATMS_2F"
TEC_MEASUREMENTS = [
    ("gps_position", 3),
    ("leo_position", 3),
    ("prn", 1),
    ("l1", 1),
    ("l2", 1),
]
IBI_COLLECTION = "SW_OPER_IBIATMS_2F"
IBI_MEASUREMENT = {
    "index": "bubble_index",
    "predictions": "bubble_predictions"
}
CDF_EPOCH_1970 = 62167219200000.0

ZIP_TIME_FMT = "%Y%m%dT%H%M%S"
PRE_FETCH = 500

URL = "https://swarm-diss.eo.esa.int/?do=list&maxfiles={maxfiles}&pos={pos}&file=swarm%2FLevel2daily%2FEntire_mission_data%2F{measurement}%2FTMS%2F{sat}"
# URL = "https://swarm-diss.eo.esa.int/#swarm%2FLevel2daily%2FEntire_mission_data%2FTEC%2FTMS%2FSat_{}"

URI = "https://swarm-diss.eo.esa.int/?do=download&file="

def expand_measurements(measurements):
    resulting_measurements = []
    for m in measurements:
        if m[1] > 1:
            resulting_measurements += [m[0] + str(i+1) for i in range(m[1])]
        else:
            resulting_measurements.append(m[0])

    return resulting_measurements

class BubbleDatasetFTP(Dataset):
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        bubble_measurements: str,
        window_size: int,
        step_size: int,
        shift: bool = True,
        prefetch: int = PRE_FETCH,
        index_filter: List[int] = [-1],
        maxfiles: int = 2000,
        pos: int = 500,
        satelite: str = "Sat_A",
    ):
        self.bubble_measurements = bubble_measurements

        self.start_time = start_time
        self.end_time = end_time
        self.prefetch = prefetch
        self.index_filter = index_filter

        self.time_diff = timedelta(days=1)
        self.window_size = window_size
        self.step_size = step_size

        tec_url, v_tec_req = self._get_data_request(
            maxfiles=maxfiles,
            pos=pos,
            measurement="TEC",
            satelite=satelite,
        )
        if not v_tec_req.overall_processed:
            self._get_data_urls(tec_url, v_tec_req)

        ibi_url, v_ibi_req = self._get_data_request(
            maxfiles=maxfiles,
            pos=pos,
            measurement="IBI",
            satelite=satelite,
        )
        if not v_ibi_req.overall_processed:
            self._get_data_urls(ibi_url, v_ibi_req)

        self._get_data_range()

        history_cols = [
            col for col in inspect(DataMeasurement).c
            if col.name in expand_measurements(TEC_MEASUREMENTS)
        ]

        self.history_subquery = session.query(*history_cols).filter(
            and_(
                DataMeasurement.timestamp >= self.start_time,
                DataMeasurement.timestamp <= self.end_time
            )
        ).order_by(
            DataMeasurement.timestamp
        )

        self.index_subquery = session.query(
            DataMeasurement.bubble_index
        ).filter(
            and_(
                DataMeasurement.timestamp >= self.start_time,
                DataMeasurement.timestamp <= self.end_time,
                DataMeasurement.bubble_index != -1
            )
        ).order_by(DataMeasurement.timestamp)

        self.size = session.query(
            DataMeasurement
        ).filter(
            and_(
                DataMeasurement.timestamp >= self.start_time,
                DataMeasurement.timestamp <= self.end_time,
                DataMeasurement.bubble_index != -1
            )
        ).count()

        # Cached Stuff
        self.index = 0
        self.history = None
        self.label = None
        self.values_set = False
        self._cache_data(self.index)

    def _get_data_request(
        self,
        maxfiles,
        pos,
        measurement,
        satelite,
    ):
        m_url = URL.format(maxfiles=maxfiles, pos=pos, measurement=measurement, sat=satelite)
        m_url_hash = hashlib.sha1(m_url.encode("UTF-8")).hexdigest()[:10]
        v_req = session.query(ViresRequests).filter_by(url_hash=m_url_hash).first()
        if v_req is None:
            v_req = ViresRequests(
                maxfiles=maxfiles,
                position=pos,
                satelite=satelite,
                measurement=measurement,
                url_used=m_url,
                url_hash=m_url_hash,
                overall_processed=False,
            )
            session.add(v_req)
            session.commit()
        
        return m_url, v_req

    def _parse_zip_datetime(self, zip_file):
        zip_list = zip_file.split('_')
        
        start_time = datetime.strptime(zip_list[-3], ZIP_TIME_FMT)
        end_time = datetime.strptime(zip_list[-2], ZIP_TIME_FMT)

        return start_time, end_time

    def _cache_data(self, index):
        scale = self.prefetch
        if self.window_size != 0:
            scale *= self.window_size

        if (
            self.index == (index // scale) and
            self.values_set
        ):
            return

        self.index = (index // scale)
        # print("Getting cache data {} {} {}".format(self.index, index, self.size))
        self.history = np.asarray(self.history_subquery.offset(
            self.index * scale
        ).limit(scale).all(), dtype=np.float32)
        self.label = np.asarray(self.index_subquery.offset(
            self.index * scale
        ).limit(scale).all())
        self.values_set = True
     
    def _get_data_urls(
        self,
        url,
        v_req,
    ):
        resp = requests.get(url, verify=False)
        
        if not resp.ok:
            raise Exception("Bad URL: {}".format(url))

        swarm_datasets = json.loads(resp.content)['results']

        vmds = []
        for swarm_set in swarm_datasets:
            swarm_set_hash = hashlib.sha1(swarm_set["path"].encode("UTF-8")).hexdigest()[:10]
            vmd = session.query(ViresMetaData).filter_by(
                zip_hash=swarm_set_hash
            ).first()
            if not vmd:
                start_time, end_time = self._parse_zip_datetime(swarm_set["name"])

                vmd = ViresMetaData(
                    zip_file=swarm_set["path"].replace("\/", "/"),
                    zip_hash=swarm_set_hash,
                    processed=False,
                    measurement=v_req.measurement,
                    start_time=start_time,
                    end_time=end_time,
                    request_id=v_req.request_id,
                )
            vmds.append(vmd)

        session.bulk_save_objects(vmds)
        v_req.overall_processed = True
        session.commit()

    def _get_data_range(self):
        zip_tec_ranges = session.query(ViresMetaData).filter(
            and_(
                ViresMetaData.start_time >= self.start_time,
                ViresMetaData.end_time <= self.end_time,
            )
        ).filter(
            ViresMetaData.measurement == "TEC"
        ).order_by(
            ViresMetaData.start_time
        ).all()

        zip_ibi_ranges = session.query(ViresMetaData).filter(
            and_(
                ViresMetaData.start_time >= self.start_time,
                ViresMetaData.end_time <= self.end_time,
            )
        ).filter(
            ViresMetaData.measurement == "IBI"
        ).order_by(
            ViresMetaData.start_time
        ).all()

        zip_ranges = zip(zip_tec_ranges, zip_ibi_ranges)

        for zip_range in zip_ranges:
            tec_df = pd.DataFrame()
            ibi_df = pd.DataFrame()

            for zip_file in zip_range:
                if zip_file.processed:
                    continue

                resp = requests.get(URI + zip_file.zip_file, verify=False)
                if not resp.ok:
                    continue

                cdf_file = zip_file.zip_file.split('/')[-1].replace('ZIP', 'cdf')
                zf = zipfile.ZipFile(io.BytesIO(resp.content), 'r')

                try:
                    tmp_dir = tempfile.mkdtemp()
                    tmp_file = zf.extract(cdf_file, path=tmp_dir)
                    cdf_obj = cdflib.CDF(tmp_file)
                except:
                    print("----------------------------")
                    print("    ERROR: Downloading:")
                    print("{}".format(cdf_file))
                    print("----------------------------")
                    continue

                if zip_file.measurement == "TEC":
                    tec_df["timestamp"] = pd.to_datetime((cdf_obj["Timestamp"] - CDF_EPOCH_1970)/1e3, unit='s')
                    tec_df = self._get_cdf_tec_data(tec_df, cdf_obj)
                    tec_df = tec_df.set_index("timestamp")
                else:
                    ibi_df["timestamp"] = pd.to_datetime((cdf_obj["Timestamp"] - CDF_EPOCH_1970)/1e3, unit='s')
                    ibi_df[self.bubble_measurements] = cdf_obj[self.bubble_measurements]
                    ibi_df = ibi_df.set_index("timestamp")

                shutil.rmtree(tmp_dir)

            data = pd.merge(tec_df, ibi_df, left_index=True, right_index=True)
            h, w = data.shape
            if self.index_filter and w > 0 and h > 0:
                data = data[data.bubble_index != -1]

            data = data.reset_index()
            data.to_sql("data_measurement", engine, if_exists='append', index_label='measurement_id')
            
            for zip_file in zip_range:
                zip_file.processed = True
            session.commit()
    
    def _get_cdf_tec_data(self, tec_df, cdf_obj):
        for measurement, values in TEC_MEASUREMENTS:
            if values > 1:
                for i in range(values):
                    cdf_meas = cdf_obj[measurement][:, i]
                    tec_df[measurement + str(i+1)] = (cdf_meas / cdf_meas.max()).astype(np.float32)
            else:
                cdf_meas = cdf_obj[measurement]
                tec_df[measurement] = (cdf_meas / cdf_meas.max()).astype(np.float32)
        
        return tec_df

    def __len__(self):
        if self.window_size == 0:
            return self.size

        return self.size // self.window_size

    def __getitem__(self, index):
        self._cache_data(index)

        if self.window_size == 0:
            try:
                history = self.history[index % self.prefetch]
                label = self.label[index % self.prefetch]
            except IndexError as e:
                print(str(e))
                print(index % self.prefetch)
                print(len(self.history))
        else:
            start_index = (self.step_size * index) % self.prefetch
            end_index = (start_index + self.window_size)

            history = self.history[start_index:end_index]
            label = self.label[start_index:end_index]

            assert history.shape[0] == self.window_size, "{}: {}: {}".format(history.shape, index, (start_index, end_index))

        return history, label


class BubbleDataset(Dataset):
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        bubble_measurements: str,
        window_size: int,
        step_size: int,
        shift: bool = True,
        prefetch: int = PRE_FETCH,
        index_filter: List[int] = [-1],
        maxfiles: int = 2000,
        pos: int = 500,
        satelite: str = "Sat_A",
    ):
        self.bubble_measurements = bubble_measurements

        self.start_time = start_time
        self.end_time = end_time
        self.prefetch = prefetch
        self.index_filter = index_filter

        self.window_size = window_size
        self.step_size = step_size

        history_cols = [
            col for col in inspect(DataPoint).c
            if col.name in expand_measurements(TEC_MEASUREMENTS)
        ]

        data_filter = and_(
            DataPoint.timestamp >= self.start_time,
            DataPoint.timestamp <= self.end_time,
            DataPoint.bubble_index != -1
        )

        self.history_subquery = session.query(*history_cols).filter(
            data_filter
        )

        self.index_subquery = session.query(
            DataPoint.bubble_index
        ).filter(
            data_filter
        )

        self.size = session.query(
            DataPoint
        ).filter(
            data_filter
        ).count()

        self.history = None
        self.label = None
        self.__cache_data(10)

        import pdb
        pdb.set_trace()

        print("Data set is of size: {}".format(self.size))

    def __cache_data(self, chunk_size):
        offset = self.size // chunk_size

        for i in range(chunk_size):
            if not (
                isinstance(self.history, zarr.core.Array)
                or isinstance(self.label, zarr.core.Array)
            ):
                self.history = zarr.array(
                    self.history_subquery.offset(offset * i).limit(offset * (i+1)).all()
                )
                self.label = zarr.array(
                    self.index_subquery.offset(offset * i).limit(offset * (i+1)).all()
                )
            else:
                temp_history = zarr.array(
                    self.history_subquery.offset(offset * i).limit(offset * (i+1)).all()
                )
                temp_label = zarr.array(
                    self.index_subquery.offset(offset * i).limit(offset * (i+1)).all()
                )

                self.history.append(temp_history)
                self.index_subquery(temp_label)
        
            

    def __len__(self):
        if self.window_size == 0:
            return self.size

        return self.size // self.window_size

    def __getitem__(self, index):
        if self.window_size == 0:
            history = np.asarray(
                self.history_subquery.filter_by(measurement_id=index).first(),
                dtype=np.float32
            )
            label = np.asarray(
                self.index_subquery.filter_by(measurement_id=index).first(),
                dtype=np.float32
            )
        else:
            start_index = (self.step_size * index)
            end_index = (start_index + self.window_size)

            index_filter = and_(
                DataPoint.measurement_id >= start_index,
                DataPoint.measurement_id < end_index
            )

            history = np.asarray(
                self.history_subquery.filter(
                    index_filter
                ).all(),
                dtype=np.float32
            )
            label = np.asarray(
                self.index_subquery.filter(
                    index_filter
                ).all(),
                dtype=np.float32
            )

            assert history.shape[0] == self.window_size, "{}: {}: {}".format(history.shape, index, (start_index, end_index))

        return history, label