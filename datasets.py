import collections
import os
import hashlib
import io
import requests
import cdflib
import tempfile
import json
import zipfile
import shutil

import numpy as np
from sqlalchemy.sql.functions import func
from torch.utils.data import Dataset
from viresclient import SwarmRequest, ReturnedData
from datetime import datetime, timedelta
from sqlalchemy import and_
from sqlalchemy.inspection import inspect
import pandas as pd

from sql_models import (
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
PRE_FETCH = 10000

URL = "https://swarm-diss.eo.esa.int/?do=list&maxfiles={maxfiles}&pos={pos}&file=swarm%2FLevel2daily%2FEntire_mission_data%2F{measurement}%2FTMS%2F{sat}"
# URL = "https://swarm-diss.eo.esa.int/#swarm%2FLevel2daily%2FEntire_mission_data%2FTEC%2FTMS%2FSat_{}"

URI = "https://swarm-diss.eo.esa.int/?do=download&file="

def expand_measurements(measurements):
    resulting_measurements = []
    for m in measurements:
        if m[1] > 1:
            resulting_measurements += [m[0] + str(i) for i in range(m[1])]
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
        maxfiles: int = 2000,
        pos: int = 500,
        satelite: str = "Sat_A",
    ):
        self.bubble_measurements = bubble_measurements

        self.start_time = start_time
        self.end_time = end_time

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

        self.history_subquery = session.query(*history_cols).order_by(
            DataMeasurement.timestamp
        )

        self.index_subquery = session.query(
            DataMeasurement.bubble_index
        ).order_by(DataMeasurement.timestamp)

        self.size = session.query(DataMeasurement).count()

        # Cached Stuff
        self.index = 0
        self.history = None
        self.label = None
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
        scale = PRE_FETCH
        if self.window_size != 0:
            scale *= self.window_size

        if (
            self.index == (index // scale) and
            self.history and
            self.label
        ):
            return

        self.index = (index // scale)
        self.history = self.history_subquery.offset(
            self.index * scale
        ).limit(scale).all()
        self.label = self.index_subquery.offset(
            self.index * scale
        ).limit(scale).all()
     
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

                tmp_dir = tempfile.mkdtemp()
                tmp_file = zf.extract(cdf_file, path=tmp_dir)
                cdf_obj = cdflib.CDF(tmp_file)

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

        return self.size / self.window_size

    def __getitem__(self, index):
        self._cache_data(index)

        if self.window_size == 0:
            try:
                history = self.history[index % PRE_FETCH]
                label = self.label[index % PRE_FETCH]
            except IndexError as e:
                print(str(e))
                print(index % PRE_FETCH)
                print(len(self.history))
        else:
            start_index = (self.step_size * index)
            end_index = start_index + self.window_size

            history = self.tec_data_df[start_index:end_index]
            label = self.ibi_data_df[start_index:end_index]

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
        transform=None
    ):
        self.bubble_measurements = bubble_measurements
        self.start_time = start_time
        self.end_time = end_time
        self.time_diff = timedelta(days=1)
        self.window_size = window_size
        self.step_size = step_size
        self.transform = transform

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        self.data = self._request_swarm_data()

        self.tec_data_df = self.data[expand_measurements(TEC_MEASUREMENTS)].to_numpy()
        self.ibi_data_df = self.data[[self.bubble_measurements]].to_numpy()

        if shift:
            self.ibi_data_df += 1

        if self.window_size != 0:
            h, w = self.tec_data_df.shape
            trim_h = h - (h % self.window_size)

            self.tec_data_df = self.tec_data_df[:trim_h]
            self.ibi_data_df = self.ibi_data_df[:trim_h]

        return

    def _request_swarm_data(self):
        date_diff = self.end_time - self.start_time
        start_time = self.start_time
        end_time = self.start_time + self.time_diff

        overall_data = None

        for _ in range(date_diff.days):
            data_file = os.path.join(
                DATA_DIR,
                "data_" + start_time.strftime("%Y_%m_%d_") + end_time.strftime("%Y_%m_%d.pickle")
            )

            if not os.path.exists(data_file):
                tec_df = self._request_swarm_data_iter(
                    TEC_COLLECTION,
                    [m[0] for m in TEC_MEASUREMENTS],
                    start_time,
                    end_time
                )

                # Expand the GPS to three individual columns
                gps_df = pd.DataFrame(tec_df.pop('GPS_Position').values.tolist(), index=tec_df.index)
                tec_df = tec_df.join(gps_df.add_prefix('GPS_Position.')).sort_index(axis=1)
                
                # Expand the LEO position to three individual columns
                leo_df = pd.DataFrame(tec_df.pop('LEO_Position').values.tolist(), index=tec_df.index)
                tec_df = tec_df.join(leo_df.add_prefix('LEO_Position.')).sort_index(axis=1)

                ibi_df = self._request_swarm_data_iter(
                    IBI_COLLECTION,
                    [self.bubble_measurements],
                    start_time,
                    end_time,
                )

                tec_df = self.normalize_cols(
                    tec_df[expand_measurements(TEC_MEASUREMENTS)]
                )
                ibi_df = ibi_df[[self.bubble_measurements]]

                data = pd.merge(tec_df, ibi_df, left_index=True, right_index=True)


                print("------- Writing data to pickle file {}".format(data_file))
                data.to_pickle(data_file)
                print("------- Finished writing--------")
            else:
                # CSV file exists. Read the file
                print("------- Reading data from pickle file {}".format(data_file))
                data = pd.read_pickle(data_file)
                print("------- Finished reading pick file ------")

            if start_time == self.start_time:
                overall_data = data

            #update
            start_time += self.time_diff
            end_time += self.time_diff

        return overall_data 
        
    def _request_swarm_data_iter(
        self,
        collection_type,
        measurement_types,
        start_time_iter,
        end_time_iter,
    ):
        request = SwarmRequest()
        request.set_collection(collection_type)
        request.set_products(measurements=measurement_types)
        data = request.get_between(
            start_time_iter, end_time_iter
        )

        return data.as_dataframe()

    def normalize_cols(self, df):
        for col in df.columns:
            df[col] = (df[col] / df[col].max()).astype(np.float32)

        return df

    def __len__(self):
        assert len(self.tec_data_df) == len(self.ibi_data_df)

        return len(self.tec_data_df)

    def __getitem__(self, index):
        if self.window_size == 0:
            history = self.tec_data_df[index]
            label   = self.ibi_data_df[index]
        else:
            start_index = (self.step_size * index)
            end_index = start_index + self.window_size

            history = self.tec_data_df[start_index:end_index]
            label = self.ibi_data_df[start_index:end_index]

        if self.transform:
            history = self.transform(history)

        return history, label