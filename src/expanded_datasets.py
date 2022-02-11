"""
This dataset collects all the data points on the 
"""
import collections
import enum
import os
import hashlib
import io
from re import S
from typing import List
import requests
import cdflib
import tempfile
import json
import zipfile
import shutil
import dask

import numpy as np
from absl import app, flags
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from sqlalchemy import and_
import pandas as pd

from sql_models import (
    ViresRequests,
    ViresMetaData,
    session,
)

from sql_models import DataMeasurement
from utils import _decode_time_str

DATA_DIR = "data"
TEC_COLLECTION = "SW_OPER_TECATMS_2F"

IBI_COLLECTION = "SW_OPER_IBIATMS_2F"
TEC_MEASUREMENTS = [
    ("gps_position", 3),
    ("leo_position", 3),
    ("prn", 1),
    ("l1", 1),
    ("l2", 1),
]
IBI_MEASUREMENT = {
    "index": "bubble_index",
    "predictions": "bubble_predictions"
}
CDF_EPOCH_1970 = 62167219200000.0
MAX_PRN = 32

ZIP_TIME_FMT = "%Y%m%dT%H%M%S"
PRE_FETCH = 5000

URL = "https://swarm-diss.eo.esa.int/?do=list&maxfiles={maxfiles}&pos={pos}&file=swarm%2FLevel2daily%2FEntire_mission_data%2F{measurement}%2FTMS%2F{sat}"

URI = "https://swarm-diss.eo.esa.int/?do=download&file="

FLAGS = flags.FLAGS

# The `ds_` prefix to all the commands is in order to avoid naming
# conflicts when using the CNN or RNN absl` 
flags.DEFINE_string('ds_start_val_time', '2016_01_01', 'The start datetime for evaluation')
flags.DEFINE_string('ds_end_val_time', '2016_01_02', 'The end datetime for evaluation')
flags.DEFINE_integer('ds_window_size', 120, 'How large the time window will be')
flags.DEFINE_integer('ds_step_size', 120, 'How much the window shifts')
flags.DEFINE_integer('ds_prefetch', 5000, 'How much to cache for the data')
flags.DEFINE_enum('ds_label', 'index', IBI_MEASUREMENT.keys(),
                    'Specifies the label for calculating the loss')
flags.DEFINE_boolean('fetch_data', False, help='If fetching data from the FTP or not')


class BubbleDatasetExpandedFTP(Dataset):
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        bubble_measurements: str,
        window_size: int,
        step_size: int,
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

        self.expanded_tec_columns = [
            "gps_position1", "gps_position2", "gps_position3",
            "l1", "l2"
        ]

        self.expanded_columns = self.get_columns() + ["bubble_index"]
        self.prns_seen = set()

        tec_url, tec_vires_request = self.get_data_request(
            maxfiles=maxfiles,
            pos=pos,
            measurement="TEC",
            satelite=satelite,
        )
        if not tec_vires_request.overall_processed:
            self.get_data_urls(tec_url, tec_vires_request)

        ibi_url, ibi_vires_request = self.get_data_request(
            maxfiles=maxfiles,
            pos=pos,
            measurement="IBI",
            satelite=satelite,
        )
        if not ibi_vires_request.overall_processed:
            self.get_data_urls(ibi_url, ibi_vires_request)

        self.get_data_range()

        
    def get_columns(self):
        def expand_measurements(measurement):
            if measurement == "prn":
                return []
            elif "leo" in measurement:
                return [measurement]
            return [measurement + f"_{i+1}" for i in range(MAX_PRN)]

        columns = []
        tec_meas_with_indexes = TEC_MEASUREMENTS
        for measurement, values in tec_meas_with_indexes:
            if values > 1:
                for i in range(values):
                    columns += expand_measurements(measurement + str(i+1))
            else:
                columns += expand_measurements(measurement)

        return columns

    def get_data_request(
        self,
        maxfiles,
        pos,
        measurement,
        satelite,
    ):
        measurement_url = URL.format(maxfiles=maxfiles, pos=pos, measurement=measurement, sat=satelite)
        measurement_url_hash = hashlib.sha1(measurement_url.encode("UTF-8")).hexdigest()[:10]
        vires_request = session.query(ViresRequests).filter_by(url_hash=measurement_url_hash).first()
        if vires_request is None:
            vires_request = ViresRequests(
                maxfiles=maxfiles,
                position=pos,
                satelite=satelite,
                measurement=measurement,
                url_used=measurement_url,
                url_hash=measurement_url_hash,
                overall_processed=False,
            )
            session.add(vires_request)
            session.commit()
        
        return measurement_url, vires_request

    def parse_zip_datetime(self, zip_file):
        zip_list = zip_file.split('_')
        
        start_time = datetime.strptime(zip_list[-3], ZIP_TIME_FMT)
        end_time = datetime.strptime(zip_list[-2], ZIP_TIME_FMT)

        return start_time, end_time

    def get_data_urls(
        self,
        url,
        vires_request,
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
                start_time, end_time = self.parse_zip_datetime(swarm_set["name"])

                vmd = ViresMetaData(
                    zip_file=swarm_set["path"].replace("\/", "/"),
                    zip_hash=swarm_set_hash,
                    processed=False,
                    measurement=vires_request.measurement,
                    start_time=start_time,
                    end_time=end_time,
                    request_id=vires_request.request_id,
                )
            vmds.append(vmd)

        session.bulk_save_objects(vmds)
        vires_request.overall_processed = True
        session.commit()

    def get_cdf_tec_data(self, tec_df, cdf_obj):
        for measurement, values in TEC_MEASUREMENTS:
            if values > 1:
                for i in range(values):
                    cdf_meas = cdf_obj[measurement][:, i]
                    tec_df[measurement + str(i+1)] = cdf_meas
            else:
                cdf_meas = cdf_obj[measurement]
                tec_df[measurement] = cdf_meas
        
        return tec_df

    def get_data_range(self):
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

            processed = False

            for zip_file in zip_range:
                print("-" * 20)
                print(f"{zip_file.zip_file}")
                print("-" * 20)

                if zip_file.processed:
                    processed = True
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
                    tec_df = self.get_cdf_tec_data(tec_df, cdf_obj)
                    tec_df = tec_df.set_index("timestamp")
                else:
                    ibi_df["timestamp"] = pd.to_datetime((cdf_obj["Timestamp"] - CDF_EPOCH_1970)/1e3, unit='s')
                    ibi_df[self.bubble_measurements] = cdf_obj[self.bubble_measurements]
                    ibi_df = ibi_df.set_index("timestamp")

                shutil.rmtree(tmp_dir)

            if processed:
                continue

            if self.index_filter:
                ibi_df = ibi_df[ibi_df.bubble_index != -1]

            data = pd.merge(tec_df, ibi_df, left_index=True, right_index=True)
            data = data.head()
            data = data.reset_index()

            def count_leo_groups(grouped_df):
                return len(grouped_df["leo_position1"].unique())

            def groupby_func(grouped_df):
                expanded_df = pd.DataFrame(
                    columns=self.expanded_columns
                )
                exp_dict = {}
                for row in grouped_df.iterrows():
                    row_obj = row[1]

                    for j in range(3):
                        exp_dict[f"leo_position{j+1}"] = row_obj[f"leo_position{j+1}"]
                    exp_dict["bubble_index"] = row_obj["bubble_index"]
                    prn = row_obj["prn"]
                    self.prns_seen.add(prn)
                    for col in self.expanded_tec_columns:
                        exp_dict[f"{col}_{prn}"] = row_obj[col]

                expanded_df = expanded_df.append(exp_dict, ignore_index=True)

                return expanded_df

            data_count = data.groupby(by="timestamp").apply(count_leo_groups)
            data_count_idx = [
                datetime.strftime(idx, '%Y-%m-%d %H:%M:%S')
                for idx in
                data_count.where(lambda x: x > 1).dropna().index
            ]
            
            data_df = data[~data['timestamp'].isin(data_count_idx)]
            data_df = data_df.groupby(by="timestamp").apply(groupby_func)

            
            exp_file = zip_file.zip_file.replace("/", "_").replace(".ZIP", ".parquet")
            data_df.to_parquet(f"data/{exp_file}")
            
            for zip_file in zip_range:
                zip_file.processed = True
                zip_file.data_file = f"data/{exp_file}"
                zip_file.data_size = len(data_df)
            session.commit()


class BubbleDatasetExpanded(Dataset):
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        bubble_measurements: str,
        window_size: int,
        step_size: int,
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

        self.vires_data_files = (
            session.query(ViresMetaData.data_file)
            .filter(
                and_(
                    ViresMetaData.processed,
                    ViresMetaData.measurement == "TEC", # TEC and IBI should have the same datafile
                    ViresMetaData.start_time >= self.start_time,
                    ViresMetaData.end_time <= self.end_time
                )
            )
        ).all()      

        self.data_df = None
        data_df_set = False

        for datafile in self.vires_data_files:
            datafile = datafile[0]

            if not data_df_set:
                self.data_df = pd.read_parquet(datafile)
                data_df_set = True
            else:
                new_df = pd.read_parquet(datafile)
                self.data_df = pd.concat([self.data_df, new_df])

        # Filter out columns with all nans
        for col in self.data_df.columns:
            if pd.isnull(self.data_df[col]).all():
                self.data_df.pop(col)
        
        for col in self.data_df.columns[:-1]:
            self.data_df[col] = self.data_df[col]  / self.data_df[col].abs().max()

        self.data_df = self.data_df.replace(np.nan, 0.0)

        self.size = len(self.data_df)
        print("-" * 20)
        print(f"Size: {self.size}")
        print("-" * 20)

        self.label = self.data_df.pop('bubble_index').to_numpy()
        self.history = self.data_df.to_numpy()

    def __len__(self):
        if self.window_size == 0:
            return self.size

        length = self.size - (self.size % self.window_size)
        length -= self.window_size

        return length // self.step_size

    def __getitem__(self, index):
        if self.window_size == 0:
            history = self.history[index]
            label = self.label[index]
        else:
            start_index = (self.step_size * index)
            end_index = (start_index + self.window_size)

            history = self.history[start_index:end_index].astype(np.float32)
            label = self.label[start_index:end_index].astype(np.long)

        return history, label

    def get_column_size(self):
        return len(self.history[0])

def get_data():
    if FLAGS.fetch_data:
        BubbleDatasetExpandedFTP(
            start_time=_decode_time_str(FLAGS.ds_start_val_time),
            end_time=_decode_time_str(FLAGS.ds_end_val_time),
            bubble_measurements=IBI_MEASUREMENT[FLAGS.ds_label],
            window_size=FLAGS.ds_window_size,
            step_size=FLAGS.ds_step_size,
            # index_filter=None,
            prefetch=FLAGS.ds_prefetch
        )
    else:
        values = BubbleDatasetExpanded(
            start_time=_decode_time_str(FLAGS.ds_start_val_time),
            end_time=_decode_time_str(FLAGS.ds_end_val_time),
            bubble_measurements=IBI_MEASUREMENT[FLAGS.ds_label],
            window_size=FLAGS.ds_window_size,
            step_size=FLAGS.ds_step_size,
            # index_filter=None,
            prefetch=FLAGS.ds_prefetch
        )

        values.get_column_size()

def main(unused_argvs):
    get_data()

if __name__ == '__main__':
    app.run(main)