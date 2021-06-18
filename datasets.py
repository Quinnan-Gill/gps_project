import collections
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from viresclient import SwarmRequest, ReturnedData
from datetime import datetime
import pandas as pd

# Dateime Parsing String
DATETIME_PSTR = '%Y_%m_%d'
DATA_DIR = "data"
TEC_COLLECTION = "SW_OPER_TECATMS_2F"
TEC_MEASUREMENTS = [
    ("GPS_Position", 3),
    ("LEO_Position", 3),
    ("PRN", 1),
    ("L1", 1),
    ("L2", 1),
]
IBI_COLLECTION = "SW_OPER_IBIATMS_2F"
IBI_MEASUREMENT = {
    "index": "Bubble_Index",
    "predictions": "Bubble_Predictions"
}

def expand_measurements(measurements):
    resulting_measurements = []
    for m in measurements:
        if m[1] > 1:
            resulting_measurements += [m[0] + "." + str(i) for i in range(m[1])]
        else:
            resulting_measurements.append(m[0])

    return resulting_measurements

class BubbleDataset(Dataset):
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        bubble_measurements: str,
        chunk_size: int,
        transform=None
    ):
        self.bubble_measurements = bubble_measurements
        self.start_time = start_time
        self.end_time = end_time
        self.chunk_size = chunk_size
        self.transform = transform

        data_file = os.path.join(
            DATA_DIR,
            "data_" + start_time.strftime("%Y_%m_%d_") + end_time.strftime("%Y_%m_%d.pickle")
        )

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        if not os.path.exists(data_file):
            # CSV does not exist collect the data
            tec_df = self._request_swarm_data(
                TEC_COLLECTION,
                [m[0] for m in TEC_MEASUREMENTS]
            )

            # Expand the GPS to three individual columns
            gps_df = pd.DataFrame(tec_df.pop('GPS_Position').values.tolist(), index=tec_df.index)
            tec_df = tec_df.join(gps_df.add_prefix('GPS_Position.')).sort_index(axis=1)
            
            # Expand the LEO position to three individual columns
            leo_df = pd.DataFrame(tec_df.pop('LEO_Position').values.tolist(), index=tec_df.index)
            tec_df = tec_df.join(leo_df.add_prefix('LEO_Position.')).sort_index(axis=1)

            ibi_df = self._request_swarm_data(IBI_COLLECTION, [self.bubble_measurements])

            self.data = pd.merge(tec_df, ibi_df, left_index=True, right_index=True)

            print("------- Writing data to pickle file {}".format(data_file))
            self.data.to_pickle(data_file)
            print("------- Finished writing--------")

        else:
            # CSV file exists. Read the file
            print("------- Reading data from pickle file {}".format(data_file))
            self.data = pd.read_pickle(data_file)

        self.tec_data_df = self.normalize_cols(
            self.data[expand_measurements(TEC_MEASUREMENTS)]
        ).to_numpy()
        self.ibi_data_df = self.data[[self.bubble_measurements]].to_numpy() + 1

        h, w = self.tec_data_df.shape
        trim_h = h - (h % self.chunk_size)

        self.tec_data_df = self.tec_data_df[:trim_h]
        self.ibi_data_df = self.ibi_data_df[:trim_h]

        self.tec_data_df = np.reshape(
            self.tec_data_df, (-1, self.chunk_size, w)
        )
        self.ibi_data_df = np.reshape(
            self.ibi_data_df, (-1, self.chunk_size, 1)
        )

        return

    def _request_swarm_data(self, collection_type, measurement_types):
        request = SwarmRequest()
        request.set_collection(collection_type)
        request.set_products(measurements=measurement_types)
        data = request.get_between(
            self.start_time, self.end_time
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
        history = self.tec_data_df[index]
        if self.transform:
            history = self.transform(history)

        label   = self.ibi_data_df[index]

        return history, label