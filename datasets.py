import collections
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from viresclient import SwarmRequest, ReturnedData
from datetime import datetime, timedelta
import pandas as pd

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

        # self.tec_data_df = self.normalize_cols(
        #     self.data[expand_measurements(TEC_MEASUREMENTS)]
        # ).to_numpy()
        # self.ibi_data_df = self.data[[self.bubble_measurements]].to_numpy()

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