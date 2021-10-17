import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags
from tqdm import tqdm

from utils import (
    _decode_time_str,
    DATETIME_PSTR,
)
from datasets import (
    BubbleDataset,
    BubbleDatasetFTP,
    IBI_MEASUREMENT,
    TEC_MEASUREMENTS,
    expand_measurements,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer('window_size', 0, 'How large the time window will be')
flags.DEFINE_integer('step_size', 5, 'How much the window shifts')
flags.DEFINE_string('start_time', '2016_01_01', 'The start datetime for metrics')
flags.DEFINE_string('end_time', '2016_01_02', 'The end datetime for metrics')
flags.DEFINE_integer('prefetch', 100000, 'The number of entries to prefetch')
flags.DEFINE_integer('data_points', 20000, 'The number of data points in the graph')
flags.DEFINE_enum('label', 'index', IBI_MEASUREMENT.keys(),
                    'Specifies the label for calculating the loss')

OUTPUTS = "./graphs/"

def plot_values(x, y, col_name):
    if not os.path.exists(OUTPUTS):
        os.mkdir(OUTPUTS)

    fig = plt.figure(figsize=(12,8))
    plt.plot(x, y)
    plt.ylabel(col_name)
    plt.savefig(OUTPUTS + f'{col_name}.png')

def plot_data(metric_dataset):
    step_size = len(metric_dataset) // FLAGS.data_points

    n = 0
    hist_x = [0]
    total_data_history, _ = metric_dataset[n]
    total_data_history = np.array([total_data_history])
    n += step_size
    progress_bar = tqdm(total=len(metric_dataset))

    while n < len(metric_dataset):
        data_history, _ = metric_dataset[n]

        total_data_history = np.concatenate(
            (total_data_history, [data_history]), axis=0
        )
        # total_data_label = np.append([total_data_label], [data_label], axis=0)
        hist_x.append(n)
        progress_bar.update(step_size)
        n += step_size

    progress_bar.close()

    print("Finished fetching data")
    for i, col in enumerate(expand_measurements(TEC_MEASUREMENTS)):
        plot_values(hist_x, total_data_history[:,i], col)


def get_data_metrics():
    metric_dataset = BubbleDataset(
        start_time=_decode_time_str(FLAGS.start_time),
        end_time=_decode_time_str(FLAGS.end_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
        window_size=0,
        shift = False,
        prefetch=FLAGS.prefetch,
        step_size=FLAGS.step_size,
    )

    plot_data(metric_dataset)

def main(unused_argvs):
    get_data_metrics()

if __name__ == '__main__':
    app.run(main)
