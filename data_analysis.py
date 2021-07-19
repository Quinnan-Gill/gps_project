import collections
import copy
import os
import sys
import time

from absl import app, flags
from datetime import datetime

from utils import (
    _decode_time_str,
    DATETIME_PSTR,
)
from datasets import (
    BubbleDataset,
    BubbleDatasetFTP,
    IBI_MEASUREMENT,
    expand_measurements,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer('window_size', 0, 'How large the time window will be')
flags.DEFINE_integer('step_size', 5, 'How much the window shifts')
flags.DEFINE_string('start_time', '2016_01_01', 'The start datetime for metrics')
flags.DEFINE_string('end_time', '2016_01_02', 'The end datetime for metrics')
flags.DEFINE_enum('label', 'index', IBI_MEASUREMENT.keys(),
                    'Specifies the label for calculating the loss')


class Bubble:
    def __init__(self, val=0, count=0, start=0, end=0):
        self.val = val
        self.count = count
        self.start = start
        self.end = end
    
    def set_count(self):
        assert self.start <= self.end
        self.count = self.end - self.start

    def __str__(self):
        return "Count: {} Start: {} End: {}".format(
            self.count, self.start, self.end
        )

    def __repr__(self):
        return "Count: {} Start: {} End: {}".format(
            self.count, self.start, self.end
        )

class BubbleList:
    def __init__(self):
        self.bubble_list = []
    
    def _create_bubble(self, index, val):
        new_bubble = Bubble(val=val, start=index)
        self.bubble_list.append(new_bubble)

    def _finish_bubble(self, index):
        old_bubble = self.bubble_list[-1]
        old_bubble.end = index-1
        old_bubble.set_count()

    def create_new_bubble(self, index, val):
        if len(self.bubble_list) == 0:
            self._create_bubble(index, val)
        else:
            self._finish_bubble(index)
            self._create_bubble(index, val)
    
    def finish_list(self, index):
        self._finish_bubble(index)

    def __len__(self):
        return len(self.bubble_list)

    def __getitem__(self, index):
        return self.bubble_list[index]

class Accummulator:
    def __init__(self):
        self.bubble_size = BubbleList()
        self.current_value = None
        
    def value_iter(self, index, new_val):
        if new_val != self.current_value:
            self.current_value = new_val
            self.bubble_size.create_new_bubble(index, new_val)
    
    def finish_iter(self, index):
        self.bubble_size.finish_list(index)

        return self.bubble_size

def bubble_classes(bubble_list):
    class_dict = collections.defaultdict(int)
    bubble_size = collections.defaultdict(list)

    for bubble in bubble_list:
        class_dict[bubble.val] += 1
        bubble_size[bubble.val].append(bubble.count)

    print("--- Frequency of Types of Index ---")
    print(dict(class_dict))

    print("--- Average Bubble Size ---")
    for key, value in bubble_size.items():
        print("{} average = {}".format(
            key,
            sum(value) / len(value)
        ))


def find_bubble_frequency(metric_dataset):
    acc = Accummulator()

    start_time = time.time()
    print("---- Starting Metrics ----")
    for i in range(len(metric_dataset)):
        _, label = metric_dataset[i]

        acc.value_iter(i, label[0])
    lap_time = time.time() - start_time
    print("Lap time: {}".format(lap_time))
    
    bubble_list = acc.finish_iter(len(metric_dataset))

    bubble_classes(bubble_list)
    
def get_data_metrics():
    metric_dataset = BubbleDatasetFTP(
        start_time=_decode_time_str(FLAGS.start_time),
        end_time=_decode_time_str(FLAGS.end_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
        window_size=0,
        shift = False,
        step_size=FLAGS.step_size,
    )

    find_bubble_frequency(metric_dataset)

def main(unused_argvs):
    get_data_metrics()

if __name__ == '__main__':
    app.run(main)