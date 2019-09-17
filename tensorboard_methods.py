import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from typing import List

def read_events_file(events_file_path: str, attribute: str) -> List:
    """
    returns values in tensorboard event file associated with the given attribute tag

    :param events_file_path: path to tensorboard events file
    :param attribute: name of tag to filter events file for

    :return values: list of values associated with attibute.
    """
    values = []
    for e in tf.train.summary_iterator(events_file_path):
        for v in e.summary.value:
            if v.tag == attribute:
                values.append(v.simple_value)
    return values

def smooth_values(values: List, window_width: int) -> List:
    """
    moving average of list of values

    :param values: raw values 
    :param window_width: width of moving average calculation

    :return smoothed_values: moving average values
    """
    cumulative_sum = np.cumsum(values, dtype=float)
    cumulative_sum[window_width:] = cumulative_sum[window_width:] - cumulative_sum[:-window_width]
    smoothed_values = cumulative_sum[window_width - 1:] / window_width
    return smoothed_values