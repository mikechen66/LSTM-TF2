#!/usr/bin/env python
# coding: utf-8

# time_series_preprocessor.py

import csv
import numpy as np

def load_series(filename, series_idx=1):
    # Define the method that loads the time series and normalizes it:
    try:
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile)
            data = [float(row[series_idx]) for row in csvreader if len(row) > 0]
            normalized_data = (data - np.mean(data)) / np.std(data)
        return normalized_data
    except IOError:
        Print("Error occurred")

        return None

def split_data(data, percent_train):
    # Split the data as train and test sets 
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)

    return train_data, test_data