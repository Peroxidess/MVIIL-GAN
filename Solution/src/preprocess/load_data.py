import numpy as np
import pandas as pd
import os
import re


def data_load(task_name):
    test_data = pd.DataFrame([])
    if 'mimic' in task_name:
        if 'preprocessed' in task_name:
            file_name = '../DataSet/mimic/data_preprocessed_row.csv'
            data = pd.read_csv(file_name, index_col=['subject_id'])
            target_dict = {'label1': 'label_dead'}
            train_data = data
    return train_data, test_data, target_dict
