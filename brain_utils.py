import numpy as np
import os
import pandas as pd
from functools import reduce
import torch
from torch.nn.utils.rnn import pad_sequence


# Get data path project and subject name
def get_data_path(data_folder):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, data_folder)
    return data_path


# Get data
def get_data(param):
    data_path = get_data_path(param['data_folder'])
    label_path = os.path.join(data_path, param['label_fname'])
    df_label = pd.read_csv(label_path)

    num_seed = int(param['present_time'])

    region_n = param['region_n']
    time_len = param['time_len']
    minmax_x = param['minmax_x']
    minmax_y = param['minmax_y']

    data_x = []
    data_y = []
    length = []
    for idx, row in df_label.iterrows():
        data_path_list = [data_path, row.project,
                          row.subject, 'rest_image.csv']
        subject_data_path = reduce(os.path.join, data_path_list)

        subject_data = np.genfromtxt(subject_data_path, delimiter=',')
        subject_x = torch.FloatTensor((subject_data[1:-1, :region_n] - minmax_x[0]) / (minmax_x[1] - minmax_x[0]))
        subject_y = row.new_age

        data_x.append(subject_x)
        data_y.append(subject_y)
        length.append(subject_x.shape[0])

    # Normalize data
    data_y = (np.array(data_y) - minmax_y[0]) / (minmax_y[1] - minmax_y[0])
    length = np.array(length)
    data_num = len(data_y)

    idx = np.arange(data_num)
    np.random.seed(num_seed)
    np.random.shuffle(idx)

    data_x = pad_sequence(data_x, batch_first=True)

    data_x = data_x[idx, :, :]
    data_y = data_y[idx]
    length = length[idx]
    return data_x, data_y, length


# Safe make
def safe_make_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)


# Write losses to file
def write_loss(epoch_number_arr, loss_arr, step_name_arr, out_path):
    df = pd.DataFrame({'epoch': epoch_number_arr,
                       'loss': loss_arr,
                       'step': step_name_arr})

    df.to_csv(os.path.join(out_path, 'loss.csv'))
