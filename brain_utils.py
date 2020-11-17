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


# Get subject idx through the brain region
def get_subject_idx(region):
    if region == 'all':
        subject_idx = np.arange(0, 94, 1)
    elif region == 'left':
        subject_idx = np.arange(0, 94, 2)
    elif region == 'right':
        subject_idx = np.arange(1, 94, 2)
    else:
        raise ValueError("Check the brain region name!")

    return subject_idx


# Get data x through the model name
def get_subject_data_x(model_name, data, time, idx, minmax_x):
    if model_name == 'FC':
        data_x = data[1:time+1, idx]
    else:
        data_x = torch.FloatTensor((data[1:-1, idx] - minmax_x[0]) /
                                   (minmax_x[1] - minmax_x[0]))

    return data_x


# Normalize data
def normalize_data(datax, datay, model_name, minmax_x, minmax_y):
    if model_name == 'FC':
        x_data = (np.array(datax) - minmax_x[0]) / (minmax_x[1] - minmax_x[0])
    else:
        x_data = datax
    y_data = (np.array(datay) - minmax_y[0]) / (minmax_y[1] - minmax_y[0])
    return x_data, y_data


# Get data
def get_data(param):
    data_path = get_data_path(param['data_folder'])
    label_path = os.path.join(data_path, param['label_fname'])
    df_label = pd.read_csv(label_path)

    num_seed = int(param['present_time'])

    time_len = param['time_len']
    minmax_x = param['minmax_x']
    minmax_y = param['minmax_y']
    model_name = param['model']
    brain_region = param['brain_region']

    data_x = []
    data_y = []
    length = []
    for idx, row in df_label.iterrows():
        data_path_list = [data_path, row.project,
                          row.subject, 'rest_image.csv']
        subject_data_path = reduce(os.path.join, data_path_list)

        subject_data = np.genfromtxt(subject_data_path, delimiter=',')
        subject_idx = get_subject_idx(brain_region)

        subject_x = get_subject_data_x(model_name, subject_data, time_len,
                                       subject_idx, minmax_x)
        subject_y = row.new_age

        data_x.append(subject_x)
        data_y.append(subject_y)
        length.append(subject_x.shape[0])

    data_x, data_y = normalize_data(data_x, data_y, model_name,
                                    minmax_x, minmax_y)
    length = np.array(length)
    data_num = len(data_y)

    idx = np.arange(data_num)
    np.random.seed(num_seed)
    np.random.shuffle(idx)

    if model_name != 'FC':
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


# Use GRU or CPU
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(777)
        print('Using CUDA')
    else:
        device = 'cpu'
        print('Using CPU')
    torch.manual_seed(777)
    return device


# Put data in tensor
def get_tensor(device, data_x, data_y, length_x, start_idx, end_idx):
    x_tensor = torch.FloatTensor(data_x[start_idx:end_idx, :, :]).to(device)
    y_tensor = torch.FloatTensor(data_y[start_idx:end_idx])
    length_tensor = torch.FloatTensor(length_x[start_idx:end_idx]).to(device)
    y_tensor.unsqueeze_(-1)
    y_tensor = y_tensor.to(device)

    return x_tensor, y_tensor, length_tensor


# Normalize in tensor
def normalize_tensor(tensor, minmax):
    min_val = minmax[0]
    max_val = minmax[1]
    arr = tensor.cpu().data.numpy()*(max_val-min_val) + min_val
    return arr


# Train
def train(device, start, rate, datax, datay, lengthx):
    train_x_tensor, train_y_tensor, train_length_tensor = get_tensor(
        device, datax, datay, lengthx, start, start + rate)
    return train_x_tensor, train_y_tensor, train_length_tensor


# Validation
def valid(device, start, rate, datax, datay, lengthx):
    valid_x_tensor, valid_y_tensor, valid_length_tensor = get_tensor(
        device, datax, datay, lengthx, start, start + rate)
    return valid_x_tensor, valid_y_tensor, valid_length_tensor
