import numpy as np
import os
import pandas as pd
import itertools
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.model import model
from brain_plot import plot_result, plot_train_val_loss
from brain_utils import (safe_make_dir, train, valid, get_tensor, get_data,
                         get_device, write_loss, normalize_tensor)


# Main
def wrapper(param, data_x, data_y, length_x, learning_rate, lr_gamma,
            hidden_dim, layers):
    device = param['device']
    model_name = param['model']
    brain = param['brain_region']
    input_dim = param['region_n']
    n_epochs = param['n_epochs']
    minmax_y = param['minmax_y']
    rate_tr = param['rate_train']
    rate_va = param['rate_valid']
    rate_te = param['rate_test']
    now_time = param['present_time']
    train_num = param['number_train']
    valid_num = param['number_valid']
    test_num = param['number_test']
    layer_rate = learning_rate
    total_num = len(data_y)

    if train_num % rate_tr != 0:
        print('Please reset rate_train')
    if valid_num % rate_va != 0:
        print('Please reset rate_valid')
    if test_num % rate_te != 0:
        print('Please reset rate_test')

    cwd = os.getcwd()
    out_fname = f'{now_time}_h_{hidden_dim}_l_{layers}_lg_{lr_gamma}' \
                f'_n_{n_epochs}_lr{layer_rate}_model{model_name}'
    out_path = os.path.join(cwd, out_fname)
    safe_make_dir(out_path)
    temp_path = os.path.join(out_path, 'temp')
    safe_make_dir(temp_path)

    start = time.time()  # Start Learning
    print("Start Learning " + out_fname)
    output_dim = 1

    loss_list = []
    step_list = []
    epoch_list = []

    if brain == 'right' or brain == 'left':
        input_dim = input_dim // 2

    mynet = model(param, input_dim, hidden_dim, output_dim, layers, device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(mynet.parameters(), lr=layer_rate)
    lr_sche = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=100, gamma=lr_gamma)

    train_xdata = data_x[0:train_num, :, :]
    train_ydata = data_y[0:train_num]
    train_length_x = length_x[0:train_num]
    valid_xdata = data_x[train_num:train_num + valid_num, :, :]
    valid_ydata = data_y[train_num:train_num + valid_num]
    valid_length_x = length_x[train_num:train_num + valid_num]
    total_loss_valid_min = np.Inf

    for i in range(n_epochs):
        # Train
        mynet.train()
        loss = 0

        for tr in range(int(train_num / rate_tr)):
            train_x_tensor, train_y_tensor, train_length_tensor = train(
                device, tr*rate_tr, rate_tr, train_xdata, train_ydata,
                train_length_x)
            if model_name != 'FC':
                train_x_tensor = pack_padded_sequence(
                    train_x_tensor, train_length_tensor,
                    batch_first=True, enforce_sorted=False)
            optimizer.zero_grad()

            outputs = mynet(train_x_tensor)
            loss_train = criterion(outputs, train_y_tensor)
            loss_train.backward()

            optimizer.step()
            loss += float(loss_train)

        epoch_list.append(i)
        loss_list.append(loss)
        step_list.append('train')

        # Validation
        mynet.eval()
        valid_loss = 0

        for va in range(int(valid_num / rate_va)):
            valid_x_tensor, valid_y_tensor, valid_length_tensor = \
                valid(device, va*rate_va, rate_va, valid_xdata,
                      valid_ydata, valid_length_x)
            if model_name != 'FC':
                valid_x_tensor = pack_padded_sequence(
                    valid_x_tensor, valid_length_tensor,
                    batch_first=True, enforce_sorted=False)

            valid_result = mynet(valid_x_tensor)
            loss_valid = criterion(valid_result, valid_y_tensor)
            valid_loss = valid_loss + loss_valid

        epoch_list.append(i)
        loss_list.append(valid_loss.item())
        step_list.append('validation')

        if valid_loss.item() <= total_loss_valid_min:
            torch.save(mynet.state_dict(), os.path.join(temp_path, 'model.pt'))
            total_loss_valid_min = valid_loss.item()

        lr_sche.step()

    epoch_arr = np.array(epoch_list)
    loss_arr = np.array(loss_list)
    step_arr = np.array(step_list)

    # Write loss values in csv file
    write_loss(epoch_arr, loss_arr, step_arr, out_path)

    # Plot train and validation losses
    plot_train_val_loss(out_path, out_fname, dpi=800,
                        yscale='log', ylim=[0.0001, 10])

    end = time.time()  # Learning Done
    print(f"Learning Done in {end-start}s")

    # Test
    mynet.load_state_dict(torch.load(os.path.join(temp_path, 'model.pt')))

    mynet.eval()
    with torch.no_grad():
        test_x_tensor, test_y_tensor, test_length_tensor = get_tensor(
            device, data_x, data_y, length_x, train_num + valid_num, total_num)
        if model_name != 'FC':
            test_x_tensor = pack_padded_sequence(
                test_x_tensor, test_length_tensor,
                batch_first=True, enforce_sorted=False)

        test_result = mynet(test_x_tensor)
        test_loss = criterion(test_result, test_y_tensor)
        print(f"Test Loss: {test_loss.item()}")
    plot_result(test_y_tensor, test_result, minmax_y, out_path, out_fname)

    real_arr = normalize_tensor(test_y_tensor, minmax_y)[:, -1]
    result_arr = normalize_tensor(test_result, minmax_y)[:, -1]
    df_result = pd.DataFrame({'test_age': real_arr, 'real_age': result_arr})
    df_result.to_csv(os.path.join(out_path, 'test_vs_real.csv'))


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # solve display issue in ssh environment
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    device = get_device()
    get_time = time.strftime("%m%d%H%M", time.localtime())

    param = {'data_folder': 'rest_csv_data',
             'device': device,
             'label_fname': 'preprocessed_data.csv',
             'model': 'GRU',
             'brain_region': 'all',
             'bidirection': False,
             'minmax_x': [4, 16789],  # x_values are between 4 and 16788.8
             'minmax_y': [10, 80],  # y_values are between 10 and 80
             'drop_p': 0.5,  # Drop probability during training
             'region_n': 94,  # Number of brain regions (input dim 2)
             'time_len': 100,  # Number of timepoints (input dim 1)
             'n_head': 5,  # Number of multi head attention
             'n_epochs': 5000,
             # Iterable values
             'learning_rate_list': [0.01, 0.001, 0.0001, 0.00001],
             'lr_gamma_list': [0.99, 0.975, 0.95],
             'hidden_dim_list': [200, 300],
             'layers_list': [3, 4],
             'rate_train': 576,
             'rate_valid': 64,
             'rate_test': 155,
             'number_train': 576,
             'number_valid': 64,
             'number_test': 155,
             'present_time': get_time}

    # Get data
    print("Generating Data")
    data_x, data_y, length = get_data(param)

    product_set = itertools.product(
        param['learning_rate_list'],
        param['lr_gamma_list'],
        param['hidden_dim_list'],
        param['layers_list'])

    for learning_rate, lr_gamma, hidden_dim, layers in product_set:
        wrapper(param, data_x, data_y, length, learning_rate, lr_gamma,
                hidden_dim, layers)
        torch.cuda.empty_cache()
