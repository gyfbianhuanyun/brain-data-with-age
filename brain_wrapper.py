import numpy as np
import os
import pandas as pd
import itertools
import time
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from brain_plot import *
from brain_utils import *
from brain_RNN import *


# Main
def wrapper(param, data_x, data_y, lr_gamma, hidden_dim, layers):

    device = param['device']
    input_dim = param['region_n']
    n_epochs = param['n_epochs']
    outputfolder = param['outputfolder']
    tempfolder = param['tempfolder']
    minmax_y = param['minmax_y']
    rate_tr = param['rate_train']
    rate_va = param['rate_valid']
    rate_te = param['rate_test']
    n_k_fold = param['number_k_fold']

    output_dim = 1  # Output dimension
    drop_prob = 0.5  # Drop probability during training
    train_num = 560
    valid_num = 80
    test_num = 155
    total_num = len(data_y)
    k = 0

    if train_num % rate_tr != 0:
        print('Please reset rate_train')
    if valid_num % rate_va != 0:
        print('Please reset rate_valid')
    if test_num % rate_te != 0:
        print('Please reset rate_test')

    out_fname = f'hidden_dim_{hidden_dim}_layers_{layers}_lr_gamma_{lr_gamma}'

    mynet = RNNClassifier(
        input_dim, hidden_dim, output_dim, layers, drop_prob).to(device)

    start = time.time()  # Start Learning
    print("Start Learning " + out_fname)

    loss_list = []
    step_list = []
    epoch_list = []
    k_fold_list = []

    safe_make_dir(f'./{tempfolder}')

    kf = KFold(n_splits=n_k_fold, shuffle=False)
    for train_index, valid_index in kf.split(data_x[0:train_num+valid_num, :, :]):

        criterion = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)
        lr_sche = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=100, gamma=lr_gamma)

        train_xdata = data_x[train_index]
        train_ydata = data_y[train_index]
        valid_xdata = data_x[valid_index]
        valid_ydata = data_y[valid_index]

        k = k + 1
        total_loss_valid_min = np.Inf

        for i in range(n_epochs):
            # Train
            mynet.train()
            lr_sche.step()
            loss = 0

            for tr in range(int(train_num / rate_tr)):
                train_x_tensor, train_y_tensor = train(device, tr*rate_tr, rate_tr, train_xdata, train_ydata)

                optimizer.zero_grad()
                outputs = mynet(train_x_tensor)
                loss_train = criterion(outputs, train_y_tensor)
                loss_train.backward()

                optimizer.step()
                loss += float(loss_train)

            epoch_list.append(i)
            loss_list.append(loss)
            step_list.append('train')
            k_fold_list.append(k)

            # Validation
            mynet.eval()
            valid_loss = 0

            for va in range(int(valid_num / rate_va)):
                valid_x_tensor, valid_y_tensor = valid(device, va*rate_va, rate_va, valid_xdata, valid_ydata)
                valid_result = mynet(valid_x_tensor)
                loss_valid = criterion(valid_result, valid_y_tensor)
                valid_loss = valid_loss + loss_valid

            epoch_list.append(i)
            loss_list.append(valid_loss.item())
            step_list.append('validation')
            k_fold_list.append(k)

            if valid_loss.item() <= total_loss_valid_min:
                torch.save(mynet.state_dict(), f'./{tempfolder}/{out_fname}_{k}.pt')
                total_loss_valid_min = valid_loss.item()

        mynet.init_weights()

    epoch_arr = np.array(epoch_list)
    loss_arr = np.array(loss_list)
    step_arr = np.array(step_list)
    k_fold_arr = np.array(k_fold_list)

    output_path = f'./figs/{outputfolder}'
    safe_make_dir(output_path)
    # Write loss values in csv file
    write_loss(epoch_arr, loss_arr, step_arr, k_fold_arr, output_path, out_fname)

    # Plot train and validation losses
    plot_train_val_loss(output_path, out_fname, dpi=800, yscale='log', ylim=[0.0001, 10])

    end = time.time()  # Learning Done
    print(f"Learning Done in {end-start}s")

    # Test
    beta = 1 / n_k_fold  # The interpolation parameter

    mynet.load_state_dict(torch.load(f'./{tempfolder}/{out_fname}_1.pt'))
    params1 = mynet.named_parameters()
    dict_params = dict(params1)

    for name1, param1 in params1:
        if name1 in dict_params:
            dict_params[name1].data.copy_(beta * param1.data)

    for i in range(1, n_k_fold):
        mynet.load_state_dict(torch.load(f'./{tempfolder}/{out_fname}_{i+1}.pt'))
        params2 = mynet.named_parameters()
        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params:
                dict_params[name1].data.copy_(beta * dict_params2[name1].data + dict_params[name1].data)

    mynet.load_state_dict(dict_params, strict=False)

    mynet.eval()
    with torch.no_grad():
        test_x_tensor, test_y_tensor = get_tensor(
            device, data_x, data_y, train_num+valid_num+1, total_num)

        test_result = mynet(test_x_tensor)
        test_loss = criterion(test_result, test_y_tensor)
        print(f"Test Loss: {test_loss.item()}")
    plot_result(
        test_y_tensor, test_result, minmax_y, output_path, out_fname)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # solve display issue in ssh environment
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    device = get_device()

    param = {'data_folder': 'rest_csv_data',
             'device': device,
             'label_fname': 'preprocessed_data.csv',
             'minmax_x': [4, 16789],  # x_values are between 4 and 16788.8
             'minmax_y': [10, 80],  # y_values are between 10 and 80
             'outputfolder': 'output',
             'tempfolder': 'model_new',
             'region_n': 94,  # Number of brain regions (input dim 2)
             'time_len': 100,  # Number of timepoints (input dim 1)
             'n_epochs': 10000,
             # Iterable values
             'gamma_list': [0.99, 0.975, 0.95],
             'hidden_dim_list': [200, 300],
             'layers_list': [3, 4, 5, 6],
             'rate_train': 140,
             'rate_valid': 80,
             'rate_test': 155,
             'number_k_fold': 8}

    # Get data
    print("Generating Data")
    data_x, data_y = get_data(param)

    product_set = itertools.product(
        param['gamma_list'],
        param['hidden_dim_list'],
        param['layers_list'])

    #for lr_gamma, hidden_dim, layers in product_set:
     #   wrapper(param, data_x, data_y, lr_gamma, hidden_dim, layers)
    lr_gamma = param['gamma_list'][0]
    hidden_dim = param['hidden_dim_list'][1]
    layers = param['layers_list'][3]
    wrapper(param, data_x, data_y, lr_gamma, hidden_dim, layers)
    torch.cuda.empty_cache()
