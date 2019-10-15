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


# Initialization parameter
def init_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

# Put data in tensor
def get_tensor(device, data_x, data_y, start_idx, end_idx):
    x_tensor = torch.FloatTensor(data_x[start_idx:end_idx, :, :]).to(device)
    y_tensor = torch.FloatTensor(data_y[start_idx:end_idx])
    y_tensor.unsqueeze_(-1)
    y_tensor = y_tensor.to(device)
    return x_tensor, y_tensor

# Model
class RNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, drop_prob):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=layers, batch_first=True,
            dropout=drop_prob)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialize RNN Module
        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        # Initialize Linear Module
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.normal_(self.fc2.bias.data)
        # If you want to guess age: output_dim=1
        # If you want to categorize age: output_dim = number of categories

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.dropout1(x[:, -1])
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

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


# Normalize in tensor
def normalize_tensor(tensor, minmax):
    min_val = minmax[0]
    max_val = minmax[1]
    arr = tensor.cpu().data.numpy()*(max_val-min_val) + min_val
    return arr


# Train
def train(start, rate, datax, datay):
    train_x_tensor, train_y_tensor = get_tensor(
    device, datax, datay, start, start + rate)
    return train_x_tensor, train_y_tensor

# Validation
def valid(start, rate, datax, datay):
    valid_x_tensor, valid_y_tensor = get_tensor(
    device, datax, datay, start, start + rate)
    return valid_x_tensor, valid_y_tensor

# Main
def wrapper(param, data_x, data_y, lr_gamma, hidden_dim, layers):

    device = param['device']
    input_dim = param['region_n']
    n_epochs = param['n_epochs']
    outputfolder = param['outputfolder']
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

    kf = KFold(n_splits=n_k_fold, shuffle=False)
    for train_index, valid_index in kf.split(data_x[0:train_num+valid_num, :, :]):

        criterion = torch.nn.MSELoss().to(device)
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
                train_x_tensor, train_y_tensor = train(tr*rate_tr, rate_tr, train_xdata, train_ydata)

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
                valid_x_tensor, valid_y_tensor = valid(va*rate_va, rate_va, valid_xdata, valid_ydata)
                valid_result = mynet(valid_x_tensor)
                loss_valid = criterion(valid_result, valid_y_tensor)
                valid_loss = valid_loss + loss_valid

            epoch_list.append(i)
            loss_list.append(valid_loss.item())
            step_list.append('validation')
            k_fold_list.append(k)

            if valid_loss.item() <= total_loss_valid_min:
                torch.save(mynet.state_dict(), './model_new/{}_{}.pt'.format(out_fname, k))
                total_loss_valid_min = valid_loss.item()

        init_weights(mynet)

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

    mynet.load_state_dict(torch.load('./model_new/{}_1.pt'.format(out_fname)))
    params1 = mynet.named_parameters()
    dict_params = dict(params1)

    for name1, param1 in params1:
        if name1 in dict_params:
            dict_params[name1].data.copy_(beta * param1.data)

    for i in range(1, n_k_fold):
        mynet.load_state_dict(torch.load('./model_new/{}_{}.pt'.format(out_fname, i + 1)))
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
             'label_fname': 'brain_entropy_model_valid_min201904041559.csv',
             'minmax_x': [4, 16789],  # x_values are between 4 and 16788.8
             'minmax_y': [10, 80],  # y_values are between 10 and 80
             'outputfolder': 'output',
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
