import torch
import torch.nn as nn


# Model
class All_fc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, drop_prob):
        super(All_fc, self).__init__()
        self.num_layers = layers

        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(True),
                                    nn.Dropout(drop_prob))
        self.hidden = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden.append(nn.ReLU(True))
            self.hidden.append(nn.Dropout(drop_prob))
        self.fc1 = nn.Linear(hidden_dim, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.fc2 = nn.Linear(100, output_dim, bias=True)

        self.init_weights()

        # weight_init

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.Linear]:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param.data, 0.00)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param.data)

    def forward(self, inp):
        x = self.layer1(inp)
        for layer in self.hidden:
            x = layer(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = torch.squeeze(x)
        output = self.fc2(x)

        return output


# Put data in tensor
def get_tensor_fc(device, data_x, data_y, start_idx, end_idx):
    x_tensor = torch.FloatTensor(data_x[start_idx:end_idx, :, :]).to(device)
    y_tensor = torch.FloatTensor(data_y[start_idx:end_idx])
    y_tensor.unsqueeze_(-1)
    y_tensor = y_tensor.to(device)
    return x_tensor, y_tensor


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
def train_fc(device, start, rate, datax, datay):
    train_x_tensor, train_y_tensor = get_tensor_fc(
        device, datax, datay, start, start + rate)
    return train_x_tensor, train_y_tensor


# Validation
def valid_fc(device, start, rate, datax, datay):
    valid_x_tensor, valid_y_tensor = get_tensor_fc(
        device, datax, datay, start, start + rate)
    return valid_x_tensor, valid_y_tensor
