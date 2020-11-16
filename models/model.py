from brain_all_FC import All_fc
from brain_RNN import RNNClassifier


def model(param, input_dim, hidden_dim, output_dim, layers,
          device):
    model_name = param['model']
    drop_prob = param['drop_p']
    time_length = param['time_len']
    bi = param['bidirection']

    if model_name == 'FC':
        net = All_fc(input_dim, hidden_dim, output_dim, layers,
                     drop_prob, time_length).to(device)
    elif model_name == 'RNN' or model_name == 'LSTM' or model_name == 'GRU':
        net = RNNClassifier(input_dim, hidden_dim, output_dim, layers,
                            drop_prob, model_name, bi).to(device)
    else:
        raise ValueError("Check the model name!(RNN/ LSTM/ GRU)")

    net.init_weights()

    return net
