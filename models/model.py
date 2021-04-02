from models.brain_all_FC import All_fc
from models.brain_RNN import RNNClassifier
from models.brain_transformer import TransformerClassifier


def model(param, input_dim, hidden_dim, output_dim, layers,
          device):
    model_name = param['model']
    drop_prob = param['drop_p']
    time_length = param['time_len']
    bi = param['bidirection']
    nhead = param['n_head']

    if model_name == 'FC':
        net = All_fc(input_dim, hidden_dim, output_dim, layers,
                     drop_prob, time_length).to(device)
    elif model_name in ['RNN', 'LSTM', 'GRU']:
        net = RNNClassifier(input_dim, hidden_dim, output_dim, layers,
                            drop_prob, model_name, bi).to(device)
    elif model_name == 'Transformer':
        net = TransformerClassifier(input_dim, hidden_dim, output_dim, layers,
                                    nhead, drop_prob).to(device)
    else:
        raise ValueError("Check the model name!(RNN/ LSTM/ GRU/ FC/ Transformer)")

    net.init_weights()

    return net
