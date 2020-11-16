import torch.nn as nn


# Model
class All_fc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, drop_prob,
                 length):
        super(All_fc, self).__init__()
        self.num_layers = layers
        self.input = input_dim * length

        self.layer1 = nn.Sequential(nn.Linear(self.input, hidden_dim),
                                    nn.ReLU(True),
                                    nn.Dropout(drop_prob))
        self.hidden = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.hidden.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden.append(nn.ReLU(True))
            self.hidden.append(nn.BatchNorm1d(hidden_dim))
            self.hidden.append(nn.Dropout(drop_prob))
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.Linear]:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param.data, 0.00)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param.data)

    def forward(self, inp):
        x = inp.view(-1, self.input)
        x = self.layer1(x)
        for layer in self.hidden:
            x = layer(x)
        output = self.fc(x)

        return output
