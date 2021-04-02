import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, nhead, drop_prob):
        super(TransformerClassifier, self).__init__()
        if hidden_dim % nhead != 0:
            raise ValueError('hidden_dim must be divisible by nhead')

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(hidden_dim, drop_prob)
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead,
                                                 hidden_dim, drop_prob)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers)
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=True)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        r'''
            Along with the input sequence, a square attention mask is required
            because the self-attention layers in nn.TransformerEncoder
            are only allowed to attend the earlier positions in the sequence.
        '''

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))\
            .masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.Linear]:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.uniform_(param.data)
                    elif 'weight' in name:
                        nn.init.xavier_uniform_(param.data)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.fc1(src)
        pos_en_out = self.pos_encoder(src)
        tf_out = self.transformer_encoder(pos_en_out, self.src_mask)
        tf_out = tf_out[:, -1]
        fc2_out = self.relu(self.bn1(self.fc2(tf_out)))
        output = self.fc3(fc2_out)

        return output


class PositionalEncoding(nn.Module):
    r'''
        PositionalEncoding module injects some information about
        the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as
        the input sequence so that the two can be summed.
        Here, use sine and cosine functions of different frequencies.
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        r'''
            The information of elements:
            d_model: data dimension [:, :, d_model]
                     (eg: Number of characters in a sequence
                          + the ending character)
            pe: positional encoding results
                torch.Size([5000, d_model)
            position: the position of each character in the sequence
                      torch.Size([5000, 1])
            div_term: Mathematical transformation of sine and cosine functions
                      eg: sin(pos/10000^(2i/d_model))
                      i: represents the position of the char vector
                      so, 1/10000^(2i/d_model) = e^(log(10000^(−2i/d_model))
                      = e^((−2i/d_model)∗log^10000)
                      = e^(2i∗(−log^10000/d_model))
        '''
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
