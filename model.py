import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x_out = self.rnn(x_packed)[0]
        x_padded = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]
        return x_padded


class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

        self.inp = nn.Linear(params.d_in, params.d_rnn, bias=False)

        if params.rnn_n_layers > 0:
            self.rnn = RNN(params.d_rnn, params.d_rnn, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=0.0)

        d_rnn_out = params.d_rnn * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.d_rnn
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=0.0)

    def forward(self, x, x_len):
        x = self.inp(x)
        if self.params.rnn_n_layers > 0:
            x = self.rnn(x, x_len)
        y = self.out(x)
        return y
