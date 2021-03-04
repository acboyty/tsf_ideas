import torch 
import torch.nn as nn


class RNNNet(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, seq_len):
        super(RNNNet, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.seq_len = seq_len

        self._build()

    def _build(self):
        self.rnn = nn.GRU(self.input_dim, self.hid_dim)
        self.lin = nn.Linear(self.hid_dim, self.output_dim)

        self.ar = nn.Linear(self.seq_len, self.output_dim)

    def forward(self, x):
        # x [batch_size, seq_len, dim]

        output, hn = self.rnn(x.permute(1, 0, 2))
        hid = hn.squeeze(dim=0)  # hid [batch_size, dim]
        # TODO: the effect of pooling is not as good as hn
        # hid = torch.mean(output, dim=0) 
        out = self.lin(hid)  # out [batch_size, dim]

        # Auto Regression
        ar_out = self.ar(x[:, :, 0])

        out = out + ar_out

        return out


class RNNGATENet(nn.Module):
    def __init__(self, seq_len, hid_dim, output_dim):
        super(RNNGATENet, self).__init__()

        self.seq_len = seq_len
        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self._build()

    def _build(self):
        self.rnn = RNNNet(2, self.hid_dim, self.output_dim, self.seq_len)
        self.gate = nn.Linear(self.seq_len, self.output_dim)

        # self.ar = nn.Linear(self.seq_len, self.output_dim)

    def forward(self, x):
        # x [batch_size, seq_len, dim]
        inputs = x.clone()
        inputs[:, :, 1] = 0

        rnn_out = self.rnn(inputs)

        gate = self.gate(x[:, :, 1])
        # if self.training == False:
        #     torch.save(gate, './holiday_effect/gate.th')
        #     torch.save(self.gate.weight, './holiday_effect/gate.weight.th')

        out = rnn_out * gate
        # TODO: add tanh brings some optimization problem
        # out = rnn_out * torch.sigmoid(gate) * 2

        # # Auto Regression
        # ar_out = self.ar(x[:, :, 0])

        # out = out + ar_out

        return out
        