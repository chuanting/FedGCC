# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.close_size
        self.hidden1 = nn.Linear(self.input_size, args.hidden_dim, bias=True)
        self.hidden2 = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.hidden3 = nn.Linear(args.hidden_dim, args.out_dim, bias=True)

    def forward(self, x):
        x = torch.relu(self.hidden1(x.squeeze()))
        x = torch.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x


class MyMLP(nn.Module):
    def __init__(self, args):
        super(MyMLP, self).__init__()
        self.args = args
        self.net = nn.Sequential()
        self.net.add_module('init', nn.Linear(args.close_size, args.hidden_dim))
        self.net.add_module('relu_init', nn.ReLU(inplace=True))
        for l in range(self.args.num_layers - 2):
            self.net.add_module('layer_{:}'.format(l+2), nn.Linear(args.hidden_dim, args.hidden_dim))
            self.net.add_module('relu_{:}'.format(l+2), nn.ReLU(inplace=True))
        self.net.add_module('last', nn.Linear(args.hidden_dim, args.out_dim))

    def forward(self, x):
        x = self.net(x.squeeze())
        return x


class LinearRegression(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, out_dim, bias=True)

    def forward(self, x):
        x = self.hidden(x.squeeze())
        return x


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.num_layers = args.num_layers
        self.close_size = args.close_size
        self.device = 'cuda' if args.gpu else 'cpu'

        self.lstm_1 = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              batch_first=True, dropout=0.2)
        self.lstm_2 = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              batch_first=True, dropout=0.2)

        self.linear_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, xc, xp=None):
        bz = xc.size(0)
        h0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)

        self.lstm_close.flatten_parameters()
        self.lstm_period.flatten_parameters()

        xc_out, xc_hn = self.lstm_close(xc, (h0, c0))
        x = xc_out[:, -1, :]
        # out = xc_out
        if self.period_size > 0:
            xp_out, xp_hn = self.lstm_period(xp, (h0, c0))
            y = xp_out[:, -1, :]
        out = x + y
        out = self.linear_layer(out)
        # out = torch.sigmoid(out)
        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        # w = self.att_weights.repeat(batch_size, 1, 1)
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )
        # weights = torch.bmm(inputs, w)

        attentions = torch.softmax(torch.relu(weights.squeeze()), dim=-1)
        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class MyLSTM(nn.Module):
    def __init__(self, args, hidden_dim=128, lstm_layer=2, dropout=0.2):
        super(MyLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_dim = 1
        self.hidden_dim = hidden_dim
        self.args = args

        self.lstm1 = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=lstm_layer,
                             bidirectional=False,
                             dropout=dropout, batch_first=True)
        self.atten1 = Attention(self.hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=2,
                             bidirectional=False,
                             dropout=dropout, batch_first=True)
        self.atten2 = Attention(self.hidden_dim, batch_first=True)

        self.lstm3 = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=2,
                             bidirectional=False,
                             dropout=dropout, batch_first=True)
        self.atten3 = Attention(self.hidden_dim, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, xc):
        with torch.autograd.set_detect_anomaly(True):
            self.lstm1.flatten_parameters()
            self.lstm2.flatten_parameters()
            self.lstm3.flatten_parameters()
            out1, (h_n, c_n) = self.lstm1(xc)
            x1, _ = self.atten1(out1)  # skip connect

            out2, (h_n, c_n) = self.lstm2(xc)
            x2, _ = self.atten2(out2)

            out3, (h_n, c_n) = self.lstm3(xc)
            x3, _ = self.atten3(out2)

            z = x1 + x2 + x3
            z = self.fc(z)
        return z


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp=6, nhead=6, nlayers=4, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layers = TransformerEncoderLayer(d_model=ninp, dim_feedforward=4*ninp, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        # src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output