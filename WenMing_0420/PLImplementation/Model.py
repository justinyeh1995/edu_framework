#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
from torch.autograd import Variable


class ET_Rnn(torch.nn.Module):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0, device='cpu'):
        super(ET_Rnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.cell = cell
        self.use_chid = use_chid
        self.device = device
        self.bi = bi

        self.embedding_list = nn.ModuleList([nn.Embedding(fd, ed, padding_idx=0) for fd, ed in sparse_dims])

        if use_chid:
            rnn_in_dim = dense_dims + sum([ed for fd, ed in sparse_dims[1:]])
            self.out_dim = hidden_dims * (bi + 1) + sparse_dims[0][1]  # chid embed dim
            self.user_layer = nn.Linear(sparse_dims[0][1], sparse_dims[0][1])

        else:
            rnn_in_dim = dense_dims + sum([ed for fd, ed in sparse_dims[:]])
            self.out_dim = hidden_dims * (bi + 1)

        if self.cell == 'LSTM':
            self.rnn = nn.LSTM(rnn_in_dim, hidden_dims, n_layers, batch_first=True, bidirectional=bi, dropout=dropout)
        elif self.cell == 'GRU':
            self.rnn = nn.GRU(rnn_in_dim, hidden_dims, n_layers, batch_first=True, bidirectional=bi, dropout=dropout)

        self.init_embedding()

    def init_embedding(self):
        for embed in self.embedding_list:
            embed.reset_parameters()

    def init_hidden(self, x):
        if self.cell == 'LSTM':
            hidden = Variable(torch.zeros(self.n_layers * (self.bi + 1), x.size(0), self.hidden_dims)).to(self.device)
            context = Variable(torch.zeros(self.n_layers * (self.bi + 1), x.size(0), self.hidden_dims)).to(self.device)
            ret = (hidden, context)
        elif self.cell == 'GRU':
            hidden = Variable(torch.zeros(self.n_layers * (self.bi + 1), x.size(0), self.hidden_dims)).to(self.device)
            ret = hidden

        return ret

    def forward(self, x_dense, x_sparse):
        if self.use_chid:
            x = torch.cat([x_dense] + [embed(x_sparse[:, :, i + 1]) for i, embed in enumerate(self.embedding_list[1:])], dim=-1)
        else:
            x = torch.cat([x_dense] + [embed(x_sparse[:, :, i]) for i, embed in enumerate(self.embedding_list[:])], dim=-1)

        self.hidden = self.init_hidden(x)
        logits, self.hidden = self.rnn(x, self.hidden)

        if self.use_chid:
            user_embed = self.user_layer(self.embedding_list[0](x_sparse[:, 0, 0]))
            last_logits = torch.cat([logits[:, -1], user_embed], dim=-1)
        else:
            last_logits = logits[:, -1]

        return last_logits


# In[ ]:


class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims=[1], out_dim=1):
        super(MLP, self).__init__()
        hidden_dims = [input_dims] + hidden_dims

        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(idim, odim),
                nn.ReLU()
            ) for idim, odim in zip(hidden_dims[:-1], hidden_dims[1:])
        ])

        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        out = self.layers(x)
        out = self.out_layer(out)

        return out


# In[ ]:


class MultiTaskModel(nn.Module):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, out_dims=[1], n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0, device='cpu'):
        super(MultiTaskModel, self).__init__()
        self.rnn = ET_Rnn(dense_dims, sparse_dims, hidden_dims, n_layers=n_layers, use_chid=use_chid,
                          cell=cell, bi=bi, dropout=dropout, device=device)

        self.mlps = nn.ModuleList([MLP(self.rnn.out_dim, hidden_dims=[self.rnn.out_dim // 2], out_dim=od) for od in out_dims])

    def forward(self, x_dense, x_sparse):
        logits = self.rnn(x_dense, x_sparse)
        outs = [mlp(logits)for mlp in self.mlps]

        return outs
