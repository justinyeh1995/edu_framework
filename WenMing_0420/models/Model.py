#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_lightning as pl


class ET_Rnn(torch.nn.Module):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0):
        super(ET_Rnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.cell = cell
        self.use_chid = use_chid
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
            hidden = Variable(torch.zeros(self.n_layers * (self.bi + 1), x.size(0), self.hidden_dims))
            context = Variable(torch.zeros(self.n_layers * (self.bi + 1), x.size(0), self.hidden_dims))
            ret = (hidden, context)
        elif self.cell == 'GRU':
            hidden = Variable(torch.zeros(self.n_layers * (self.bi + 1), x.size(0), self.hidden_dims))
            ret = hidden

        return ret

    def forward(self, x_dense, x_sparse):
        if self.use_chid:
            x = torch.cat([x_dense] + [embed(x_sparse[:, :, i + 1]) for i, embed in enumerate(self.embedding_list[1:])], dim=-1)
        else:
            x = torch.cat([x_dense] + [embed(x_sparse[:, :, i]) for i, embed in enumerate(self.embedding_list[:])], dim=-1)

        self.hidden = self.init_hidden(x)
        logits, self.hidden = self.rnn(x, self.hidden)
        # TODO: why rnn takes two input? what is the purpose of self.hidden ?
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


class MultiTaskModel(pl.LightningModule):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, out_dims=[1], n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0, lr=2e-3):
        self.dense_dims = dense_dims
        self.sparse_dims = sparse_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.n_layers = n_layers
        self.user_chid = use_chid
        self.cell = cell
        self.bi = bi
        self.dropout = dropout
        self.lr = lr

        super(MultiTaskModel, self).__init__()
        self.rnn = ET_Rnn(dense_dims, sparse_dims, hidden_dims, n_layers=n_layers, use_chid=use_chid,
                          cell=cell, bi=bi, dropout=dropout)

        self.mlps = nn.ModuleList([MLP(self.rnn.out_dim, hidden_dims=[self.rnn.out_dim // 2], out_dim=od) for od in out_dims])

        self.batch_cnt = 0

    def forward(self, x_dense, x_sparse):
        logits = self.rnn(x_dense, x_sparse)
        outs = [mlp(logits)for mlp in self.mlps]

        return outs

    def training_step(self, batch, batch_idx):
        self.batch_cnt += 1
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        objmean_hat, tscnt_hat, label_0_value = self(x_dense, x_sparse)
        label_0_hat = F.sigmoid(label_0_value)
        logs = {}
        objmean_loss = F.mse_loss(objmean_hat, objmean)
        logs['objmean_train_loss'] = objmean_loss
        self.logger.experiment.add_scalar("Loss/Train/objmean", objmean_loss, self.batch_cnt)
        tscnt_loss = F.mse_loss(tscnt_hat, tscnt)
        # print(tscnt_hat, tscnt)
        logs['tscnt_train_loss'] = tscnt_loss
        self.logger.experiment.add_scalar("Loss/Train/tscnt", tscnt_loss, self.batch_cnt)
        # print(label_0_hat, label_0)
        label_0_loss = F.binary_cross_entropy(label_0_hat, label_0)
        logs['label_0_train_loss'] = label_0_loss
        self.logger.experiment.add_scalar("Loss/Train/label_0", label_0_loss, self.batch_cnt)

        correct = ((label_0_hat > 0.5).float() == label_0).float().sum()
        self.logger.experiment.add_scalar("Accuracy/Train/label_0", correct / label_0.shape[0], self.batch_cnt)
        # TODO: define in training_step
        loss = objmean_loss
        # if self.current_epoch >= 20:
        loss += tscnt_loss
        # if self.current_epoch >= 40:
        loss += label_0_loss

        return {'loss': loss, 'log': logs, 'objmean_hat': objmean_hat, 'tscnt_hat': tscnt_hat, 'label_0_hat': label_0_hat}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)