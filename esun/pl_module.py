#!/usr/bin/env python
# coding: utf-8
import os
import sys

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics import AUROC

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import dataset_builder 

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

train_accuracy = Accuracy()
valid_accuracy  = Accuracy(compute_on_step=False)
train_auroc = AUROC(pos_label=1)
valid_auroc  = AUROC(compute_on_step=False, pos_label=1)

class MultiTaskModule(pl.LightningModule):
    
    @blockPrinting
    def __init__(self, config, batch_size = 64, lr=2e-3):
        
        super(MultiTaskModule, self).__init__()
        
        self.hidden_dims = config['hidden_dims']
        self.n_layers = config['n_layers']
        self.cell = config['cell']
        self.bi = config['bi']
        self.dropout = config['dropout']
        
        self.use_chid, self.dense_dims, self.sparse_dims, self.out_dims = self._get_data_dependent_hparams(
            dataset_builder
        )
        
        self.rnn = ET_Rnn(
            self.dense_dims, 
            self.sparse_dims, 
            self.hidden_dims, 
            n_layers=self.n_layers, 
            use_chid=self.use_chid,
            cell=self.cell, 
            bi=self.bi, 
            dropout=self.dropout
        )
        
        self.mlps = nn.ModuleList([
            MLP(
                self.rnn.out_dim, 
                hidden_dims=[self.rnn.out_dim // 2], 
                out_dim=od
            ) for od in self.out_dims
        ])
        
        self.batch_size = batch_size
        self.lr = lr

    @blockPrinting
    def _get_data_dependent_hparams(self, dataset_builder):
        use_chid = dataset_builder.USE_CHID
        dense_dims = dataset_builder.dense_dims.run()[0]
        sparse_dims = dataset_builder.sparse_dims.run()[0]
        out_dims = self._get_output_dims(dataset_builder)
        return use_chid, dense_dims, sparse_dims, out_dims 
    
    def _get_output_dims(self, dataset_builder):
        num_y_data = len(dataset_builder.processed_y_data.run())
        num_y_data = num_y_data//2
        return [y_data.shape[1] for y_data in dataset_builder.processed_y_data.run()[-num_y_data:]]

    
    def forward(self, x_dense, x_sparse):
        logits = self.rnn(x_dense, x_sparse)
        outs = [mlp(logits)for mlp in self.mlps]
        return outs
    
    def on_train_start(self):
        self._batch_cnt = 0
        self._architecture_logged = False
        
    def on_fit_start(self): # def on_train_start(self):
        self._batch_cnt = 0
        self._best_val_loss = 1e13
        self._best_val_auroc = 0.
        self.logger.log_hyperparams(
            params = {
                "hidden_dims": self.hidden_dims,
                "n_layers": self.n_layers, 
                "cell": self.cell,
                "bi": int(self.bi), 
                "dropout": self.dropout, 
                "use_chid" : int(self.use_chid), 
                "num_dense_feat": int(self.dense_dims), 
                "num_sparse_feat": len(self.sparse_dims),
                "num_tasks": len(self.out_dims),
                "lr": self.lr, 
                "batch_size": self.batch_size
            }, 
            metrics = {
                "best_val_loss": self._best_val_loss,
                "best_val_auroc": self._best_val_auroc
            }
        )
        print(self.hparams)

    def training_step(self, batch, batch_idx):
        
        losses_and_metrics = self._calculate_multitask_loss_and_metrics(batch, mode = 'train')
        
        loss = losses_and_metrics['objmean_loss']
        # if self.current_epoch >= 20:
        loss += losses_and_metrics['tscnt_loss']
        # if self.current_epoch >= 40:
        loss += losses_and_metrics['label_0_loss']
        
        for value_name, value in losses_and_metrics.items():
            self.logger.experiment.add_scalar(f'Train/{value_name}', value, self._batch_cnt)
        self.logger.experiment.add_scalar("Train/total_loss", loss, self._batch_cnt)
        
        if not self._architecture_logged: 
            self.logger.experiment.add_graph(self, [batch[0], batch[1]])
            print("Model Architecture Logged")
            self._architecture_logged = True
            self._fit_start = False
            
        self._batch_cnt += 1
        return {'loss': loss, 'log': losses_and_metrics}
    
    def validation_step(self, batch, batch_idx):
        
        losses_and_metrics = self._calculate_multitask_loss_and_metrics(batch, mode = 'valid')
        
        loss = losses_and_metrics['objmean_loss']
        # if self.current_epoch >= 20:
        loss += losses_and_metrics['tscnt_loss']
        # if self.current_epoch >= 40:
        loss += losses_and_metrics['label_0_loss']
        
        return {'val_loss': loss, 'val_log': losses_and_metrics}
    
    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            
            if avg_loss < self._best_val_loss:
                self._best_val_loss = avg_loss 
                self.log('best_val_loss', self._best_val_loss)
                
            self.logger.experiment.add_scalar("Val/total_loss", avg_loss, self.current_epoch)

            for value_name in ['objmean_loss', 'tscnt_loss', 'label_0_loss']:
                avg_value = torch.stack([x['val_log'][value_name] for x in outputs]).mean()
                self.logger.experiment.add_scalar(f"Val/{value_name}", avg_value, self.current_epoch)
                
            acc_value = valid_accuracy.compute()
            self.logger.experiment.add_scalar(f"Val/Acc", acc_value, self.current_epoch)
            
            auroc_value = valid_auroc.compute()
            if auroc_value > self._best_val_auroc:
                self._best_val_auroc = auroc_value
                self.log('best_val_auroc', self._best_val_auroc)
            self.logger.experiment.add_scalar(f"Val/AUROC", auroc_value, self.current_epoch)
            # self.logger.finalize('Running')
            
            
    
    def _calculate_multitask_loss_and_metrics(self, batch, mode = 'train'):
        assert mode == 'train' or mode == 'valid'
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        objmean_hat, tscnt_hat, label_0_value = self(x_dense, x_sparse)
        
        objmean_loss = F.mse_loss(objmean_hat, objmean)
        tscnt_loss = F.mse_loss(tscnt_hat, tscnt)
        label_0_hat = F.sigmoid(label_0_value)
        label_0_loss = F.binary_cross_entropy(label_0_hat, label_0)
        if mode == 'train':
            label_0_accuracy = train_accuracy(label_0_hat, label_0.int())
            # label_0_auroc = self.train_auroc(label_0_hat, label_0.int())
            return {
                'objmean_loss': objmean_loss, 
                'tscnt_loss': tscnt_loss, 
                'label_0_loss': label_0_loss, 
                'label_0_accuracy': label_0_accuracy
                # 'label_0_auroc': label_0_auroc
            }
        else:
            valid_accuracy(label_0_hat, label_0.int())
            valid_auroc(label_0_hat, label_0.int())
            return {
                'objmean_loss': objmean_loss, 
                'tscnt_loss': tscnt_loss, 
                'label_0_loss': label_0_loss
            }
        
    
    @blockPrinting
    def prepare_data(self):
        self.train_dataset = dataset_builder.train_dataset.run()[0]
        self.test_dataset = dataset_builder.test_dataset.run()[0]
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.train_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
        }

    
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
