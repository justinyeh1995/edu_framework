import torch
from torch import nn

import pytorch_lightning as pl
from models.Model import MultiTaskModel


class MultiTasksLightningModel(pl.LightningModule):
    def __init__(self, dense_dims, sparse_dims, hidden_dims=64, out_dims=[1], n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0):
        super().__init__()
        self.model = MultiTaskModel(
            dense_dims,
            sparse_dims,
            hidden_dims=hidden_dims,
            out_dims=out_dims,
            n_layers=n_layers,
            use_chid=use_chid,
            cell=cell,
            bi=bi,
            dropout=dropout
        )
        self.epoch_cnt = 0
        self.criterion_reg = nn.MSELoss()
        self.criterion_clf = nn.CrossEntropyLoss()

    def forward(self, x):
        x_dense, x_sparse = x
        outputs = self.model(x_dense, x_sparse)
        return outputs

    def training_step(self, batch, batch_idx):
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        objmean_hat, tscnt_hat, label_0_hat = self([x_dense, x_sparse])
        if batch_idx == 0:
            self.epoch_cnt += 1
        loss = self.criterion_reg(objmean_hat, objmean)
        # TODO: define in training_step
        if self.epoch_cnt >= 20:
            loss += self.criterion_reg(tscnt_hat, tscnt)
        if self.epoch_cnt >= 40:
            loss += self.criterion_clf(label_0, label_0)
        return {'loss': loss, 'objmean_hat': objmean_hat, 'tscnt_hat': tscnt_hat, 'label_0_hat': label_0_hat}
    # def training_epoch_end(self, training_step_outputs):
    #    for out in training_step_outputs:

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-3)
