#!/usr/bin/env python
# coding: utf-8
from pl_module import MultiTaskModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    module = MultiTaskModule(hidden_dims=64, n_layers=2, cell='GRU', bi=False, dropout=0.1)

    print('Module Built')
    logger = TensorBoardLogger('/home/ai/work/logs/tensorboard', 'ncku_customer_emebedding')
    trainer = pl.Trainer(
        auto_scale_batch_size='power',
        auto_lr_find=True,
        # stochastic_weight_avg=True,
        logger = logger
    )
    trainer.tune(module)
    print('Hyper-parameters tuned:')
    print(module.hparams)
    trainer.fit(module)

