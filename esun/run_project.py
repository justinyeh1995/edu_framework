#!/usr/bin/env python
# coding: utf-8
from pl_module import MultiTaskModule

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

seed_everything(1, workers=True)

if __name__ == "__main__":
    
    config = {
            'hidden_dims': 64, 
            'n_layers': 2, 
            'cell': 'LSTM', 
            'bi': False, 
            'dropout': 0.5
    }
    module = MultiTaskModule(config)
    
    print('Module Built')
    logger = TensorBoardLogger('/home/ai/work/logs/tensorboard', 
                               'ncku_customer_embedding', 
                               default_hp_metric=False, 
                               log_graph=True)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum = True)
    early_stopping = EarlyStopping('best_val_auroc', mode='max', verbose=True)
    
    checkpoint = ModelCheckpoint(
        monitor='best_val_auroc',
        mode='max',
        dirpath='./checkpoint',
        filename='epoch{epoch:02d}-auroc{best_val_auroc:.2f}',
        save_last=True,
        auto_insert_metric_name=False,
        every_n_val_epochs=1,
        verbose=True
    )

    
    print("Model Architecture Logged")
    trainer = pl.Trainer(
        auto_scale_batch_size='power',
        auto_lr_find=True,
        logger = logger, 
        callbacks=[lr_monitor, early_stopping, checkpoint], 
        deterministic=True
    )
    
    trainer.tune(module)    
    print('Hyper-parameters tuned')
    trainer.fit(module)

    
    
# TODO: 
# - [V] 要用torch metric 才會快: https://torchmetrics.readthedocs.io/en/latest/?_ga=2.184197729.610530333.1621525374-364715702.1621241882
# - [ ] logging of hyperparameters: https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#logging-hyperparameters
#       - [V] show hparams and metric values on Tensorboard 
#       - [ ] logging of trial mode : success, fail, running, unknown 
# - [ ] Survey and organize the tools in https://pytorch-lightning.readthedocs.io/en/1.2.2/common/trainer.html (找出適合Debug用的工具) 
# - [ ] Adopt lightning Callbacks : 
#       - [V] EarlyStopping: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
#       - [V] LearningRateMonitor: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor
#       - [V] ModelCheckpoint: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
#       - [ ] Make sure we can load checkpoint for testing 
# - [ ] 實作lightning DataModule以進一步把Data的部分和pl_module解偶: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#why-do-i-need-a-datamodule
# - [ ] Incorporate with ray[tune]. Ref: https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html
# - [ ] In pl_module, need to have a strategy for splitting training into train and 'val', other than using 'test' for 'val'. 
# - [X] XXX: come up with some thing to do cross-validation. Ref: https://towardsdatascience.com/5x-faster-scikit-learn-parameter-tuning-in-5-lines-of-code-be6bdd21833c. (p.s., Cross validation is often not used for evaluating deep learning models because of the greater computational expense) 
# - [ ] Many speed up tips: https://pytorch-lightning.readthedocs.io/en/stable/benchmarking/performance.html