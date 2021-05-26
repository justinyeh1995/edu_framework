#!/usr/bin/env python
# coding: utf-8
import sys, getopt

from pl_module import MultiTaskModule

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

seed_everything(1, workers=True)

'''
# [ ] model parameters: 
# - [V] data-independent parts:
"hidden_dims": self.hidden_dims,
"n_layers": self.n_layers, 
"cell": self.cell,
"bi": int(self.bi), 
# - data-dependent parts:  
"use_chid" : int(self.use_chid), 
"num_dense_feat": int(self.dense_dims), 
"num_sparse_feat": len(self.sparse_dims),
"num_tasks": len(self.out_dims),
"num_class_outputs": sum(self.class_outputs), 
# - dropout  
"dropout": self.dropout, 
# training parameters 
"batch_size": self.batch_size,
"lr": self.lr, 
"warmup_epochs": self.warmup_epochs,
"annealing_cycle_epochs": self.annealing_cycle_epochs

'''
config = {
    "model_parameters": {
        "data_independent":{
            'hidden_dims': 64, 
            'n_layers': 2, 
            'cell': 'LSTM', 
            'bi': False
        },
    },
    "training_parameters":{
        "dropout": 0.5, 
        "warmup_epochs": 5, 
        "annealing_cycle_epochs": 40
    }
}

module = MultiTaskModule(config)
print('Module Built')
logger = TensorBoardLogger('/home/ai/work/logs/tensorboard', 
                           'ncku_customer_embedding', 
                           default_hp_metric=False, 
                           log_graph=True)

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath='./checkpoint',
    filename='epoch{epoch:02d}-loss{best_val_loss:.2f}',
    save_last=True,
    auto_insert_metric_name=False,
    every_n_val_epochs=1,
    verbose=True
)

lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum = True)
early_stopping = EarlyStopping('val_loss', mode='min', verbose=True)

if __name__ == "__main__":
    # Load run_mode Argument 
    try:
        opts, args = getopt.getopt(sys.argv[1:],"m:")
        print(opts)
    except getopt.GetoptError:
        print('test.py -m <run_mode>')
        sys.exit(2)
    run_mode = dict(opts)['-m'] 
    print("run_mode:", run_mode) 
    
    if run_mode != 'fit1batch' and run_mode != 'fastdebug' and run_mode != 'train' and run_mode != 'test':
        print('test.py -m <run_mode>')
        print('<run_mode> should be "fit1batch", "fastdebug", "train", or "test"')
        sys.exit(2)
    # Start Running ... 
    if run_mode == 'fit1batch':
        # Overfit Test:
        # Note: Why you should always overfit a single batch to debug your deep learning model
        # https://www.youtube.com/watch?v=nAZdK4codMk
        trainer = pl.Trainer(
            auto_scale_batch_size='power',
            auto_lr_find=True, 
            logger = logger, 
            callbacks=[checkpoint, lr_monitor], 
            deterministic=True,
            num_sanity_val_steps=1, 
            overfit_batches=1
        )
        trainer.tune(module)
        # module.batch_size = 64
        # module.lr = 2e-2
        trainer.fit(module)
    elif run_mode == 'fastdebug' or run_mode == 'train':
        trainer = pl.Trainer(
            auto_scale_batch_size='power',
            auto_lr_find=True,
            logger = logger, 
            callbacks=[checkpoint, lr_monitor, early_stopping], 
            deterministic=True,
            num_sanity_val_steps=1, # Debug 
            fast_dev_run=(run_mode == 'fastdebug')
        )
        trainer.tune(module)
        trainer.fit(module)
    elif run_mode == 'test':
        trainer = pl.Trainer(
            logger = logger, 
            callbacks=[checkpoint], 
            deterministic=True
        )
        trainer.test(module)
    
# TODO: 
# - [V] 要用torch metric 才會快: https://torchmetrics.readthedocs.io/en/latest/?_ga=2.184197729.610530333.1621525374-364715702.1621241882
# - [ ] logging of hyperparameters: https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#logging-hyperparameters
#       - [V] show hparams and metric values on Tensorboard 
#       - [ ] logging of trial mode : success, fail, running, unknown 
# - [V] Survey and organize the tools in https://pytorch-lightning.readthedocs.io/en/1.2.2/common/trainer.html (找出適合Debug用的工具) 
#       - [V] fast_dev_run 
#       - [V] overfit_batches
# - [V] Adopt lightning Callbacks : 
#       - [V] EarlyStopping: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
#       - [V] LearningRateMonitor: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor
#       - [V] ModelCheckpoint: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
#       - [V] Make sure we can load checkpoint for testing 
# - [ ] 實作lightning DataModule以進一步把Data的部分和pl_module解偶: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#why-do-i-need-a-datamodule
# - [ ] Incorporate with ray[tune]. Ref: https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html
# - [ ] In pl_module, need to have a strategy for splitting training into train and 'val', other than using 'test' for 'val'. 
# - [X] XXX: come up with some thing to do cross-validation. Ref: https://towardsdatascience.com/5x-faster-scikit-learn-parameter-tuning-in-5-lines-of-code-be6bdd21833c. (p.s., Cross validation is often not used for evaluating deep learning models because of the greater computational expense) 
# - [ ] Many speed up tips: https://pytorch-lightning.readthedocs.io/en/stable/benchmarking/performance.html


# Notes: 

# # num_sanity_val_steps should be set so that 
# 1. model architecture can be plot in the beginning. 
# 2. validation step can be tested before training start. 
# 