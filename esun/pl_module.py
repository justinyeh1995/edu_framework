#!/usr/bin/env python
# coding: utf-8
# TODO: 
# - [V] no need to have best_metric_value replacement (already handled by the lightning early_stopping and checkpoint callbacks)
# - [V] 把 model 和 lightning module分開 
# - [V] metrics定義在lightning module的train_start/validation_start/...裡
# - [X] showing loss in the progress bar (already shown in TensorBoard. need to re-generate files to show the correct arrangement)
# - [ ] Making Pytorch Lightning Module Model and Data Agnostic 
#       - [ ] Identify functions related to Data Formate 
#       - [ ] Identify functions related to Model 
# - [ ] Add lr schedular warmup_epochs and max_epochs, and weight_decay as training parameters 
# - [ ] 把 metrics calculation和logging的部分抽離至callback 
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy, AUROC

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import dataset_builder
from model import MultiTaskModel

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



class MultiTaskModule(pl.LightningModule):
    
    @blockPrinting
    def __init__(self, config, batch_size = 64, lr=2e-4):
    	# @DataDependent 
        # @ModelDependent 
        super(MultiTaskModule, self).__init__()
        
        # Step 1: 載入模型參數 # @ModelDependent
        self.hidden_dims = config['hidden_dims']
        self.n_layers = config['n_layers']
        self.cell = config['cell']
        self.bi = config['bi']
        
        # Step 2: 載入data-dependent參數 # @DataDependent
        self.dense_dims, self.sparse_dims, self.use_chid, self.out_dims = self._get_data_dependent_hparams(
            dataset_builder
        )
        
        # Step 3: 載入訓練參數 
        self.dropout = config['dropout'] # @ModelDependent
        self.batch_size = batch_size
        self.lr = lr
 
        # Step 4: 建立模型 # @ModelDependent
        self.model = MultiTaskModel(
	        	self.hidden_dims, 
	        	self.n_layers, 
	        	self.cell, 
	        	self.bi, 
	        	self.dense_dims, 
	        	self.sparse_dims, 
	        	self.use_chid, 
	        	self.out_dims, 
	        	dropout = self.dropout
        	)
        
        self._batch_cnt = 0
        self._architecture_logged = False
        
        self._metric_dict = {}
        self._metric_dict['train'] = {}
        self._metric_dict['val'] = {}
        self._metric_dict['test'] = {}
        
    @blockPrinting
    def _get_data_dependent_hparams(self, dataset_builder):
    	# @DataDependent
        use_chid = dataset_builder.USE_CHID
        dense_dims = dataset_builder.dense_dims.run()[0]
        sparse_dims = dataset_builder.sparse_dims.run()[0]

        num_y_data = len(dataset_builder.processed_y_data.run())
        num_y_data = num_y_data//2

        out_dims = [y_data.shape[1] for y_data in dataset_builder.processed_y_data.run()[-num_y_data:]]
        return dense_dims, sparse_dims, use_chid, out_dims 
    
    
    def forward(self, *x):
        return self.model(*x) 
    
    def on_train_start(self):
        self._batch_cnt = 0
        self._architecture_logged = False
        
        
        self._metric_dict['train'] = self._initialize_metric_calculators()
        
    def on_validation_start(self):
        self._metric_dict['val'] = self._initialize_metric_calculators()

    def on_test_start(self):
        self._metric_dict['test'] = self._initialize_metric_calculators()
        
    def on_fit_start(self): # def on_train_start(self):
    	# @ModelDependent
    	# @DataDependent
        self._batch_cnt = 0
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
                "val_loss": float("Inf")
            }
        )

    def training_step(self, batch, batch_idx):
        
        # Step 1: calculate training loss and metrics 
        losses_and_metrics = self._calculate_losses_and_metrics_step_wise(batch, mode = 'train')
        total_loss = losses_and_metrics['total_loss']
        # Step 2: log values 
        for value_name, value in losses_and_metrics.items():
            self.logger.experiment.add_scalar(f'train/{value_name}', value, self._batch_cnt)
        
        # Step 3: log neural architecture # TODO: [ ] move to sanity validation step 
        if not self._architecture_logged: 
            self.logger.experiment.add_graph(self, [batch[0], batch[1]])
            print("Model Architecture Logged")
            self._architecture_logged = True
            
        # Step 4: increase batch count 
        self._batch_cnt += 1
        
        return {
            'loss': total_loss,
            'train_log': losses_and_metrics
        }
    
    def validation_step(self, batch, batch_idx):
        # Step 1: calculate training loss and metrics 
        losses_and_metrics = self._calculate_losses_and_metrics_step_wise(batch, mode = 'val')
        
        # Step 2: log total loss         
        self.log('val_loss', losses_and_metrics['total_loss'])
        return {'val_log': losses_and_metrics}
    
    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0:
            # Step 1: calculate average loss and metrics
            avg_total_loss = self._calculate_losses_and_metrics_on_end(outputs, 'val')
                
    def test_step(self, batch, batch_idx):
        # Step 1: calculate training loss and metrics 
        losses_and_metrics = self._calculate_losses_and_metrics_step_wise(batch, mode = 'test')
        return {'test_loss': losses_and_metrics}
        
    def test_epoch_end(self, outputs):
        self._calculate_losses_and_metrics_on_end(outputs, 'test', verbose = True)
        
    def _calculate_losses_and_metrics_on_end(self, outputs, mode, verbose = False):
        assert mode == 'train' or mode == 'val' or mode == 'test' 
        avg_losses = {}
        for loss_name in outputs[0][f'{mode}_log'].keys():
            avg = torch.stack([x[f'{mode}_log'][loss_name] for x in outputs]).mean()
            avg_losses[loss_name] = avg
            if verbose:
                print(f'{loss_name}: {round(avg,3)}')

        avg_total_loss = avg_losses['total_loss']

        metric_values = {}
        for metric_name in self._metric_dict[mode].keys():
            metric_values[metric_name] = self._metric_dict[mode][metric_name].compute()
            if verbose:
                print(f'{metric_name}: {round(metric_values[metric_name],3)}')
        
        if mode == 'train' or mode == 'val':
            # Step 2: log values 
            for loss_name in avg_losses.keys():
                self.logger.experiment.add_scalar(f"{mode}/{loss_name}", avg_losses[loss_name], self.current_epoch)

            for metric_name in metric_values.keys():
                self.logger.experiment.add_scalar(f"{mode}/{metric_name}", metric_values[metric_name], self.current_epoch)
            
        return avg_total_loss
        
    
    def _calculate_losses_and_metrics_step_wise(self, batch, mode = 'train'):
    	# @DataDependent
        assert mode == 'train' or mode == 'val' or mode == 'test'
        # TODO: [ ] put the following three lines into batch-wise forward 
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        objmean_hat, tscnt_hat, label_0_value = self(x_dense, x_sparse)
        label_0_hat = torch.sigmoid(label_0_value)
        
        objmean_loss = F.mse_loss(objmean_hat, objmean)
        tscnt_loss = F.mse_loss(tscnt_hat, tscnt)
        label_0_loss = F.binary_cross_entropy(label_0_hat, label_0)
        
        total_loss = objmean_loss + tscnt_loss + label_0_loss
        
        self._metric_dict[mode]['mse_objmean'](objmean_hat, objmean)
        self._metric_dict[mode]['mae_objmean'](objmean_hat, objmean)
        
        self._metric_dict[mode]['mse_tscnt'](tscnt_hat, tscnt)
        self._metric_dict[mode]['mae_tscnt'](tscnt_hat, tscnt)
        
        self._metric_dict[mode]['acc_label_0'](label_0_hat, label_0.int())
        self._metric_dict[mode]['auc_label_0'](label_0_hat, label_0.int())
        
        return {
            'total_loss': total_loss, 
            'objmean_loss': objmean_loss, 
            'tscnt_loss': tscnt_loss, 
            'label_0_loss': label_0_loss
        }
        
    def _initialize_metric_calculators(self):
    	# @DataDependent
        return {
            'mse_objmean': MeanSquaredError(compute_on_step=False),
            'mae_objmean': MeanAbsoluteError(compute_on_step=False),
            'mse_tscnt': MeanSquaredError(compute_on_step=False),
            'mae_tscnt': MeanAbsoluteError(compute_on_step=False),
            'acc_label_0': Accuracy(compute_on_step=False), 
            'auc_label_0': AUROC(compute_on_step=False, pos_label=1)
        }
    
    @blockPrinting
    def prepare_data(self):
        self.train_dataset = dataset_builder.train_dataset.run()[0]
        self.test_dataset = dataset_builder.test_dataset.run()[0]
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4, pin_memory=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=1000)
        return {
            "optimizer": optimizer,#  
            "lr_scheduler": lr_scheduler
        }

  