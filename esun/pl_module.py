#!/usr/bin/env python
# coding: utf-8
# TODO: 
# - [V] no need to have best_metric_value replacement (already handled by the lightning early_stopping and checkpoint callbacks)
# - [V] 把 model 和 lightning module分開 
# - [V] metrics定義在lightning module的train_start/validation_start/...裡
# - [X] showing loss in the progress bar (already shown in TensorBoard. need to re-generate files to show the correct arrangement)
# - [ ] Making Pytorch Lightning Module Model and Data Agnostic 
#       - [V] Identify functions related to Data Formate and label with @DataDependent
#       - [V] Identify functions related to Model and label with @ModelDependent
#       - [V] Resolve @ModelDependent 
#       - [ ] Resolve @DataDependent 
# - [ ] Add lr schedular warmup_epochs and max_epochs, and weight_decay as training parameters 
# - [ ] 把 metrics calculation和logging的部分抽離至callback 
# - Note: pin_memory = True only when GPU is available, or it may slowdown dramatically. 

import os
import sys
import gc 
import copy 

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


class MultiTaskDataModule(pl.LightningDataModule):
    # TODO: 
    # [ ] change prepare_data to download only script 
    # [ ] change setup to ETL script 
    # [ ] implement data_dependent parameters getter as follows: dm.num_classes, dm.width, dm.vocab
    # [ ] allow batch_size to be saved as parameter 
    def __init__(self, batch_size = 64):
        super(MultiTaskDataModule, self).__init__()
        self.batch_size = batch_size

    @blockPrinting
    def prepare_data(self):
        self.train_dataset = dataset_builder.train_dataset.run()[0]
        self.test_dataset = dataset_builder.test_dataset.run()[0]
        self.model_parameters = self._get_data_dependent_model_parameters()

    @blockPrinting
    def _get_data_dependent_model_parameters(self):
        # @DataDependent
        use_chid = dataset_builder.USE_CHID
        dense_dims = dataset_builder.dense_dims.run()[0]
        sparse_dims = dataset_builder.sparse_dims.run()[0]

        num_y_data = len(dataset_builder.processed_y_data.run())
        num_y_data = num_y_data//2

        ground_truths = dataset_builder.processed_y_data.run()[-num_y_data:]
        out_dims = [y_data.shape[1] for y_data in ground_truths]

        class_outputs = [type(y_data[0].item())!=float for y_data in ground_truths]

        return {
            'dense_dims': dense_dims, 
            'sparse_dims': sparse_dims,
            'use_chid': use_chid, 
            'out_dims': out_dims,
            'class_outputs': class_outputs 
        }

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4, pin_memory=False)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4, pin_memory=False)

class MultiTaskModule(pl.LightningModule):
    
    @blockPrinting
    def __init__(self, config, lr=2e-4):
        # @DataDependent 
        # [Resolved] @ModelDependent 
        super(MultiTaskModule, self).__init__()
        
        # Step 1: 載入參數
        self.config = config
        
        # Step 2: 彙整模型參數
        self.model_parameters = {
            **self.config['model_parameters']['data_independent'], 
            **self.config['model_parameters']['data_dependent']
            }

        # Step 3: 載入訓練參數 
        self.training_parameters = self.config['training_parameters'] 
        self.lr = lr 

        # Step 4: 建立模型 
        self.model = MultiTaskModel(
                self.model_parameters,
                dropout = self.config['training_parameters']['dropout']
            )
        

        # @DataDependent -> Task Dependent         
        self.loss_funcs = [F.mse_loss, F.mse_loss, F.binary_cross_entropy]
        self.task_names = ['objmean', 'tscnt', 'label_0']
        self.task_metric_names = {
            'objmean': ['mse', 'mae'], 
            'tscnt': ['mse', 'mae'], 
            'label_0': ['acc', 'auc']
        }
        self.metric_calculators = {
            'mse': lambda: MeanSquaredError(compute_on_step=False), 
            'mae': lambda: MeanAbsoluteError(compute_on_step=False), 
            'acc': lambda: Accuracy(compute_on_step=False),
            'auc': lambda: AUROC(compute_on_step=False, pos_label=1)
        }

        # Step 5: 彙整private variables 
        self._batch_cnt = 0
        
        self._metric_dict = {}
        self._metric_dict['val'] = self._initialize_metric_calculators(self.task_metric_names)
        
        self._warmup_epochs = self.training_parameters['warmup_epochs']
        self._annealing_cycle_epochs = self.training_parameters['annealing_cycle_epochs']

        self._architecture_logged = False 

    def _initialize_metric_calculators(self, task_metric_names):
        # [Resolved] @DataDependent -> metric definition 
        metric_computer_dict = copy.copy(task_metric_names) 
        for task_name in self.task_names:
            metric_computer_list = [(
                metric_name, 
                self.metric_calculators[metric_name]()
                ) for metric_name in metric_computer_dict[task_name]
            ]
            metric_computer_dict[task_name] = dict(metric_computer_list) 
        return metric_computer_dict

        
    def forward(self, *x):
        return self.model(*x) 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=self.lr)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
            warmup_epochs=self._warmup_epochs, 
            max_epochs=self._annealing_cycle_epochs 
            )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
        }

    def on_fit_start(self): 
        # [Resolved] @ModelDependent -> model parameters definition + what to log in TensorBoard  
        # [Resolved] @DataDependent -> some parameters are data-dependent 
        # [Resolved] + training parameters definition
        self._batch_cnt = 0
        params = {
                **self.training_parameters, 
                **self.model_parameters
            }
        params['batch_size'] = self.trainer.datamodule.batch_size 
        params['lr'] = self.lr 
        self.logger.log_hyperparams(
            params = params, 
            metrics = {
                "val_loss": float("Inf")
            }
        )

    def on_train_start(self):
        self._batch_cnt = 0
    
    def on_test_start(self):
        self._metric_dict['test'] = self._initialize_metric_calculators(self.task_metric_names) 
    
    def training_step(self, batch, batch_idx):
        # Step 1: calculate training loss and metrics 
        outputs, ground_truths = self._forward_batch_wise(batch)
        losses = self._calculate_losses(outputs, ground_truths)
        total_loss = losses['total_loss']
        # Step 2: log values
        for value_name, value in losses.items():
            self.logger.experiment.add_scalar(f'train/{value_name}', value, self._batch_cnt)
        # Step 3: increase batch count 
        self._batch_cnt += 1
        return {
            'loss': total_loss,
            'train_log': losses
        }
    
    def validation_step(self, batch, batch_idx):
        # Step 1: calculate validation loss and metrics 
        outputs, ground_truths = self._forward_batch_wise(batch)
        losses = self._calculate_losses(outputs, ground_truths)
        self._calculate_metrics(outputs, ground_truths, mode = 'val')
        # Step 2: log neural architecture 
        if not self._architecture_logged: 
            self.logger.experiment.add_graph(self, [batch[0], batch[1]])
            print("Model Architecture Logged")
            self._architecture_logged = True 
        return {'val_log': losses}
    
    def test_step(self, batch, batch_idx):
        # calculate training loss and metrics 
        outputs, ground_truths = self._forward_batch_wise(batch)
        losses = self._calculate_losses(outputs, ground_truths)
        self._calculate_metrics(outputs, ground_truths, mode = 'test')
        return {'test_log': losses}
    
    def _forward_batch_wise(self, batch): # TODO: rename as step 
        # TODO: should be override >>
        # @DataDependent -> input and output definition
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        outputs = self(x_dense, x_sparse)
        ground_truths = objmean, tscnt, label_0
        return outputs, ground_truths
        # << 
    def _calculate_losses(self, outputs, ground_truths):
        losses = dict([(
                task_name, 
                loss_fun(out, gnd)
            ) for task_name, loss_fun, out, gnd in zip(
                self.task_names, 
                self.loss_funcs, 
                outputs, 
                ground_truths
            )])
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss  
        return losses

    def _calculate_metrics(self, outputs, ground_truths, mode = 'train'):
        # [Resolved] @DataDependent -> input and output definition + metric definition 
        assert mode in self._metric_dict.keys()
        for task_name, out, gnd in zip(self.task_names, outputs, ground_truths): 
            for metric_name in self.task_metric_names[task_name]:
                if metric_name == 'acc' or metric_name == 'auc':
                    gnd = gnd.int() 
                self._metric_dict[mode][task_name][metric_name](out, gnd)

    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0:
            # avg_total_loss = self._calculate_losses_and_metrics_on_end(outputs, 'val')
            avg_losses = self._calculate_losses_on_end(outputs, 'val')
            self.log('val_loss', avg_losses['total_loss'])
            metric_values = self._calculate_metrics_on_end('val')
            self._log_losses_and_metrics_on_end(avg_losses, metric_values, 'val')
        gc.collect()
    
    def test_epoch_end(self, outputs):
        # self._calculate_losses_and_metrics_on_end(outputs, 'test', verbose = True)
        avg_losses = self._calculate_losses_on_end(outputs, 'test', verbose = True)
        metric_values = self._calculate_metrics_on_end('test', verbose = True)
        gc.collect()

    def _calculate_losses_on_end(self, outputs, mode, verbose = False):
        '''
        - input: 
            - outputs: a list of dictionaries each of which stored at the end of {train/val/test}_step.
            - mode: train, val, or test. 
            - verbose: whether to print the calculated result. 
        - output: 
            - avg_losses: a dictionary containing the average loss of each output the average total loss. 
        '''
        assert mode == 'train' or mode == 'val' or mode == 'test'
        avg_losses = {}
        for loss_name in outputs[0][f'{mode}_log'].keys():
            avg = torch.stack([x[f'{mode}_log'][loss_name] for x in outputs]).mean()
            avg_losses[loss_name] = avg
            if verbose:
                print(f'{loss_name}: {round(avg.item(),3)}')
        return avg_losses

    def _calculate_metrics_on_end(self, mode, verbose = False): 
        '''
        - input: 
            - mode: train, val, or test. 
            - verbose: whether to print the calculated result. 
        - output: 
            - metric_values: a dictionary containing the overall metric values, for each output, as defined in 'self._metrc_dict'.
        '''
        assert mode == 'train' or mode == 'val' or mode == 'test'
        metric_values = {}
        for task_name in self._metric_dict[mode].keys():
            for metric_name in self._metric_dict[mode][task_name].keys():
                # TODO: add loop 
                metric_value = self._metric_dict[mode][task_name][metric_name].compute()
                metric_values[f'{task_name}_{metric_name}'] = metric_value
                if verbose:
                    print(f'{task_name}_{metric_name}: {round(metric_value.item(),3)}')
        return metric_values 

    def _log_losses_and_metrics_on_end(self, avg_losses, metric_values, mode):
        '''
        - input: 
            - avg_losses: a dictionary containing the average loss of each output the average total loss. 
            - metric_values: a dictionary containing the overall metric values, for each output, as defined in 'self._metrc_dict'.
        '''
        assert mode == 'train' or mode == 'val' or mode == 'test'
        for loss_name in avg_losses.keys():
            self.logger.experiment.add_scalar(
                f"{mode}/{loss_name}", 
                avg_losses[loss_name], 
                self.current_epoch
                )

        for metric_name in metric_values.keys():
            self.logger.experiment.add_scalar(
                f"{mode}/{metric_name}", 
                metric_values[metric_name], 
                self.current_epoch
                )


    
        

    
    

  