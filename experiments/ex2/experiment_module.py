#!/usr/bin/env python
# coding: utf-8
from common.utils import blockPrinting 
from common.pl_module import BaseMultiTaskModule, BaseMultiTaskDataModule
import torch.nn.functional as F

from experiments.ex1 import dataset_builder, preprocess

from experiments.ex1.model import MultiTaskModel


experiment_name = __file__.split("/")[-2] # same as the name of current folder 
experiment_group_name = 'rnn' # folder saving the data of all rnn experiments 

preprocessor = preprocess.PreProcessor(
    origin_path = 'data/source',
    sample_path = 'data/sample50',
    tmp_path = f'data/sample50/{experiment_group_name}/tmp', 
    result_path = f'data/sample50/{experiment_group_name}/result',
    n_chid_sample = 50, 
    window_size = 20, 
    test_size = 4 
)

dataset = dataset_builder.DatasetBuilder(
        tmp_path = f'data/{experiment_name}/tmp', 
        result_path = f'data/{experiment_name}/result', 
        numeric_cols = ['bnspt', 'timestamp_0', 'timestamp_1', 'objam'], 
        category_cols = ['chid', 'bnsfg', 'iterm', 'mcc', 'scity'], 
        dense_feat = ['bnspt', 'timestamp_0'], 
        sparse_feat = ['chid', 'bnsfg', 'iterm'], 
        USE_CHID = False
    )


(x_train, x_test, y_train, y_test, 
    feature_map, chid_to_nid_map, columns) = preprocessor.connect_pipeline()

train_dataset, test_dataset = dataset.connect_pipeline(
    x_train, x_test, y_train, y_test, 
        feature_map, chid_to_nid_map, columns)
# TODO: allow input of preprocessor and output of train_dataset and test_dataset 

@blockPrinting
def _get_data_dependent_model_parameters(dataset):
    # @DataDependent
    use_chid = dataset.USE_CHID
    dense_dims = dataset.dense_dims.run()[0]
    sparse_dims = dataset.sparse_dims.run()[0]

    num_y_data = len(dataset.processed_y_data.run())
    num_y_data = num_y_data//2

    ground_truths = dataset.processed_y_data.run()[-num_y_data:]
    out_dims = [y_data.shape[1] for y_data in ground_truths]

    class_outputs = [type(y_data[0].item())!=float for y_data in ground_truths]

    return {
        'dense_dims': dense_dims, 
        'sparse_dims': sparse_dims,
        'use_chid': use_chid, 
        'out_dims': out_dims,
        'class_outputs': class_outputs 
    }

class ExperimentConfig:
    # @ExperimentDependent 
    best_model_checkpoint = 'epoch07-loss0.00.ckpt'

    name = experiment_name 

    experiment_parameters = {
        "model_parameters": {
            "data_independent":{
                'hidden_dims': 64, 
                'n_layers': 2, 
                'cell': 'LSTM', 
                'bi': False
            },
            "data_dependent": _get_data_dependent_model_parameters(dataset)
        },
        "training_parameters":{
            "dropout": 0.5, 
            "warmup_epochs": 5, 
            "annealing_cycle_epochs": 40
        }
    }




class ExperimentalMultiTaskDataModule(BaseMultiTaskDataModule):
    # TODO: 
    # [ ] change prepare_data to download only script 
    # [ ] change setup to ETL script 
    # [ ] implement data_dependent parameters getter as follows: dm.num_classes, dm.width, dm.vocab
    # [V] allow batch_size to be saved as parameter 

    def __init__(self, batch_size = 64, num_workers = 4, pin_memory = False):
        super(ExperimentalMultiTaskDataModule, self).__init__(
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory
        )
        

    # @blockPrinting
    def prepare_data(self):
        # things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode. 
        # e.g., download 
        self.train_dataset = train_dataset.run()[0]
        self.test_dataset = test_dataset.run()[0]


class ExperimentalMultiTaskModule(BaseMultiTaskModule):
    
    def __init__(self, experiment_parameters, lr=2e-4):
        super(ExperimentalMultiTaskModule, self).__init__(experiment_parameters, lr=lr)

    def config_model(self, model_parameters, dropout):
        return MultiTaskModel(
                model_parameters,
                dropout = dropout
            )
    def config_loss_funcs(self):
        return [F.mse_loss, F.mse_loss, F.binary_cross_entropy]
    
    def config_task_names(self):
        return ['objmean', 'tscnt', 'label_0']

    def config_task_metrics(self):
        return {
            'objmean': ['mse', 'mae'], 
            'tscnt': ['mse', 'mae'], 
            'label_0': ['acc', 'auc']
        }
    '''
    If additional metrics are considered in the experiment, 
    we can override the following function with new metric calculator added.  

    def config_metric_calculators(self):
        from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy, AUROC
        return {
            'mse': lambda: MeanSquaredError(compute_on_step=False), 
            'mae': lambda: MeanAbsoluteError(compute_on_step=False), 
            'acc': lambda: Accuracy(compute_on_step=False),
            'auc': lambda: AUROC(compute_on_step=False, pos_label=1)
        }
    '''
    def batch_forward(self, batch): 
        # TODO: should be override >>
        # @DataDependent -> input and output definition
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        outputs = self(x_dense, x_sparse)
        ground_truths = objmean, tscnt, label_0
        return outputs, ground_truths

