#!/usr/bin/env python
# coding: utf-8
import os
import gc
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler


from models.Model import MultiTaskModel
# from PLImplementation.Trainer import Trainer

from ETL import ETLProcesses
from ETL.ETLBase import ETLPro, SelectResult

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# from MultiTasksLightningModel import MultiTasksLightningModel


class ProcessX(ETLPro):
    def process(self, inputs):
        x_train, x_test, sparse_index, dense_index = inputs
        print('Processing x_train')
        x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_sparse, x_train_dense = self._process_x_data(x_train, sparse_index, dense_index, x_scaler, mode='train')
        print('Processing x_test')
        x_test_sparse, x_test_dense = self._process_x_data(x_test, sparse_index, dense_index, x_scaler, mode='test')
        return [x_train_sparse, x_train_dense, x_test_sparse, x_test_dense]

    def _process_x_data(self, x, sparse_index, dense_index, x_scaler, mode='train'):
        w_size = x.shape[1]
        # window size of the time series data, x_train

        print('Split Sparse and Dense Parts and Apply np.float64 and np.int64, respectively')
        x_sparse = x[:, -w_size:, sparse_index].astype(np.int64)  # split sparse feature
        x_dense = x[:, -w_size:, dense_index].astype(np.float64)  # split dense feature
        gc.collect()
        print('apply Log(1+x) transformation to the dense features')
        x_dense = np.log1p(x_dense - x_dense.min(axis=0))
        gc.collect()
        print('apply MinMaxScale((0,1)) to the dense features')

        # transform:
        # 把前兩維濃縮為一維, 因為MinMaxScaler不支援三維以上Tensor
        # 還原為原本的shape
        if mode == 'train':
            x_dense = x_scaler.fit_transform(x_dense.reshape(-1, x_dense.shape[-1])).reshape(x_dense.shape)
        else:
            x_dense = x_scaler.transform(x_dense.reshape(-1, x_dense.shape[-1])).reshape(x_dense.shape)
        gc.collect()
        return x_sparse, x_dense


class ProcessY(ETLPro):
    def process(self, inputs):
        Y_train, Y_test, columns = inputs
        objmean_scaler = MinMaxScaler((0, 1))
        train_objmean, train_tscnt, train_label_0 = self._process_y_data(Y_train, columns["y_columns"], objmean_scaler, mode='train')
        test_objmean, test_tscnt, test_label_0 = self._process_y_data(Y_test, columns["y_columns"], objmean_scaler, mode='test')
        return [train_objmean, train_tscnt, train_label_0, test_objmean, test_tscnt, test_label_0]

    def _process_y_data(self, Y, y_columns, objmean_scaler, mode='train'):
        '''
        input:
         - Y: Y tensor (e.g., Y_train or Y_test)
         - columns: the y_columns in saved columns
        output:
         - target tensors: objmean, tscnt, label_0
        '''
        print('Convert Numeric Y values to np.float64 for Regression')
        index = y_columns.index('objam_sum')
        objsum = Y[:, [index]].astype(np.float64)

        index = y_columns.index('objam_mean')
        objmean = Y[:, [index]].astype(np.float64)

        index = y_columns.index('trans_count')
        tscnt = Y[:, [index]].astype(np.float64)
        gc.collect()
        print('Apply Log(1+x) Transformation')

        objsum = np.log1p(objsum)
        objmean = np.log1p(objmean)
        tscnt = np.log1p(tscnt)
        gc.collect()
        print('Apply  MinMaxScaler((0,1)) to objmean')
        # TODO: why apply MinMaxScaler on obj mean?
        if mode == 'train':
            objmean = objmean_scaler.fit_transform(objmean)
        else:
            objmean = objmean_scaler.transform(objmean)
        gc.collect()
        # classfication :
        print('Convert objsum to class target: whether or not objsum > 0 ')
        bounds = [0]
        lable_trans = np.vectorize(lambda x: sum([x > bound for bound in bounds]))
        label_0 = lable_trans(objsum)
        gc.collect()
        return objmean, tscnt, label_0


class GenerateSparseDenseSetting(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, category_cols=[], sparse_feat=[], numeric_cols=[], dense_feat=[], USE_CHID=True):
        super(GenerateSparseDenseSetting, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.category_cols = category_cols
        self.sparse_feat = sparse_feat
        self.numeric_cols = numeric_cols
        self.dense_feat = dense_feat
        self.USE_CHID = USE_CHID

    def process(self, inputs):
        feat_mapper = inputs[0]
        feat_mapper = self._remove_unused_sparse_feat_mapper(feat_mapper, self.sparse_feat)
        chid_mapper = np.load(os.path.join('../data', 'sample_idx_map.npy'), allow_pickle=True).item()
        sparse_dims, sparse_index = self._get_sparse_dims(self.category_cols, self.sparse_feat, chid_mapper, feat_mapper, USE_CHID=USE_CHID)
        dense_dims, dense_index = self._get_dense_dims(self.numeric_cols, self.dense_feat)
        return [sparse_dims, sparse_index, dense_dims, dense_index]

    def _remove_unused_sparse_feat_mapper(self, feat_mapper, sparse_feat):
        keys = list(feat_mapper.keys())
        for key in keys:
            if key not in sparse_feat:
                del feat_mapper[key]
        return feat_mapper

    def _get_sparse_dims(self, category_cols, sparse_feat, chid_mapper, feat_mapper, USE_CHID=True, chid_embed_dim=64, feat_embed_dim=16):
        '''
        inputs:
         - sparse_feat: the selected sparse feature
         - USE_CHID: whether to include chid
         - chid_embed_dim: the dimension of user embedding
         - feat_embed_dim: the dimension of dense feature embedding

        outputs:
        - sparse_dims: a list of tuple, whether each tuple represent the shape of the embedding of the sparse feature.
            The first element of the tuple represents the number of class or id of the feature.
            The second element of the tuple represents the embedding dimension of the feature.
        '''
        idx_start = 1 - int(USE_CHID)
        sparse_index = [category_cols.index(feat) for feat in sparse_feat][idx_start:]

        # dense dims: the number of dense feature
        feat_dims = np.array([len(chid_mapper)] + [len(v) for v in feat_mapper.values()]) + 1  # 0 is padding index, so add 1 dims
        # feat_dims: a list where each element represent the class or id count of a feature
        embed_dims = [chid_embed_dim] + [feat_embed_dim] * len(feat_mapper)  # dims of chid and other sparse feature
        # the embedding dimension of each dense feature.
        sparse_dims = [(fd, ed) for fd, ed in zip(feat_dims[idx_start:], embed_dims[idx_start:])]
        # combine feat_dims and sparse_dims for later embedding layer construction
        return sparse_dims, sparse_index

    def _get_dense_dims(self, numeric_cols, dense_feat):
        dense_index = [numeric_cols.index(feat) for feat in dense_feat]
        dense_dims = len(dense_feat)
        return dense_dims, dense_index


class BuildTensorDataset(ETLPro):
    def process(self, inputs):
        x_dense, x_sparse, y_objmean, y_tscnt, y_label_0 = inputs
        dataset = TensorDataset(
            torch.FloatTensor(x_dense),
            torch.LongTensor(x_sparse),
            torch.FloatTensor(y_objmean),
            torch.FloatTensor(y_tscnt),
            torch.FloatTensor(y_label_0)  # .flatten()
        )
        return [dataset]


sample_path = './data/sample_50k'

tmp_path = './ETL/tmp'
result_path = './data/sample_50k/RNN'

category_cols = ['chid', 'bnsfg', 'iterm', 'mcc', 'scity']
numeric_cols = ['bnspt', 'timestamp_0', 'timestamp_1', 'objam']

USE_CHID = True

sparse_feat = category_cols[:5]  # +['stonc_tag', 'stonc_6_label']
dense_feat = numeric_cols

sparse_dense_setting = GenerateSparseDenseSetting('generate sparse_dims, sparse_index, dense_dims, dense_index',
                                                  [ETLProcesses.feature_map],
                                                  category_cols=category_cols,
                                                  numeric_cols=numeric_cols,
                                                  sparse_feat=sparse_feat,
                                                  dense_feat=dense_feat,
                                                  USE_CHID=USE_CHID
                                                  )

sparse_dims_dense_dims = SelectResult(
    'select sparse_dims, dense_dims from sparse_dense_setting',
    [sparse_dense_setting],
    selected_indices=[0, 2]
)

sparse_dims = SelectResult(
    'select sparse_dims',
    [sparse_dims_dense_dims],
    selected_indices=[0]
)

dense_dims = SelectResult(
    'select dense_dims',
    [sparse_dims_dense_dims],
    selected_indices=[1]
)


sparse_index_dense_index = SelectResult(
    'select sparse_index, dense_index from sparse_dense_setting',
    [sparse_dense_setting],
    selected_indices=[1, 3]
)


processed_x_data = ProcessX('processing x_train and x_test',
                            [ETLProcesses.x_train, ETLProcesses.x_test, sparse_index_dense_index],
                            result_dir=[
                                os.path.join(tmp_path, 'x_train_sparse.npy'),
                                os.path.join(tmp_path, 'x_train_dense.npy'),
                                os.path.join(tmp_path, 'x_test_sparse.npy'),
                                os.path.join(tmp_path, 'x_test_dense.npy')
                            ]
                            )

processed_y_data = ProcessY('processing Y_train and Y_test',
                            [ETLProcesses.y_train, ETLProcesses.y_test, ETLProcesses.columns],
                            result_dir=[
                                os.path.join(tmp_path, 'train_objmean.npy'),
                                os.path.join(tmp_path, 'train_tscnt.npy'),
                                os.path.join(tmp_path, 'train_label_0.npy'),
                                os.path.join(tmp_path, 'test_objmean.npy'),
                                os.path.join(tmp_path, 'test_tscnt.npy'),
                                os.path.join(tmp_path, 'test_label_0.npy'),
                            ]
                            )

processed_train_data = SelectResult(
    'select training data',
    [processed_x_data, processed_y_data],
    selected_indices=[1, 0, 4, 5, 6],
    result_dir=[
        os.path.join(result_path, 'x_train_dense.npy'),
        os.path.join(result_path, 'x_train_sparse.npy'),
        os.path.join(result_path, 'train_objmean.npy'),
        os.path.join(result_path, 'train_tscnt.npy'),
        os.path.join(result_path, 'train_label_0.npy')
    ]
)

processed_test_data = SelectResult(
    'select test data',
    [processed_x_data, processed_y_data],
    selected_indices=[3, 2, 7, 8, 9],
    result_dir=[
        os.path.join(result_path, 'x_test_dense.npy'),
        os.path.join(result_path, 'x_test_sparse.npy'),
        os.path.join(result_path, 'test_objmean.npy'),
        os.path.join(result_path, 'test_tscnt.npy'),
        os.path.join(result_path, 'test_label_0.npy')
    ]
)


train_dataset = BuildTensorDataset('build torch TensorDataset', [processed_train_data])

test_dataset = BuildTensorDataset('build torch TensorDataset', [processed_test_data])

if __name__ == "__main__":
    batch_size = 64
    train_dataset = train_dataset.run()[0]
    test_dataset = test_dataset.run()[0]
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, num_workers=4)

    print('DataLoader Built', train_loader)

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskModel(dense_dims.run()[0], sparse_dims.run()[0], hidden_dims=64, out_dims=[1, 1, 1], n_layers=2, use_chid=USE_CHID, cell='GRU', bi=False, dropout=0.1)

    print('Model Built')
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    # trainer = pl.Trainer(auto_lr_find=True)
    # trainer = Trainer(model, optimizer, device)
    logger = TensorBoardLogger('tb_logs', 'run_pytorch_lightening')
    # model.inputs
    single_batch = next(iter(test_loader))
    logger.experiment.add_graph(model, [single_batch[0], single_batch[1]])
    trainer = pl.Trainer(logger=logger)
    lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_loader)
    # print(lr_finder.results)
    new_lr = lr_finder.suggestion()
    print('suggested lr:', new_lr)
    model.hparams.lr = new_lr

    print('Trainer Built')

    # t0 = time()
    # history = trainer.fit(train_loader, test_loader, epoch=1, early_stop=20)
    # t1 = time()
    trainer.fit(model, train_loader, test_loader)

    # print('cost: {:.2f}'.format(t1 - t0))

# processed_x_data.run()
'''
print('Apply Split and Dense Feature Transformation again on Test Data')
x_test_dense = x_test[:, -w_size:, len(category_cols):].astype(np.float64)
x_test_sparse = x_test[:, -w_size:, sparse_index].astype(np.int64)

x_test_dense = np.log1p(x_test_dense - x_test_dense.min(axis=0))
x_test_dense = x_scaler.transform(x_test_dense.reshape(-1, x_test_dense.shape[-1])).reshape(x_test_dense.shape)
'''

# print(x_train_dense.shape, x_train_sparse.shape)
# print(train_objmean.shape, train_tscnt.shape, train_label_0.shape)  # train_spcnt.shape,


# print(x_test_dense.shape, x_test_sparse.shape)
# print(test_objmean.shape, test_tscnt.shape, test_label_0.shape)  # test_spcnt.shape
