#-*- coding: utf-8 -*-
import os
import gc

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch 
from torch.utils.data import TensorDataset

from common.ETLBase import ETLPro, SelectResult

class DatasetBuilder():
    def __init__(self, tmp_path, result_path, numeric_cols, category_cols, dense_feat, sparse_feat, USE_CHID):

        self.USE_CHID = USE_CHID

        self.tmp_path = tmp_path # 'data/tmp'
        self.result_path = result_path # 'data/result'

        self.numeric_cols = numeric_cols # ['bnspt', 'timestamp_0', 'timestamp_1', 'objam']
        self.category_cols = category_cols # ['chid', 'bnsfg', 'iterm', 'mcc', 'scity']
        
        self.dense_feat = dense_feat # self.numeric_cols
        self.sparse_feat = sparse_feat # self.category_cols[:5]  # +['stonc_tag', 'stonc_6_label']

    def connect_pipeline(self, x_train, x_test, y_train, y_test, feature_map, chid_to_nid_map, columns):
        sparse_dense_setting = GenerateSparseDenseSetting(
            'generate sparse_dims, sparse_index, dense_dims, dense_index',
            [feature_map, chid_to_nid_map],
            category_cols=self.category_cols,
            numeric_cols=self.numeric_cols,
            sparse_feat=self.sparse_feat,
            dense_feat=self.dense_feat,
            USE_CHID=self.USE_CHID
        )

        sparse_dims_dense_dims = SelectResult(
            'select sparse_dims, dense_dims from sparse_dense_setting',
            [sparse_dense_setting],
            selected_indices=[0, 2]
        )

        self.sparse_dims = SelectResult(
                'select sparse_dims',
                [sparse_dims_dense_dims],
                selected_indices=[0]
        )

        self.dense_dims = SelectResult(
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
                            [x_train, x_test, sparse_index_dense_index],
                            result_dir=[
                                os.path.join(self.tmp_path, 'x_train_sparse.npy'),
                                os.path.join(self.tmp_path, 'x_train_dense.npy'),
                                os.path.join(self.tmp_path, 'x_test_sparse.npy'),
                                os.path.join(self.tmp_path, 'x_test_dense.npy')
                            ]
                            )
        
        self.processed_y_data = ProcessY('processing Y_train and Y_test',
                            [y_train, y_test, columns],
                            result_dir=[
                                os.path.join(self.tmp_path, 'train_objmean.npy'),
                                os.path.join(self.tmp_path, 'train_tscnt.npy'),
                                os.path.join(self.tmp_path, 'train_label_0.npy'),
                                os.path.join(self.tmp_path, 'test_objmean.npy'),
                                os.path.join(self.tmp_path, 'test_tscnt.npy'),
                                os.path.join(self.tmp_path, 'test_label_0.npy'),
                            ]
                            )

        processed_train_data = SelectResult(
            'select training data',
            [processed_x_data, self.processed_y_data],
            selected_indices=[1, 0, 4, 5, 6],
            result_dir=[
                os.path.join(self.result_path, 'x_train_dense.npy'),
                os.path.join(self.result_path, 'x_train_sparse.npy'),
                os.path.join(self.result_path, 'train_objmean.npy'),
                os.path.join(self.result_path, 'train_tscnt.npy'),
                os.path.join(self.result_path, 'train_label_0.npy')
            ]
        )

        processed_test_data = SelectResult(
            'select test data',
            [processed_x_data, self.processed_y_data],
            selected_indices=[3, 2, 7, 8, 9],
            result_dir=[
                os.path.join(self.result_path, 'x_test_dense.npy'),
                os.path.join(self.result_path, 'x_test_sparse.npy'),
                os.path.join(self.result_path, 'test_objmean.npy'),
                os.path.join(self.result_path, 'test_tscnt.npy'),
                os.path.join(self.result_path, 'test_label_0.npy')
            ]
        )


        self.train_dataset = BuildTensorDataset('build torch TensorDataset', [processed_train_data])
        self.test_dataset = BuildTensorDataset('build torch TensorDataset', [processed_test_data])
        return self.train_dataset, self.test_dataset 


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
        index = y_columns.tolist().index('objam_sum')
        objsum = Y[:, [index]].astype(np.float64)

        index = y_columns.tolist().index('objam_mean')
        objmean = Y[:, [index]].astype(np.float64)

        index = y_columns.tolist().index('trans_count')
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
        feat_mapper, chid_mapper = inputs
        feat_mapper = self._remove_unused_sparse_feat_mapper(feat_mapper, self.sparse_feat)
        sparse_dims, sparse_index = self._get_sparse_dims(self.category_cols, self.sparse_feat, chid_mapper, feat_mapper, USE_CHID=self.USE_CHID)
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




'''
if __name__ == "__main__":
    Train = train_dataset.run()
    Test = test_dataset.run()
    
    Dense_dims = dense_dims.run()
    Sparse_dims = sparse_dims.run()
    
    print(Train, Test)
    print(Dense_dims, Sparse_dims)
'''
    # TODO:
    # - [V] change temp object name to obj rather than the function name
    # - [V] make the ETLBase allow save when a result directory is assigned
    # - [V] allow saving of multiple files (ETL might have multiple results)
    # - [ ] allow single input and single output (not list)
    # - [ ] allow subset selection from previous ETL process
    # - [ ] allow the input to be a dictionary and the output to be a dictionary, too
    # - [ ] make the ETL Process object allows two step construction like nn.module. 1. first initialized with configuration . 2. Be called to assign inputs and obtain outputs later
    # - [ ] incorporate google drive download as the first step of ETL
    # - [ ] allows zero input ETL if the ETL does not have previous ETL
    # - [ ] implement __item__ selected so that the ETL can be splitted by output !
