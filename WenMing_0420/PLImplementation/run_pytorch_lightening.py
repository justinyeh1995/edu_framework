#!/usr/bin/env python
# coding: utf-8
import os
import gc
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from Model import MultiTaskModel
from Trainer import Trainer


def process_x_data(x, dense_index, sparse_index, x_scaler, mode='train'):
    w_size = x.shape[1]
    # window size of the time series data, x_train

    print('Split Dense and Sparse Parts and Apply np.float64 and np.int64, respectively')
    x_dense = x[:, -w_size:, dense_index].astype(np.float64)  # split dense feature
    x_sparse = x_train[:, -w_size:, sparse_index].astype(np.int64)  # split sparse feature
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
    return x_dense, x_sparse


def process_y_data(Y, y_columns, objmean_scaler, mode='train'):
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


def get_dense_dims(numeric_cols, dense_feat):
    dense_index = [numeric_cols.index(feat) for feat in dense_feat]
    dense_dims = len(dense_feat)
    return dense_dims, dense_index


def get_sparse_dims(category_cols, sparse_feat, chid_mapper, feat_mapper, USE_CHID=True, chid_embed_dim=64, feat_embed_dim=16):
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


def remove_unused_sparse_feat_mapper(feat_mapper, sparse_feat):
    keys = list(feat_mapper.keys())
    for key in keys:
        if key not in sparse_feat:
            del feat_mapper[key]
    return feat_mapper


sample_path = '../data/sample_50k'


x_train = np.load(os.path.join(sample_path, 'RNN', 'x_train.npy'), allow_pickle=True)
x_test = np.load(os.path.join(sample_path, 'RNN', 'x_test.npy'), allow_pickle=True)

# f_train = np.load(os.path.join(sample_path, 'RNN', 'f_train.npy'), allow_pickle=True)
# f_test = np.load(os.path.join(sample_path, 'RNN', 'f_test.npy'), allow_pickle=True)

Y_train = np.load(os.path.join(sample_path, 'RNN', 'y_train.npy'), allow_pickle=True)
Y_test = np.load(os.path.join(sample_path, 'RNN', 'y_test.npy'), allow_pickle=True)

print('Data Loaded')

columns = np.load(os.path.join(sample_path, 'RNN', 'columns.npy'), allow_pickle=True).item()

print('Data Info Loaded')

chid_mapper = np.load(os.path.join(sample_path, 'sample_50k_chid_idx_map.npy'), allow_pickle=True).item()
feat_mapper = np.load(os.path.join(sample_path, 'RNN', 'feature_map.npy'), allow_pickle=True).item()
# cust_feature_map = np.load(os.path.join(sample_path, 'RNN', 'cust_feature_map.npy'), allow_pickle=True).item()
print('Data Mapper Loaded')

category_cols = columns['x_columns'][:-4]
numeric_cols = columns['x_columns'][-4:]

sparse_feat = category_cols[:5]  # +['stonc_tag', 'stonc_6_label']
dense_feat = numeric_cols

feat_mapper = remove_unused_sparse_feat_mapper(feat_mapper, sparse_feat)

sparse_dims, sparse_index = get_sparse_dims(category_cols, sparse_feat, chid_mapper, feat_mapper)
dense_dims, dense_index = get_dense_dims(numeric_cols, dense_feat)

print('Model Configuration Info Generated')

print('Processing X-Data')

x_scaler = MinMaxScaler(feature_range=(0, 1))

x_train_dense, x_train_sparse = process_x_data(x_train, dense_index, sparse_index, x_scaler, mode='train')
x_test_dense, x_test_sparse = process_x_data(x_test, dense_index, sparse_index, x_scaler, mode='test')

print('Processing Y-Data')

objmean_scaler = MinMaxScaler((0, 1))
train_objmean, train_tscnt, train_label_0 = process_y_data(Y_train, columns["y_columns"], objmean_scaler, mode='train')
test_objmean, test_tscnt, test_label_0 = process_y_data(Y_test, columns["y_columns"], objmean_scaler, mode='test')


'''

print('Apply Split and Dense Feature Transformation again on Test Data')
x_test_dense = x_test[:, -w_size:, len(category_cols):].astype(np.float64)
x_test_sparse = x_test[:, -w_size:, sparse_index].astype(np.int64)

x_test_dense = np.log1p(x_test_dense - x_test_dense.min(axis=0))
x_test_dense = x_scaler.transform(x_test_dense.reshape(-1, x_test_dense.shape[-1])).reshape(x_test_dense.shape)

'''
print('Finish Preprocessing...')
print(x_train_dense.shape, x_train_sparse.shape)
print(train_objmean.shape, train_tscnt.shape, train_label_0.shape)  # train_spcnt.shape,


print(x_test_dense.shape, x_test_sparse.shape)
print(test_objmean.shape, test_tscnt.shape, test_label_0.shape)  # test_spcnt.shape
