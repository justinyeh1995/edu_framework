#-*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import gc
import random 
random.seed(0)
from ETLBase import ETLPro, SelectResult

origin_path = '../data'
sample_path = 'data/sample'
tmp_path = 'data/tmp'
result_path = 'data/result'

chid_file = os.path.join(origin_path, 'sample_chid.txt')
cdtx_file = os.path.join(origin_path, 'sample_zip_if_cca_cdtx0001_hist.csv')
cust_f_file = os.path.join(origin_path, 'sample_zip_if_cca_cust_f.csv')


class Load_chids(ETLPro):
    def process(self, inputs):
        chid_array = np.loadtxt(chid_file, dtype=np.str)
        return [chid_array]
    
class Sample_chids(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, n_sample = None):
        super(Sample_chids, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.n_sample = n_sample
    def process(self, inputs):
        chids = inputs[0]
        if type(self.n_sample) == int:
            return [random.sample(chids.tolist(), self.n_sample)]
        else:
            return [chids.tolist()]

class Build_chid_to_nid_map(ETLPro):
    def process(self, inputs):
        chids_list = inputs[0]
        chid_to_nid_map = dict([(v,k) for k, v in enumerate(chids_list)])
        return [chid_to_nid_map]

class Load_cdtx(ETLPro):
    def process(self, inputs):
        chids = inputs[0]
        t0 = time()
        df_cdtx = pd.read_csv(cdtx_file, skipinitialspace=True)
        df_cdtx = df_cdtx[df_cdtx.chid.isin(chids)]
        # df_cdtx.csmdt.apply(lambda x: x[:-len('T00:00:00.000Z')])
        df_cdtx.sort_values(by=['csmdt', 'chid'], inplace=True, ignore_index=True)
        # df_cdtx.objam = df_cdtx.objam.astype(np.int64)
        print('loading time:', time() - t0)
        # checking
        assert len(set(df_cdtx.chid) - set(chids)) == 0 \
            and len(set(chids) - set(df_cdtx.chid)) == 0
        assert type(df_cdtx.objam[0]) == np.int64
        assert len(df_cdtx.csmdt[0]) == 10
        return [df_cdtx]


class Load_cust_f(ETLPro):
    def process(self, inputs):
        chids = inputs[0]
        t0 = time()
        df_cust_f = pd.read_csv(cust_f_file, skipinitialspace=True)
        df_cust_f = df_cust_f[df_cust_f.chid.isin(chids)]
        df_cust_f.sort_values(
            by=['data_dt', 'chid'],
            inplace=True,
            ignore_index=True
        )
        df_cust_f.drop_duplicates(ignore_index=True, inplace=True)
        print('loading time:', time() - t0)
        assert len(set(df_cust_f.chid) - set(chids)) == 0 \
            and len(set(chids) - set(df_cust_f.chid)) == 0
        return [df_cust_f]


class ConvertUidToNid(ETLPro):
    def process(self, inputs):
        df_input, chid_to_nid_map = inputs
        df_input.chid = df_input.chid.map(chid_to_nid_map) + 1
        return [df_input]


class ConvertTimeCol_into_np_datetime64(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, time_column='data_dt'):
        super(ConvertTimeCol_into_np_datetime64, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.time_column = time_column

    def process(self, inputs):
        df_full_y_sum = inputs[0]
        df_full_y_sum[self.time_column] = df_full_y_sum[self.time_column].astype(np.datetime64)
        return [df_full_y_sum]


class AddMonthTo_cdtx(ETLPro):
    def process(self, inputs):
        df_cdtx = inputs[0]
        df_cdtx['month'] = df_cdtx.csmdt.apply(lambda x: x[:-3] + '-01')
        return [df_cdtx]


class OuterProductTableOfChidsMonth(ETLPro):
    def process(self, inputs):
        '''
        產生一個包含所有顧客與所有月份的一個table。column的數量為: (# chids) X (# months)
        '''
        df_cdtx = inputs[0]
        list_chid = sorted(df_cdtx.chid.unique())
        list_month = sorted(df_cdtx.month.unique())[:]
        print('list_chid, list_month calculated')
        del df_cdtx
        print('[DELETE] df_cdtx')
        gc.collect()

        df_full_y_sum = pd.DataFrame({
            'chid': list_chid * len(list_month),
        }).sort_values(by='chid', ignore_index=True)  # 讓list_chid重複的次數和月的數量一樣多
        df_full_y_sum['data_dt'] = list_month * len(list_chid)  # 讓list_month重複出現的次數和顧客數一樣多

        return [df_full_y_sum]


class CalculateMonthlyTargetValues(ETLPro):
    def process(self, inputs):
        shrinked_df_cdtx = inputs[0]

        for col in set(shrinked_df_cdtx.columns) - set(['chid', 'month', 'objam']):
            del shrinked_df_cdtx[col]
            gc.collect()

        cdtx_group = shrinked_df_cdtx.groupby(['chid', 'month'])
        print('df_cdtx grouped')
        del shrinked_df_cdtx
        gc.collect()

        cdtx_sum = cdtx_group.sum()  # 總金額
        cdtx_mean = cdtx_group.mean()  # 平均金額
        cdtx_count = cdtx_group.count()  # 消費次數
        del cdtx_group
        gc.collect()

        # TODO: implement the loading of stonc_6_label
        # cdtx_group = df_cdtx[['chid', 'month', 'stonc_6_label']].drop_duplicates().groupby(['chid', 'month'])
        # cdtx_shop_kind_count = cdtx_group.count()

        # del cdtx_group
        # gc.collect()

        # del df_cdtx
        # gc.collect()

        df_cdtx_objam = pd.DataFrame(list(map(list, cdtx_sum.index)), columns=['chid', 'data_dt'])
        df_cdtx_objam['objam_sum'] = cdtx_sum.values[:, 0]
        df_cdtx_objam['objam_mean'] = cdtx_mean.values[:, 0]
        df_cdtx_objam['trans_count'] = cdtx_count.values[:, 0]  # 交易次數

        # df_cdtx_objam['shop_count'] = cdtx_shop_kind_count.values[:, 0] # 一個月內消費店家種類個數

        del cdtx_sum, cdtx_mean, cdtx_count  # , cdtx_shop_kind_count
        gc.collect()
        return [df_cdtx_objam]


class MergeWithOtherTable(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, join_method='left'):
        super(MergeWithOtherTable, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.join_method = join_method

    def process(self, inputs):
        df_full_y_sum = inputs[0]
        df_to_be_merged = inputs[1]
        df_full_y_sum = df_full_y_sum.merge(df_to_be_merged,
                                            how=self.join_method,
                                            left_on=['chid', 'data_dt'],
                                            right_on=['chid', 'data_dt']).fillna(0)
        return [df_full_y_sum]


class AddMeanOfPrevTwoMonths(ETLPro):
    def process(self, inputs):
        # 本月 前1、2月 平均金額
        df_full_y_sum = inputs[0]
        df_full_y_sum.insert(6, 'objam_mean_M3', 0)
        for chid in tqdm(sorted(df_full_y_sum.chid.unique())):
            mask = df_full_y_sum.chid == chid

            temp = (df_full_y_sum.loc[mask, 'objam_sum'] +
                    df_full_y_sum.loc[mask, 'objam_sum'].shift(1).fillna(0) +
                    df_full_y_sum.loc[mask, 'objam_sum'].shift(2).fillna(0)) // 3

            df_full_y_sum.loc[mask, 'objam_mean_M3'] = temp
            gc.collect()
        return [df_full_y_sum]


class AddDurationSinceLastTrans(ETLPro):
    def process(self, inputs):
        df_cdtx = inputs[0]
        df_cdtx['timestamp_0'] = (df_cdtx.csmdt - df_cdtx.csmdt.shift()).values / np.timedelta64(1, 'D')
        df_cdtx['timestamp_0'] = df_cdtx['timestamp_0'].fillna(0)
        df_cdtx.sort_values(by=['chid', 'csmdt'], ignore_index=True, inplace=True)
        mask_list = []
        chid_pre = -1
        for i, chid in tqdm(enumerate(df_cdtx.chid.values)):
            if chid != chid_pre:  # 不是-1，也不是前一個chid，代表是沒有算到另一個chid的前一次時間。
                chid_pre = chid
                mask_list.append(i)

        df_cdtx.loc[mask_list, 'timestamp_0'] = 0
        return [df_cdtx]


class AddDurationSince20180101(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, time_column='csmdt', result_column='timestamp_1'):
        super(AddDurationSince20180101, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.time_column = time_column
        self.result_column = result_column

    def process(self, inputs):
        df_cdtx = inputs[0]
        df_cdtx[self.result_column] = (
            df_cdtx[self.time_column] - np.datetime64('2018-01-01')
        ).values / np.timedelta64(1, 'D')
        df_cdtx[self.result_column].fillna(0)
        return [df_cdtx]


class ExtractTargetCols(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, target_cols=[]):
        super(ExtractTargetCols, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.target_cols = target_cols

    def process(self, inputs):
        df_origin = inputs[0]
        df_y = df_origin[self.target_cols].copy().reset_index(drop=True)
        del df_origin
        gc.collect()
        return [df_y]


class ExtractCatNumColsAndEncodeWithCatID(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, category_cols=[], numeric_cols=[]):
        super(ExtractCatNumColsAndEncodeWithCatID, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.category_cols = category_cols
        self.numeric_cols = numeric_cols

    def process(self, inputs):
        df_original = inputs[0]
        df_result = df_original[self.category_cols + self.numeric_cols]
        del df_original
        gc.collect()
        print('table extracted')
        for col in self.category_cols[1:]:
            if type(df_result[col][0]) == str:
                df_result[col] = df_result[col].fillna('')
                print(col, 'na filled')
            elif type(df_result[col][0]) == np.int64:
                df_result[col] = df_result[col].fillna(-1)
                print(col, 'na filled')
            if type(df_result[col][0]) != str and type(df_result[col][0]) != np.int64:
                df_result[col] = df_result[col].values.astype(np.str)
                print(col, 'type casted')
            gc.collect()
        print('all non str category col casted')
        mapper = {col: {value: index + 1 for index, value in enumerate(sorted(df_result[col].unique()))}
                  for col in self.category_cols[1:]}
        print('mapper created')

        for col in self.category_cols[1:]:
            df_result[col] = df_result[col].map(mapper[col])
            gc.collect()
            print(col, 'map applied')
        return [df_result, mapper]


class DataSplit(ETLPro):
    def __init__(self, process_name, pre_request_etls, result_dir=None, window_size=120, test_size=2):
        super(DataSplit, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.window_size = 120
        self.test_size = 2

    def process(self, inputs):
        print(len(inputs))
        df_x, df_f, df_y = inputs
        x_train, x_test, f_train, f_test, y_train, y_test = [], [], [], [], [], []

        for i in tqdm(sorted(df_y.chid.unique())):
            # 抓出各個顧客的資料
            data_x = df_x[df_x.chid == i].reset_index(drop=True)
            data_f = df_f[df_f.chid == i].reset_index(drop=True)
            data_y = df_y[df_y.chid == i].reset_index(drop=True)

            # 抓出某一顧客的最後一月的月份
            last = data_y.shape[0] - 1
            # 把資料的月份按照順序列出
            ts_list = sorted(data_y.timestamp.unique())

            for j, (ts_f, ts_y) in enumerate(zip(ts_list[:-1], ts_list[1:])):
                # ts_f是前一月 ts_y是下一個月
                data_x_ws = data_x[data_x.timestamp_1 < ts_y][-self.window_size:].copy()
                # 把和 ts_y 的差距月數作為新的時間因子
                data_x_ws.timestamp_1 = ts_y - data_x_ws.timestamp_1
                # 轉換為 np array
                data_x_ws = data_x_ws.values
                # 如果取的資料量比window_size還小，補進0
                if data_x_ws.shape[0] < self.window_size:
                    tmp = np.zeros((self.window_size, data_x.shape[1]))
                    if data_x_ws.shape[0] > 0:
                        tmp[-data_x_ws.shape[0]:] = data_x_ws
                    data_x_ws = tmp

                if j < last - self.test_size:
                    x_train.append(data_x_ws)
                    # 取前一個月的顧客特徵
                    f_train.append(data_f[data_f.timestamp == ts_f].values[0, :-1])
                    # 往下一個月去取資料
                    y_train.append(data_y.values[j + 1, :-1])
                elif j < last:
                    x_test.append(data_x_ws)
                    # 取前一個月的顧客特徵
                    f_test.append(data_f[data_f.timestamp == ts_f].values[0, :-1])
                    # 往下一個月去取資料
                    y_test.append(data_y.values[j + 1, :-1])
                else:
                    break
        x_train, x_test = np.array(x_train), np.array(x_test)
        f_train, f_test = np.array(f_train), np.array(f_test)
        y_train, y_test = np.array(y_train), np.array(y_test)
        return [x_train, x_test, f_train, f_test, y_train, y_test]


class AddNewTarget_objam_mean_M3_diff(ETLPro):
    def process(self, inputs):
        df_y = inputs[0]
        y_train = inputs[1]
        y_test = inputs[2]
        y_columns = list(df_y)
        y_columns[-1] = 'objam_mean_M3_diff'

        y_train[:, -1] = y_train[:, 2] - y_train[:, -1]
        y_test[:, -1] = y_test[:, 2] - y_test[:, -1]
        return [y_train, y_test, y_columns]


class XFYColumnBuilder(ETLPro):
    def process(self, inputs):
        df_input, df_feat_input, y_columns = inputs
        columns = {
            'x_columns': list(df_input),
            'f_columns': list(df_feat_input),
            'y_columns': y_columns,
        }
        return [columns]

chids = Load_chids('load_chids',[])


sampled_chids = Sample_chids(
    'sample_chids', 
    [chids], 
    result_dir = os.path.join(sample_path,'sampled_chids.npy'), 
    n_sample = 500) 

chid_to_nid_map = Build_chid_to_nid_map('build_chid_to_nid_map',
                      [sampled_chids],
                      result_dir = os.path.join(sample_path,'chid_to_nid_map.npy')
                     )

df_cdtx = Load_cdtx('load_cdtx',[sampled_chids], os.path.join(sample_path,'sampled_cdtx0001_hist.feather'))

'''df_cdtx = Load_cdtx(
    'load_cdtx',
    [])'''
df_cdtx = ConvertUidToNid(
    'convert_uid_to_nid',
    [df_cdtx, chid_to_nid_map]
)
df_cdtx = AddMonthTo_cdtx(
    'add_month_to_cdtx',
    [df_cdtx],
    result_dir=os.path.join(tmp_path, 'df_cdtx.feather')
)
# add_month_to_cdtx.run()
df_full_y_sum = OuterProductTableOfChidsMonth(
    'outer_product_table_of_chids_and_months',
    [df_cdtx],
    result_dir=os.path.join(tmp_path, 'df_full_y_sum.feather')
)

df_cdtx_monthly_objam = CalculateMonthlyTargetValues(
    'calculate_monthly_target_values',
    [df_cdtx]
)

# calculate_monthly_target_values.run()
df_full_y_sum = MergeWithOtherTable(
    'merge_df_full_y_sum_with_df_cdtx',
    [df_full_y_sum, df_cdtx_monthly_objam],
    result_dir=os.path.join(tmp_path, 'df_full_y_sum_1.feather'),
    join_method='left'
)
# df_full_y_sum.run()

# df_cust_f = Load_cust_f('load_cust_f', [])

df_cust_f = Load_cust_f('load_cust_f',[sampled_chids], os.path.join(sample_path,'sampled_cust_f.feather'))

df_cust_f = ConvertUidToNid(
    'convert_uid_to_nid_on_cust_f',
    [df_cust_f, chid_to_nid_map],
    result_dir=os.path.join(tmp_path, 'df_cust_f.feather')
)

df_full_y_sum = MergeWithOtherTable(
    'merge_df_full_y_sum_with_df_cust_f',
    [df_full_y_sum, df_cust_f],
    result_dir=os.path.join(tmp_path, 'df_full_y_sum_2.h5'),
    join_method='inner'
)
# df_full_y_sum.run()

df_full_y_sum = AddMeanOfPrevTwoMonths('add_mean_of_previous_two_months', [df_full_y_sum])
# df_full_y_sum.run()

df_full_y_sum = ConvertTimeCol_into_np_datetime64(
    'convert_data_dt_into_np_datetime64_on_df_full_y_sum',
    [df_full_y_sum],
    result_dir=os.path.join(tmp_path, 'df_full_y_sum_3.h5'),
    time_column='data_dt'
)

# df_full_y_sum.run()
df_cdtx = ConvertTimeCol_into_np_datetime64(
    'convert_data_dt_into_np_datetime64_on_df_cdtx',
    [df_cdtx],
    time_column='csmdt'
)
# df_cdtx.run()

df_cdtx = AddDurationSince20180101(
    'add_duration_since_20180101',
    [df_cdtx],
    time_column='csmdt',
    result_column='timestamp_1'
)

df_cdtx = AddDurationSinceLastTrans(
    'add_duration_since_last_trans',
    [df_cdtx],
    result_dir=os.path.join(tmp_path, 'df_cdtx_2.feather')
)
# df_cdtx.run()

df_input_feature_map = ExtractCatNumColsAndEncodeWithCatID(
    'extract_cat_num_cols_and_encode_with_catid_on_df_input',
    [df_cdtx],
    category_cols=['chid', 'bnsfg', 'iterm', 'mcc', 'scity'],
    numeric_cols=['bnspt', 'timestamp_0', 'timestamp_1', 'objam'],
    result_dir=[
        os.path.join(tmp_path, 'df_input.feather'),
        os.path.join(tmp_path, 'feature_map.npy')
    ]
)


# df_input.run()

df_input = SelectResult(
    'select_df_input_from_df_input_feature_map',
    [df_input_feature_map],
    selected_indices=[0])


df_feat_input_cust_feature_map = ExtractCatNumColsAndEncodeWithCatID(
    'extract_cat_num_cols_and_encode_with_catid_on_df_cust_f',
    [df_cust_f],
    category_cols=['chid', 'masts', 'educd', 'trdtp', 'poscd'],
    numeric_cols=['slam', 'first_mob', 'constant_change', 'sum_l2_ind',
                  'sum_u2_ind', 'constant_l2_ind', 'constant_u4_ind',
                  'growth_rate', 'monotone_down', 'monotone_up', 'data_dt'],
    result_dir=[
        os.path.join(tmp_path, 'df_feat_input.feather'),
        os.path.join(tmp_path, 'cust_feature_map.npy')
    ]
)


df_feat_input = SelectResult(
    'select_df_feat_input_from_df_feat_input_cust_feature_map',
    [df_feat_input_cust_feature_map],
    selected_indices=[0])

df_feat_input = ConvertTimeCol_into_np_datetime64(
    'convert_data_dt_into_np_datetime64_on_df_feat_input',
    [df_feat_input],
    time_column='data_dt',
    result_dir=os.path.join(tmp_path, 'df_feat_input_2.feather')
)

df_feat_input = AddDurationSince20180101(
    'add_duration_since_20180101_on_df_feat_input',
    [df_feat_input],
    time_column='data_dt',
    result_column='timestamp'
)
# df_feat_input.run()

df_y = ExtractTargetCols(
    'extract_target_columns_from_df',
    [df_full_y_sum],
    target_cols=['chid', 'data_dt', 'objam_sum', 'objam_mean', 'trans_count', 'objam_mean_M3'],  # 'shop_count'
    result_dir=os.path.join(tmp_path, 'df_y.feather')
)
# df_y.run()
df_y = AddDurationSince20180101(
    'add_duration_since_20180101_on_df_y',
    [df_y],
    time_column='data_dt',
    result_column='timestamp'
)

x_train_x_test_f_train_f_test_y_train_y_test = DataSplit(
    'data_split',
    [df_input, df_feat_input, df_y],
    result_dir=[
        os.path.join(tmp_path, 'x_train.npy'),
        os.path.join(tmp_path, 'x_test.npy'),
        os.path.join(tmp_path, 'f_train.npy'),
        os.path.join(tmp_path, 'f_test.npy'),
        os.path.join(tmp_path, 'y_train_tmp.npy'),
        os.path.join(tmp_path, 'y_test_tmp.npy')
    ],
    window_size=120,
    test_size=2
)

y_train_y_test = SelectResult(
    'extract_y_train_y_test',
    [x_train_x_test_f_train_f_test_y_train_y_test],
    selected_indices=[4, 5])


y_train_y_test_y_columns = AddNewTarget_objam_mean_M3_diff(
    'add_objam_mean_M3_diff_as_new_target',
    [df_y, y_train_y_test],
    result_dir=[
        os.path.join(tmp_path, 'y_train.npy'),
        os.path.join(tmp_path, 'y_test.npy'),
        os.path.join(tmp_path, 'y_columns.npy')
    ]
)


y_columns = SelectResult(
    'extract_y_columns_from_y_train_test_columns',
    [y_train_y_test_y_columns],
    selected_indices=[2])


x_train = SelectResult(
    'select_x_train',
    [x_train_x_test_f_train_f_test_y_train_y_test],
    selected_indices=[0],
    result_dir=os.path.join(result_path, 'x_train.npy')
)

x_test = SelectResult(
    'select_x_test',
    [x_train_x_test_f_train_f_test_y_train_y_test],
    selected_indices=[1],
    result_dir=os.path.join(result_path, 'x_test.npy')
)

f_train = SelectResult(
    'select_f_train',
    [x_train_x_test_f_train_f_test_y_train_y_test],
    selected_indices=[2],
    result_dir=os.path.join(result_path, 'f_train.npy')
)

f_test = SelectResult(
    'select_f_test',
    [x_train_x_test_f_train_f_test_y_train_y_test],
    selected_indices=[3],
    result_dir=os.path.join(result_path, 'f_test.npy')
)

y_train = SelectResult(
    'select_y_train',
    [y_train_y_test_y_columns],
    selected_indices=[0],
    result_dir=os.path.join(result_path, 'y_train.npy')
)

y_test = SelectResult(
    'select_y_test',
    [y_train_y_test_y_columns],
    selected_indices=[1],
    result_dir=os.path.join(result_path, 'y_test.npy')
)

columns = XFYColumnBuilder('create_x_f_y_columns',
                           [df_input, df_feat_input, y_columns],
                           result_dir=os.path.join(result_path, 'columns.npy')
                           )

feature_map = SelectResult(
    'select_feature_map',
    [df_input_feature_map],
    selected_indices=[1],
    result_dir=os.path.join(result_path, 'feature_map.npy')
)

cust_feature_map = SelectResult(
    'select_cust_feature_map',
    [df_feat_input_cust_feature_map],
    selected_indices=[1],
    result_dir=os.path.join(result_path, 'cust_feature_map.npy')
)



if __name__ == "__main__":
    # create_data_folder()
    X_Train = x_train.run()
    F_Train = f_train.run()
    Y_Train = y_train.run()
    X_Test = x_test.run()
    F_Test = f_test.run()
    Y_Test = y_test.run()
    Cols = columns.run()
    # print(X_Train, Y_Train, F_Train)
    # print(X_Test, Y_Test, F_Test)
    # print(Cols)

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
