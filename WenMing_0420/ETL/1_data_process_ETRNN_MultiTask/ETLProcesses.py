#-*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm, trange
from sklearn.preprocessing import MinMaxScaler
import gc
import feather

from ETLBase import ETLPro

sample_path = '../../data'
specific_path = '../data/sample_50k'

chid_file = os.path.join(sample_path, 'sample_chid.txt')
chid_dict_file = os.path.join(sample_path, 'sample_idx_map.npy')
cdtx_file = os.path.join(sample_path, 'sample_zip_if_cca_cdtx0001_hist.csv')
cust_f_file = os.path.join(sample_path, 'sample_zip_if_cca_cust_f.csv')


def load_chid_dict():
    chid_array = np.loadtxt(chid_file, dtype=np.str)
    chid_dict = np.load(chid_dict_file, allow_pickle=True).item()
    for i in range(len(chid_array)):
        assert chid_dict[chid_array[i]] == i
    return chid_dict


class Load_cdtx(ETLPro):
    def process(self, inputs):
        t0 = time()
        df_cdtx = pd.read_csv(cdtx_file, skipinitialspace=True)
        # df_cdtx.csmdt.apply(lambda x: x[:-len('T00:00:00.000Z')])
        df_cdtx.sort_values(by=['csmdt', 'chid'], inplace=True, ignore_index=True)
        # df_cdtx.objam = df_cdtx.objam.astype(np.int64)
        print('loading time:', time() - t0)
        chid_dict = load_chid_dict()
        assert len(set(df_cdtx.chid) - set(chid_dict.keys())) == 0 \
            and len(set(chid_dict.keys()) - set(df_cdtx.chid)) == 0
        assert type(df_cdtx.objam[0]) == np.int64
        assert len(df_cdtx.csmdt[0]) == 10
        return [df_cdtx]


class Load_cust_f(ETLPro):
    def process(self, inputs):
        t0 = time()
        df_cust_f = pd.read_csv(cust_f_file, skipinitialspace=True)
        df_cust_f.sort_values(
            by=['data_dt', 'chid'],
            inplace=True,
            ignore_index=True
        )
        df_cust_f.drop_duplicates(ignore_index=True, inplace=True)
        print('loading time:', time() - t0)
        chid_dict = load_chid_dict()
        assert len(set(df_cust_f.chid) - set(chid_dict.keys())) == 0 \
            and len(set(chid_dict.keys()) - set(df_cust_f.chid)) == 0
        return [df_cust_f]


class ConvertUidToNid(ETLPro):
    def process(self, inputs):
        df_input = inputs[0]
        chid_dict = load_chid_dict()
        df_input.chid = df_input.chid.map(chid_dict) + 1
        return [df_input]


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

        for col in set(shrinked_df_cdtx.columns) - set(['chid', 'month']):
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
        super(ETLPro, self).__init__(process_name, pre_request_etls, result_dir=result_dir)
        self.join_method = join_method

    def process(self, inputs):
        df_full_y_sum = inputs[0]
        df_to_be_merged = inputs[1]
        df_full_y_sum = df_full_y_sum.merge(df_to_be_merged,
                                            how=self.join_method,
                                            left_on=['chid', 'data_dt'],
                                            right_on=['chid', 'data_dt']).fillna(0)
        return [df_full_y_sum]


load_cdtx = Load_cdtx(
    'load_cdtx',
    [])
convert_uid_to_nid = ConvertUidToNid(
    'convert_uid_to_nid',
    [load_cdtx]
)
add_month_to_cdtx = AddMonthTo_cdtx(
    'add_month_to_cdtx',
    [convert_uid_to_nid],
    result_dir='df_cdtx.feather'
)
# add_month_to_cdtx.run()
df_full_y_sum = OuterProductTableOfChidsMonth(
    'outer_product_table_of_chids_and_months',
    [add_month_to_cdtx],
    result_dir='df_full_y_sum.feather'
)

calculate_monthly_target_values = CalculateMonthlyTargetValues(
    'calculate_monthly_target_values',
    [add_month_to_cdtx]
)

# calculate_monthly_target_values.run()
df_full_y_sum = MergeWithOtherTable(
    'merge_df_full_y_sum_with_df_cdtx',
    [df_full_y_sum, calculate_monthly_target_values],
    result_dir='df_full_y_sum_1.feather',
    join_method='left'
)
# df_full_y_sum.run()

load_cust_f = Load_cust_f('load_cust_f', [])
df_cust_f = ConvertUidToNid(
    'convert_uid_to_nid_on_cust_f',
    [load_cust_f]
)

df_cust_f = ConvertUidToNid(
    'convert_uid_to_nid',
    [df_cust_f],
    result_dir='df_cust_f.feather'
)
df_full_y_sum = MergeWithOtherTable(
    'merge_df_full_y_sum_with_df_cust_f',
    [df_full_y_sum, df_cust_f],
    result_dir='df_full_y_sum_2.h5',
    join_method='inner'
)
df_full_y_sum.run()

# TODO:
# - change temp object name to obj rather than the function name
# - [V] make the ETLBase allow save when a result directory is assigned
