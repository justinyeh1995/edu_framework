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

sample_path = '../../data'
specific_path = '../data/sample_50k'

chid_file = os.path.join(sample_path, 'sample_chid.txt')
cdtx_file = os.path.join(sample_path, 'sample_zip_if_cca_cdtx0001_hist.csv')
cust_f_file = os.path.join(sample_path, 'sample_zip_if_cca_cust_f.csv')
chid_dict_file = os.path.join(sample_path, 'sample_idx_map.npy')


def load_chid_dict():
    chid_array = np.loadtxt(chid_file, dtype=np.str)
    chid_dict = np.load(chid_dict_file, allow_pickle=True).item()
    for i in range(len(chid_array)):
        assert chid_dict[chid_array[i]] == i
    return chid_dict


'''
chid_dict = load_chid_to_nid_dict()
print(len(chid_dict))
'''


def load_cust_f():
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
    # df_cust_f.head()
    return df_cust_f


'''
df_cust_f = load_cust_f()
print(df_cust_f.shape, df_cust_f.chid.nunique())
'''


def load_cdtx():
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
    return df_cdtx


'''
df_cdtx = load_cdtx()
print(df_cdtx.shape, df_cdtx.chid.nunique())
'''


def add_month_to_cdtx(df_cdtx):
    df_cdtx['month'] = df_cdtx.csmdt.apply(lambda x: x[:-3] + '-01')
    return df_cdtx


def convert_uid_to_nid(df_input):
    chid_dict = load_chid_dict()
    df_input.chid = df_input.chid.map(chid_dict) + 1
    return df_input


'''
df_cdtx = load_cdtx()
print('data_loaded')
gc.collect()

df_cdtx = convert_uid_to_nid(df_cdtx)
print('id converted')
gc.collect()

df_cdtx = add_month_to_cdtx(df_cdtx)
print('month added')
gc.collect()

print(df_cdtx.columns)
'''

def calculate_monthly_target_values(shrinked_df_cdtx):

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
    return df_cdtx_objam
    # 每個顧客，每個月會有一個數值


def shrink_table(df_cdtx, selected_columns):
    '''
    Shrink table for memory efficiency
    '''
    columns_to_remove = list(set(df_cdtx.columns) - set(selected_columns))
    for col in columns_to_remove:
        del df_cdtx[col]
    gc.collect()


'''
df_cdtx = load_cdtx()
print('cdtx loaded')
gc.collect()

df_cdtx = convert_uid_to_nid(df_cdtx)
print('id converted')
gc.collect()

df_cdtx = add_month_to_cdtx(df_cdtx)
print('month added')
gc.collect()


shrink_table(df_cdtx, ['chid', 'month', 'objam'])  # df_cdtx[['chid', 'month', 'objam']]
print('data shrinked')

df_cdtx_objam = calculate_monthly_target_values(df_cdtx)
print('group info calculated')
print(df_cdtx_objam.shape)
'''


def outer_product_table_of_chids_and_months(df_cdtx):
    '''
    產生一個包含所有顧客與所有月份的一個table。column的數量為: (# chids) X (# months)
    '''
    list_chid = sorted(df_cdtx.chid.unique())
    list_month = sorted(df_cdtx.month.unique())[:]
    print('list_chid, list_month calculated')
    del df_cdtx
    gc.collect()

    df_full_y_sum = pd.DataFrame({
        'chid': list_chid * len(list_month),
    }).sort_values(by='chid', ignore_index=True)  # 讓list_chid重複的次數和月的數量一樣多
    df_full_y_sum['data_dt'] = list_month * len(list_chid)  # 讓list_month重複出現的次數和顧客數一樣多

    return df_full_y_sum


'''df_cdtx = load_cdtx()
print('cdtx loaded')
gc.collect()

df_cdtx = convert_uid_to_nid(df_cdtx)
print('id converted')
gc.collect()

df_cdtx = add_month_to_cdtx(df_cdtx)
print('month added')
gc.collect()

df_full_y_sum = outer_product_table_of_chids_and_months(df_cdtx)
print(df_full_y_sum.shape)'''


def merge_with_other_table(df_full_y_sum, df_to_be_merged, join_method='left'):
    df_full_y_sum = df_full_y_sum.merge(df_to_be_merged,
                                        how=join_method,
                                        left_on=['chid', 'data_dt'],
                                        right_on=['chid', 'data_dt']).fillna(0)
    return df_full_y_sum


'''
df_cdtx = load_cdtx()
print('cdtx loaded')
gc.collect()

df_cdtx = convert_uid_to_nid(df_cdtx)
print('id converted')
gc.collect()

df_cdtx = add_month_to_cdtx(df_cdtx)
print('month added')
gc.collect()

feather.write_dataframe(df_cdtx, 'df_cdtx.feather')

df_full_y_sum = outer_product_table_of_chids_and_months(df_cdtx)
print('df_full_y_sum created')
del df_cdtx
gc.collect()

print('df_cdtx deleted again')

feather.write_dataframe(df_full_y_sum, 'df_full_y_sum.feather')

del df_full_y_sum
gc.collect()
print('df_full_y_sum saved')


df_cdtx_with_chid_month_objam = feather.read_dataframe('df_cdtx.feather', columns = ['chid', 'month', 'objam'])

print('processed df_cdtx load')

df_cdtx_objam = calculate_monthly_target_values(df_cdtx_with_chid_month_objam)

del df_cdtx_with_chid_month_objam
gc.collect()

print('group info calculated')
print(df_cdtx_objam.shape)

df_full_y_sum = feather.read_dataframe('df_full_y_sum.feather')
print('df_full_y_sum load')

merge_with_other_table(df_full_y_sum, df_cdtx_objam,join_method='left')
print('merged with df_cdtx_objam')
print(df_full_y_sum.shape)

del df_cdtx_objam
gc.collect()

df_cust_f = load_cust_f()
gc.collect()
print('df_cust_f loaded')

df_cust_f = convert_uid_to_nid(df_cust_f)
gc.collect()
print('df_cust_f chid converted')

merge_with_other_table(df_full_y_sum, df_cust_f, join_method='inner')
print('merged with df_cust_f')

del df_cust_f
gc.collect()
'''


def add_mean_of_previous_two_months(df_full_y_sum):
    # 本月 前1、2月 平均金額
    df_full_y_sum.insert(6, 'objam_mean_M3', 0)
    for chid in tqdm(sorted(df_full_y_sum.chid.unique())):
        mask = df_full_y_sum.chid == chid

        temp = (df_full_y_sum.loc[mask, 'objam_sum'] +
                df_full_y_sum.loc[mask, 'objam_sum'].shift(1).fillna(0) +
                df_full_y_sum.loc[mask, 'objam_sum'].shift(2).fillna(0)) // 3

        df_full_y_sum.loc[mask, 'objam_mean_M3'] = temp


def add_duration_since_20180101(df_cdtx, time_col='csmdt', result_col='timestamp_1'):
    df_cdtx[result_col] = (df_cdtx[time_col] - np.datetime64('2018-01-01')).values / np.timedelta64(1, 'D')
    df_cdtx[result_col].fillna(0)


def add_duration_since_last_trans(df_cdtx):
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


'''
df_cdtx = load_cdtx()
print('cdtx loaded')
gc.collect()

df_cdtx = convert_uid_to_nid(df_cdtx)
print('id converted')
gc.collect()

df_cdtx = add_month_to_cdtx(df_cdtx)
print('month added')
gc.collect()

feather.write_dataframe(df_cdtx, 'df_cdtx.feather')

df_full_y_sum = outer_product_table_of_chids_and_months(df_cdtx)
print('df_full_y_sum created')
del df_cdtx
gc.collect()

print('df_cdtx deleted again')

feather.write_dataframe(df_full_y_sum, 'df_full_y_sum.feather')

del df_full_y_sum
gc.collect()
print('df_full_y_sum saved')

df_cdtx_with_chid_month_objam = feather.read_dataframe('df_cdtx.feather', columns = ['chid', 'month', 'objam'])

print('processed df_cdtx load')

df_cdtx_objam = calculate_monthly_target_values(df_cdtx_with_chid_month_objam)

del df_cdtx_with_chid_month_objam
gc.collect()

print('group info calculated')
print(df_cdtx_objam.shape)

df_full_y_sum = feather.read_dataframe('df_full_y_sum.feather')
print('df_full_y_sum load')

df_full_y_sum = merge_with_other_table(df_full_y_sum, df_cdtx_objam,join_method='left')
gc.collect()
print('merged with df_cdtx_objam')
print(df_full_y_sum.shape)

feather.write_dataframe(df_full_y_sum, 'df_full_y_sum_1.feather')
print("df_full_y_sum_1 saved")

del df_cdtx_objam
gc.collect()


df_full_y_sum = feather.read_dataframe('df_full_y_sum_1.feather')
print("df_full_y_sum_1 load")

df_cust_f = load_cust_f()
gc.collect()
print('df_cust_f loaded')

df_cust_f = convert_uid_to_nid(df_cust_f)
gc.collect()
print('df_cust_f chid converted')

feather.write_dataframe(df_cust_f, 'df_cust_f.feather')
print("df_cust_f saved")

df_full_y_sum = merge_with_other_table(df_full_y_sum, df_cust_f, join_method='inner')
gc.collect()
print('merged with df_cust_f')

del df_cust_f
gc.collect()


df_full_y_sum.to_hdf('df_full_y_sum_2.h5', key='df_full_y_sum_2', mode='w')
print("df_full_y_sum_2 saved")

add_mean_of_previous_two_months(df_full_y_sum)
gc.collect()
print("mean of previous two months added to df_full_y_sum")

# converting data_dt as np.datetime64
df_full_y_sum.data_dt = df_full_y_sum.data_dt.astype(np.datetime64)
print("data_dt converted")


df_full_y_sum.to_hdf('df_full_y_sum_3.h5', key='df_full_y_sum_3', mode='w')
print("df_full_y_sum_3 saved")

del df_full_y_sum
gc.collect()
'''

# Processing df_cdtx - convert types and add new columns
'''
df_cdtx = feather.read_dataframe('df_cdtx.feather')
gc.collect()
print('df_cdtx loaded')
df_cdtx.csmdt = df_cdtx.csmdt.astype(np.datetime64)
gc.collect()
print('df_cdtx time converted')
add_duration_since_20180101(df_cdtx)
gc.collect()
assert 'timestamp_1' in df_cdtx.columns
print('add duration since 2018.01.01')

add_duration_since_last_trans(df_cdtx)
assert 'timestamp_0' in df_cdtx.columns
gc.collect()
print('add duration since last transaction')

feather.write_dataframe(df_cdtx, 'df_cdtx_2.feather')
del df_cdtx
gc.collect()
print('df_cdtx_2 saved')
'''


def extract_cat_num_cols_and_encode_with_catid(df_original, category_cols, numeric_cols):
    df_result = df_original[category_cols + numeric_cols]
    del df_original
    gc.collect()
    print('table extracted')
    for col in category_cols[1:]:
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
              for col in category_cols[1:]}
    print('mapper created')

    for col in category_cols[1:]:
        df_result[col] = df_result[col].map(mapper[col])
        gc.collect()
        print(col, 'map applied')
    return df_result, mapper


def extract_target_columns_from_df(df_origin, y_cols):
    df_y = df_origin[y_cols].copy().reset_index(drop=True)
    del df_origin
    gc.collect()
    return df_y


'''df_cdtx = load_cdtx()
print('[LOAD] df_cdtx')
gc.collect()

df_cdtx = convert_uid_to_nid(df_cdtx)
print('[CONVERT] df_cdtx.chid ')
gc.collect()

df_cdtx = add_month_to_cdtx(df_cdtx)
print('[ADD] month to df_cdtx')
gc.collect()

feather.write_dataframe(df_cdtx, 'df_cdtx.feather')

df_full_y_sum = outer_product_table_of_chids_and_months(df_cdtx)
print('[CREATE] df_full_y_sum')
del df_cdtx
gc.collect()

print('[DELETE] df_cdtx')

feather.write_dataframe(df_full_y_sum, 'df_full_y_sum.feather')

del df_full_y_sum
gc.collect()
print('[SAVE] df_full_y_sum')

df_cdtx_with_chid_month_objam = feather.read_dataframe('df_cdtx.feather', columns = ['chid', 'month', 'objam'])

print('[LOAD] df_cdtx_with_chid_month_objam')

df_cdtx_objam = calculate_monthly_target_values(df_cdtx_with_chid_month_objam)
print('[CREATE] monthly table: df_cdtx_objam')

del df_cdtx_with_chid_month_objam
gc.collect()
print('[DELETE] df_cdtx_with_chid_month_objam')

print(df_cdtx_objam.shape)

df_full_y_sum = feather.read_dataframe('df_full_y_sum.feather')
print('[LOAD] df_full_y_sum')

df_full_y_sum = merge_with_other_table(df_full_y_sum, df_cdtx_objam,join_method='left')
gc.collect()
print('[MERGE] df_full_y_sum, df_cdtx_objam')
print(df_full_y_sum.shape)

feather.write_dataframe(df_full_y_sum, 'df_full_y_sum_1.feather')
print("[SAVE] df_full_y_sum_1 ")

del df_cdtx_objam
gc.collect()
print("[DELETE] df_cdtx_objam")

df_full_y_sum = feather.read_dataframe('df_full_y_sum_1.feather')
print("[LOAD] df_full_y_sum_1")

df_cust_f = load_cust_f()
gc.collect()
print('[LOAD] df_cust_f')

df_cust_f = convert_uid_to_nid(df_cust_f)
gc.collect()
print('[CONVERT] df_cust_f chid')

feather.write_dataframe(df_cust_f, 'df_cust_f.feather')
print("[SAVE] df_cust_f")

df_full_y_sum = merge_with_other_table(df_full_y_sum, df_cust_f, join_method='inner')
gc.collect()
print('[MERGE] df_full_y_sum, df_cust_f')

del df_cust_f
gc.collect()
print('[DELETE] df_cust_f')

df_full_y_sum.to_hdf('df_full_y_sum_2.h5', key='df_full_y_sum_2', mode='w')
print("[SAVE] df_full_y_sum_2")

add_mean_of_previous_two_months(df_full_y_sum)
gc.collect()
print("[ADD] mean of previous two months added to df_full_y_sum")

# converting data_dt as np.datetime64
df_full_y_sum.data_dt = df_full_y_sum.data_dt.astype(np.datetime64)
print("[CONVERT] df_full_y_sum.data_dt to np.datetime64")


df_full_y_sum.to_hdf('df_full_y_sum_3.h5', key='df_full_y_sum_3', mode='w')
print("[SAVE] df_full_y_sum_3")

del df_full_y_sum
gc.collect()
print("[DELETE] df_full_y_sum")


df_cdtx = feather.read_dataframe('df_cdtx.feather')
gc.collect()
print('[LOAD] df_cdtx')
df_cdtx.csmdt = df_cdtx.csmdt.astype(np.datetime64)
gc.collect()
print("[CONVERT] df_cdtx.csmdt to np.datetime64")

add_duration_since_20180101(df_cdtx)
gc.collect()
assert 'timestamp_1' in df_cdtx.columns
print('[ADD] duration since 2018.01.01 to df_cdtx')

add_duration_since_last_trans(df_cdtx)
assert 'timestamp_0' in df_cdtx.columns
gc.collect()
print('[ADD] duration since last transaction to df_cdtx')

feather.write_dataframe(df_cdtx, 'df_cdtx_2.feather')
gc.collect()
print('[SAVE] df_cdtx_2')

# extract and organize columns from df_cdtx and save them into df_input


df_input, mapper = extract_cat_num_cols_and_encode_with_catid(
    df_cdtx,
    ['chid', 'bnsfg', 'iterm', 'mcc', 'scity'], # , 'stonc_tag', 'stonc_label', 'stonm_label', 'stonc_6_label', 'stonc_10_label'
    ['bnspt', 'timestamp_0', 'timestamp_1', 'objam']
)
print('[CREATE] df_input')

# TODO: check the meaning of iterm (why it is integer?)

del df_cdtx
gc.collect()
print('[DELETE] df_cdtx')

feather.write_dataframe(df_input, 'df_input.feather')
print('[SAVE] df_input')

# extract and organize columns from df_cust_f and save them into df_feat_input

df_cust_f = feather.read_dataframe('df_cust_f.feather')
print('[LOAD] df_cust_f ')

df_feat_input, feat_mapper = extract_cat_num_cols_and_encode_with_catid(
    df_cust_f,
    ['chid', 'masts', 'educd', 'trdtp', 'poscd'],
    ['slam', 'first_mob', 'constant_change', 'sum_l2_ind', 'sum_u2_ind', 'constant_l2_ind', 'constant_u4_ind',
                     'growth_rate', 'monotone_down', 'monotone_up']
)
print('[CREATE] df_feat_input')

del df_cust_f
gc.collect()

print('[DELTE] df_cust_f')

feather.write_dataframe(df_feat_input, 'df_feat_input.feather')
print('[SAVE] df_feat_input ')

# extract target columns from df_full_y_sum and save them into df_y

df_full_y_sum = pd.read_hdf('df_full_y_sum_3.h5', key='df_full_y_sum_3', mode='r')
print('[LOAD] df_full_y_sum_3')

df_y = extract_target_columns_from_df(df_full_y_sum, ['chid', 'data_dt', 'objam_sum', 'objam_mean', 'trans_count', 'objam_mean_M3']) # , 'shop_count'
print('[CTEATE] df_y')

del df_full_y_sum
gc.collect()
print('[DELETE] df_full_y_sum')

feather.write_dataframe(df_y, 'df_y.feather')
print('[SAVE] df_y')
'''
