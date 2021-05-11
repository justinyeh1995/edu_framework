from ETLComponent import *


def data_split(df_x, df_f, df_y, window_size, test_size=2):
    df_x = df_x.copy()
    df_f = df_f.copy()
    df_y = df_y.copy()

    df_f['timestamp'] = (df_f.data_dt - np.datetime64('2018-01-01')).apply(lambda x: x.days).fillna(0)
    df_y['timestamp'] = (df_y.data_dt - np.datetime64('2018-01-01')).apply(lambda x: x.days).fillna(0)

    x_train, x_test, f_train, f_test, y_train, y_test = [], [], [], [], [], []

    for i in tqdm(sorted(df_y.chid.unique())):
        data_x = df_x[df_x.chid == i].reset_index(drop=True)
        data_f = df_f[df_f.chid == i].reset_index(drop=True)
        data_y = df_y[df_y.chid == i].reset_index(drop=True)

        last = data_y.shape[0] - 1
        ts_list = sorted(data_y.timestamp.unique())

        for j, (ts_f, ts_y) in enumerate(zip(ts_list[:-1], ts_list[1:])):
            data_x_ws = data_x[data_x.timestamp_1 < ts_y][-window_size:].copy()
            data_x_ws.timestamp_1 = ts_y - data_x_ws.timestamp_1
            data_x_ws = data_x_ws.values

            if data_x_ws.shape[0] < window_size:
                tmp = np.zeros((window_size, data_x.shape[1]))
                if data_x_ws.shape[0] > 0:
                    tmp[-data_x_ws.shape[0]:] = data_x_ws
                data_x_ws = tmp

            if j < last - test_size:
                x_train.append(data_x_ws)
                f_train.append(data_f[data_f.timestamp == ts_f].values[0, :-1])
                y_train.append(data_y.values[j + 1, :-1])
            elif j < last:
                x_test.append(data_x_ws)
                f_test.append(data_f[data_f.timestamp == ts_f].values[0, :-1])
                y_test.append(data_y.values[j + 1, :-1])
            else:
                break

    x_train, x_test = np.array(x_train), np.array(x_test)
    f_train, f_test = np.array(f_train), np.array(f_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    return x_train, x_test, f_train, f_test, y_train, y_test


print('[LOAD] df_cdtx')
df_cdtx = load_cdtx()
gc.collect()

print('[CONVERT] df_cdtx.chid ')
df_cdtx = convert_uid_to_nid(df_cdtx)
gc.collect()

print('[ADD] month to df_cdtx')
df_cdtx = add_month_to_cdtx(df_cdtx)
gc.collect()

print('[SAVE] df_cdtx')
feather.write_dataframe(df_cdtx, 'df_cdtx.feather')

print('[CREATE] df_full_y_sum')
df_full_y_sum = outer_product_table_of_chids_and_months(df_cdtx)

print('[DELETE] df_cdtx')
del df_cdtx
gc.collect()

print('[SAVE] df_full_y_sum')
feather.write_dataframe(df_full_y_sum, 'df_full_y_sum.feather')

del df_full_y_sum
gc.collect()

print('[LOAD] df_cdtx_with_chid_month_objam')
df_cdtx_with_chid_month_objam = feather.read_dataframe('df_cdtx.feather', columns=['chid', 'month', 'objam'])

print('[CREATE] monthly table: df_cdtx_objam')
df_cdtx_objam = calculate_monthly_target_values(df_cdtx_with_chid_month_objam)

print('[DELETE] df_cdtx_with_chid_month_objam')
del df_cdtx_with_chid_month_objam
gc.collect()

print('[LOAD] df_full_y_sum')
df_full_y_sum = feather.read_dataframe('df_full_y_sum.feather')

print('[MERGE] df_full_y_sum, df_cdtx_objam')
df_full_y_sum = merge_with_other_table(df_full_y_sum, df_cdtx_objam, join_method='left')
gc.collect()

print("[SAVE] df_full_y_sum_1 ")
feather.write_dataframe(df_full_y_sum, 'df_full_y_sum_1.feather')

print("[DELETE] df_cdtx_objam")
del df_cdtx_objam
gc.collect()

# print("[LOAD] df_full_y_sum_1")
# df_full_y_sum = feather.read_dataframe('df_full_y_sum_1.feather')

print('[LOAD] df_cust_f')
df_cust_f = load_cust_f()
gc.collect()

print('[CONVERT] df_cust_f chid')
df_cust_f = convert_uid_to_nid(df_cust_f)
gc.collect()

print("[SAVE] df_cust_f")
feather.write_dataframe(df_cust_f, 'df_cust_f.feather')

print('[MERGE] df_full_y_sum, df_cust_f')
df_full_y_sum = merge_with_other_table(df_full_y_sum, df_cust_f, join_method='inner')
gc.collect()

print('[DELETE] df_cust_f')
del df_cust_f
gc.collect()

print("[SAVE] df_full_y_sum_2")
df_full_y_sum.to_hdf('df_full_y_sum_2.h5', key='df_full_y_sum_2', mode='w')

print("[ADD] mean of previous two months added to df_full_y_sum")
add_mean_of_previous_two_months(df_full_y_sum)
gc.collect()

print("[CONVERT] df_full_y_sum.data_dt to np.datetime64")
df_full_y_sum.data_dt = df_full_y_sum.data_dt.astype(np.datetime64)

print("[SAVE] df_full_y_sum_3")
df_full_y_sum.to_hdf('df_full_y_sum_3.h5', key='df_full_y_sum_3', mode='w')

print("[DELETE] df_full_y_sum")
del df_full_y_sum
gc.collect()

print('[LOAD] df_cdtx')
df_cdtx = feather.read_dataframe('df_cdtx.feather')
gc.collect()

print("[CONVERT] df_cdtx.csmdt to np.datetime64")
df_cdtx.csmdt = df_cdtx.csmdt.astype(np.datetime64)
gc.collect()

print('[ADD] duration since 2018.01.01 to df_cdtx')
add_duration_since_20180101(df_cdtx)
gc.collect()
assert 'timestamp_1' in df_cdtx.columns


print('[ADD] duration since last transaction to df_cdtx')
add_duration_since_last_trans(df_cdtx)
assert 'timestamp_0' in df_cdtx.columns
gc.collect()

print('[SAVE] df_cdtx_2')
feather.write_dataframe(df_cdtx, 'df_cdtx_2.feather')
gc.collect()

# extract and organize columns from df_cdtx and save them into df_input

print('[CREATE] df_input')
df_input, mapper = extract_cat_num_cols_and_encode_with_catid(
    df_cdtx,
    ['chid', 'bnsfg', 'iterm', 'mcc', 'scity'],  # , 'stonc_tag', 'stonc_label', 'stonm_label', 'stonc_6_label', 'stonc_10_label'
    ['bnspt', 'timestamp_0', 'timestamp_1', 'objam']
)

# TODO: check the meaning of iterm (why it is integer?)

print('[DELETE] df_cdtx')
del df_cdtx
gc.collect()

print('[SAVE] df_input')
feather.write_dataframe(df_input, 'df_input.feather')

# extract and organize columns from df_cust_f and save them into df_feat_input
print('[LOAD] df_cust_f ')
df_cust_f = feather.read_dataframe('df_cust_f.feather')

print('[CREATE] df_feat_input')
df_feat_input, feat_mapper = extract_cat_num_cols_and_encode_with_catid(
    df_cust_f,
    ['chid', 'masts', 'educd', 'trdtp', 'poscd'],
    ['slam', 'first_mob', 'constant_change', 'sum_l2_ind', 'sum_u2_ind', 'constant_l2_ind', 'constant_u4_ind',
     'growth_rate', 'monotone_down', 'monotone_up']
)

print('[DELTE] df_cust_f')
del df_cust_f
gc.collect()

print('[SAVE] df_feat_input ')
feather.write_dataframe(df_feat_input, 'df_feat_input.feather')

# extract target columns from df_full_y_sum and save them into df_y
print('[LOAD] df_full_y_sum_3')
df_full_y_sum = pd.read_hdf('df_full_y_sum_3.h5', key='df_full_y_sum_3', mode='r')

print('[CTEATE] df_y')
df_y = extract_target_columns_from_df(df_full_y_sum, ['chid', 'data_dt', 'objam_sum', 'objam_mean', 'trans_count', 'objam_mean_M3'])  # , 'shop_count'

print('[DELETE] df_full_y_sum')
del df_full_y_sum
gc.collect()

print('[SAVE] df_y')
feather.write_dataframe(df_y, 'df_y.feather')


# Data Split:

x_train, x_test, f_train, f_test, y_train, y_test = data_split(df_input, df_feat_input, df_y,
                                                               window_size=120, test_size=2)


# add new target:
y_columns = list(df_y)
y_columns[-1] = 'objam_mean_M3_diff'

y_train[:, -1] = y_train[:, 2] - y_train[:, -1]
y_test[:, -1] = y_test[:, 2] - y_test[:, -1]

print(y_columns)


np.save(os.path.join(specific_path, 'RNN', 'x_train'), x_train)
np.save(os.path.join(specific_path, 'RNN', 'x_test'), x_test)
np.save(os.path.join(specific_path, 'RNN', 'f_train'), f_train)
np.save(os.path.join(specific_path, 'RNN', 'f_test'), f_test)
np.save(os.path.join(specific_path, 'RNN', 'y_train'), y_train)
np.save(os.path.join(specific_path, 'RNN', 'y_test'), y_test)


np.save(os.path.join(specific_path, 'RNN', 'feature_map'), mapper)
np.save(os.path.join(specific_path, 'RNN', 'cust_feature_map'), feat_mapper)

columns = {
    'x_columns': list(df_input),
    'f_columns': list(df_feat_input),
    'y_columns': y_columns,
}
np.save(os.path.join(specific_path, 'RNN', 'columns'), columns)
print(columns)
