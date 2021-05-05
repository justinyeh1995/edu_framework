import os
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from utils import read_and_convert, plus_month

def main():
    
    sample_data_path = './sample'
    
    chid_dict_file_name = 'sample_50k_idx_map.npy'
    cdtx_file_name = 'sample_50k_cdtx.json'
    cust_file_name = 'sample_50k_cust_f.json'
    
    sample_chid_dict = os.path.join(sample_data_path, chid_dict_file_name)
    sample_cdtx_file = os.path.join(sample_data_path, cdtx_file_name)
    sample_cust_file = os.path.join(sample_data_path, cust_file_name)
    
    df_cdtx = read_and_convert(sample_cdtx_file, sortby = 'csmdt')
    df_cust = read_and_convert(sample_cust_file)
    df_cust.drop_duplicates(ignore_index=True, inplace=True)
    
    chid_dict = np.load(sample_chid_dict, allow_pickle=True).tolist()
    
    shop_col = 'mcc'
    n_users = df_cdtx.chid.nunique()
    n_shops = df_cdtx[shop_col].nunique()
    
    df_cdtx.chid = df_cdtx.chid.map(chid_dict)
    df_cdtx['month'] = df_cdtx.csmdt.apply(lambda x: x[:8]+'01')
    df_cdtx.objam = df_cdtx.objam.apply(lambda x: int(x))
    shop_f = LabelEncoder()
    df_cdtx[shop_col] = shop_f.fit_transform(df_cdtx[shop_col])
    
    df_cust.chid = df_cust.chid.map(chid_dict)
    df_cust.data_dt = df_cust.data_dt.apply(lambda x: x[:10])
    
    list_chid = sorted(df_cust.chid.unique())
    list_month = sorted(df_cdtx.month.unique())
    
    df_cust_full = pd.DataFrame({'chid': list_chid*len(list_month)})
    df_cust_full['data_dt'] = np.repeat(list_month, n_users)
    
    df_cust_full = df_cust_full.merge(df_cust,
                                      how='left',
                                      left_on=['chid', 'data_dt'],
                                      right_on=['chid', 'data_dt'])

    ignore_cols = ['chid', 'data_dt']
    category_cols = ['masts', 'educd', 'naty', 'trdtp', 'poscd', 'cuorg']
    numeric_cols = set(df_cust_full.columns) - set(category_cols) - set(ignore_cols)

    values = dict()
        
    for col in numeric_cols:
        values[col] = 0
    for col in category_cols:
        values[col] = '-1'
    df_cust_full.fillna(value=values, inplace=True)
    
    cat_encoder = LabelEncoder()
    cat_f = df_cust_full[category_cols].apply(cat_encoder.fit_transform).to_numpy()
    
    temp_cdtx = df_cdtx.groupby(['chid', 'month']).sum()
    df_cdtx_objam = pd.DataFrame(list(map(list, temp_cdtx.index)), columns=['chid', 'data_dt'])
    df_cdtx_objam['objam'] = np.log(temp_cdtx.objam.values+1)
    
    df_cust_full = df_cust_full.merge(df_cdtx_objam,
                                      how='left',
                                      left_on=['chid', 'data_dt'],
                                      right_on=['chid', 'data_dt']).fillna(0)

#     lag_month = 11
#     last_colname = 'objam'
#     for k in trange(lag_month):
#         df_cdtx_objam['data_dt'] = df_cdtx_objam['data_dt'].apply(lambda x: plus_month(x, 1))
#         df_cdtx_objam.rename(columns={last_colname: f'objam_lag{k+1}'}, inplace=True)
#         df_cust_full = df_cust_full.merge(df_cdtx_objam,
#                                           how='left',
#                                           left_on=['chid', 'data_dt'],
#                                           right_on=['chid', 'data_dt']).fillna(0)
    
#         last_colname = f'objam_lag{k+1}'
        
#     cust_shop_objam = []
#     for month in tqdm(list_month):
#         x = np.zeros([n_users, n_shops])
#         groupby = df_cdtx[df_cdtx.month==month].groupby(['chid', shop_col]).sum()
#         index = groupby.index
#         objams = groupby.values
    
#         for (row, col), objam in zip(index, objams):
#             if objam > 0:
#                 x[row, col] = np.log(objam)
#             else:
#                 x[row, col] = 0
    
#         cust_shop_objam.append(x)
    
#     cust_shop_objam = np.concatenate(cust_shop_objam, 0)
#     df_cust_full = pd.concat([df_cust_full, pd.DataFrame(cust_shop_objam, columns=[f'shop_{i+1}' for i in range(n_shops)])], axis=1)
    
    numeric_cols = list(set(df_cust_full.columns) - set(category_cols) - set(ignore_cols))
    num_scaler = MinMaxScaler()
    df_cust_full[numeric_cols] = num_scaler.fit_transform(df_cust_full[numeric_cols])
    
    df_final = pd.concat([df_cust_full[ignore_cols+numeric_cols], pd.DataFrame(cat_f, columns=[f'category_{i+1}' for i in range(cat_f.shape[1])])], axis=1)
   
    df_final.to_csv('./preprocessed/df_cust_log_without_shop.csv', index=False)

if __name__ == '__main__':
    main()