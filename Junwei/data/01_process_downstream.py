import os
import numpy as np
import pandas as pd

from time import time
from utils import read_and_convert, data_split

sample_data_path = './sample'

chid_file_name = 'sample_50k_chid.txt'
chid_dict_file_name = 'sample_idx_map.npy'
cdtx_file_name = 'sample_50k_cdtx.json'
cust_file_name = 'sample_50k_cust_f.json'

sample_chid_file = os.path.join(sample_data_path, chid_file_name)
sample_chid_dict = os.path.join(sample_data_path, chid_dict_file_name)
sample_cdtx_file = os.path.join(sample_data_path, cdtx_file_name)
sample_cust_file = os.path.join(sample_data_path, cust_file_name)


downstream_data_path = './downstream'

x_train_file_name = 'x_train.csv'
x_test_file_name = 'x_test.csv'
y_train_file_name = 'y_train.csv'
y_test_file_name = 'y_test.csv'

x_train_file = os.path.join(downstream_data_path, x_train_file_name)
x_test_file = os.path.join(downstream_data_path, x_test_file_name)
y_train_file = os.path.join(downstream_data_path, y_train_file_name)
y_test_file = os.path.join(downstream_data_path, y_test_file_name)

df_cdtx = read_and_convert(sample_cdtx_file, sortby = 'csmdt')
df_cust = read_and_convert(sample_cust_file)
df_cust.drop_duplicates(ignore_index=True, inplace=True)

chid_array = np.loadtxt(sample_chid_file, dtype=np.str)
chid_dict = np.load(sample_chid_dict, allow_pickle=True).tolist()

df_cdtx.chid = df_cdtx.chid.map(chid_dict)
df_cdtx['month'] = df_cdtx.csmdt.apply(lambda x: x[:8]+'01')
df_cdtx.objam = df_cdtx.objam.apply(lambda x: int(x))

df_cust.chid = df_cust.chid.map(chid_dict)
df_cust.data_dt = df_cust.data_dt.apply(lambda x: x[:10])

## 填滿後12個月

list_chid = sorted(df_cust.chid.unique())
list_month = sorted(df_cust.data_dt.unique())[11:]

df_full_y_sum = pd.DataFrame({
    'chid': list_chid*len(list_month),
}).sort_values(by='chid', ignore_index=True)
df_full_y_sum['data_dt'] = list_month*len(list_chid)

ignore_cols = ['chid', 'data_dt']
category_cols = ['masts', 'educd', 'naty', 'trdtp', 'poscd', 'cuorg']
numeric_cols = sorted(set(df_cust.columns) - set(category_cols) - set(ignore_cols))

df_full_y_sum = df_full_y_sum.merge(df_cust[ignore_cols + category_cols + numeric_cols], 
                                    how='left', 
                                    left_on=['chid', 'data_dt'], 
                                    right_on=['chid', 'data_dt'])

#df_full_y_sum.dropna(thresh=len(numeric_cols+category_cols), inplace=True)

## fill na value, numerical: 0, category: '-1'
values = dict()

for col in numeric_cols:
    values[col] = 0
    
for col in category_cols:
    values[col] = '-1'
    
df_full_y_sum.fillna(value=values, inplace=True)


## 取得整個月的 objam 
temp_cdtx = df_cdtx.groupby(['chid', 'month']).sum()
df_cdtx_objam = pd.DataFrame(list(map(list, temp_cdtx.index)), columns=['chid', 'data_dt'])
df_cdtx_objam['objam'] = np.ma.log(temp_cdtx.objam.values).filled(0)

## join objam

df_full_y_sum = df_full_y_sum.merge(df_cdtx_objam, 
                                    how='left', 
                                    left_on=['chid', 'data_dt'], 
                                    right_on=['chid', 'data_dt']).fillna(0)
mapper = {col: {value: index for index, value in enumerate(sorted(df_full_y_sum[col].unique()))} 
          for col in category_cols}

df_full_y_sum[category_cols] = df_full_y_sum[category_cols].apply(lambda x: x.map(mapper[x.name]))


    
x_minmax, y_minmax = (0,1), None

if x_minmax or y_minmax:
    x_train, x_test, y_train, y_test, scaler_dcit = data_split(df_full_y_sum, numeric_cols, category_cols, 
                                                               x_minmax=x_minmax, y_minmax=y_minmax, test_size=0.166)
else:
    x_train, x_test, y_train, y_test = data_split(df_full_y_sum, numeric_cols, category_cols, test_size=0.166)    

num_chid = len(set(df_full_y_sum.chid))
print('train:{}, test:{}'.format(x_train.shape[0]//num_chid, x_test.shape[0]//num_chid))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train.to_csv(x_train_file, index=False)
x_test.to_csv(x_test_file, index=False)

y_train.to_csv(y_train_file, index=False)
y_test.to_csv(y_test_file, index=False)
