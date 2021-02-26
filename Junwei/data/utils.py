import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time
from sklearn.preprocessing import MinMaxScaler

def read_and_convert(file_path, sortby = None):
    t0 = time()

    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    print(f'Read {file_path}\ncost time : {time()-t0}')

    t0 = time()
    np_data = np.array(list(map(lambda x:list(x.values()), json_data.values())))
    
    print(f'轉換成np array cost time : {time() - t0}')
    
    column_names = list(json_data['0'].keys())

    t0 = time()
    if sortby:
        df = pd.DataFrame(np_data, columns = column_names).sort_values(by=[sortby])
    else:
        df = pd.DataFrame(np_data, columns = column_names)
        
    print(f'轉換成Data Frame cost time : {time() - t0}')
    
    return df

def data_split(df, numeric_cols=[], category_cols=[], test_size=0.16, x_minmax=None, y_minmax=None):
    
    numeric_cols += ['objam']
    x_train, x_test, y_train, y_test = [], [], [], []
    df = df[['chid'] + category_cols + numeric_cols].copy()
            
    for i in tqdm(sorted(df.chid.unique())):
        data = df[df.chid == i]
        last = data.shape[0] - 1
        test_num = round(data.shape[0]*test_size)            

        x_train.append(data.iloc[0:last - test_num])
        y_train.append(data.iloc[1:last - test_num + 1, [-1]])

        x_test.append(data.iloc[last - test_num: last])
        y_test.append(data.iloc[last - test_num + 1: last + 1, [-1]])

    x_train = pd.concat(x_train)
    y_train = pd.concat(y_train)
    
    x_test = pd.concat(x_test)
    y_test = pd.concat(y_test)
    
    if x_minmax or y_minmax:
        scaler_dcit = dict()
    
    if x_minmax:
        x_scaler = MinMaxScaler(feature_range=x_minmax)
        x_train[numeric_cols] = x_scaler.fit_transform(x_train[numeric_cols])
        x_test[numeric_cols] = x_scaler.transform(x_test[numeric_cols]) 
        
        scaler_dcit['x'] = x_scaler
    if y_minmax:
        y_scaler = MinMaxScaler(feature_range=y_minmax)  
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)    
        
        scaler_dict['y'] = y_scaler
         
    if x_minmax or y_minmax:
        return x_train, x_test, y_train, y_test, scaler_dcit
    else:
        return x_train, x_test, y_train, y_test