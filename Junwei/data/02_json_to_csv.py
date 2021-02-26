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

df_cdtx = read_and_convert(sample_cdtx_file, sortby = 'csmdt')
df_cust = read_and_convert(sample_cust_file)
df_cust.drop_duplicates(ignore_index=True, inplace=True)

df_cdtx.to_csv('sample_50k_cdtx.csv', index=False)
df_cust.to_csv('sample_50k_cust.csv', index=False)
