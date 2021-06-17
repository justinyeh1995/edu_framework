import os
from common.ETLBase import PipeConfigBuilder

origin_path = 'data/source'

CATEGORY_COLS = ['chid', 'bnsfg', 'iterm', 'mcc', 'scity']
NUMERIC_COLS = ['bnspt', 'timestamp_0', 'timestamp_1', 'objam']

config = PipeConfigBuilder()

config.setups(
    chid_file=os.path.join(origin_path, 'sample_chid.txt'),
    cdtx_file=os.path.join(origin_path, 'sample_zip_if_cca_cdtx0001_hist.csv'),
    cust_f_file=os.path.join(origin_path, 'sample_zip_if_cca_cust_f.csv'),
    category_cols=CATEGORY_COLS,
    numeric_cols=NUMERIC_COLS,
    cust_category_cols=['chid', 'masts', 'educd', 'trdtp', 'poscd'],
    cust_numeric_cols=['slam', 'first_mob', 'constant_change', 'sum_l2_ind',
                  'sum_u2_ind', 'constant_l2_ind', 'constant_u4_ind',
                  'growth_rate', 'monotone_down', 'monotone_up', 'data_dt'],
    target_cols=['chid', 'data_dt', 'objam_sum', 'objam_mean', 'trans_count', 'objam_mean_M3'],
    time_column_data_dt='data_dt',
    time_column_csmdt='csmdt',
    result_column_timestamp_1='timestamp_1',
    result_column_timestamp_0='timestamp_0',
    result_column_timestamp='timestamp',
    LEFT='left',
    INNER='inner',
    n_sample=50,
    window_size=120,
    test_size=2
)


