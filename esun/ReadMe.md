# 安裝dependencies: 

```
pip install google-api-python-client
pip install oauth2client

pip install numpy 
pip install pandas 
pip install tqdm 
pip install feather-format
pip install tables

pip install sklearn
pip install torch

pip install torchmetrics
pip install pytorch-lightning
pip install lightning-bolts

pip install tensorboard

```

# 如何設定與執行? 

1.  至../data執行**download_data_from_google_drive.ipynb**進行訓練與測試資料下載
2.  於../esun建立data資料夾，並在其中建立**sample**、**tmp**、**result**三個資料夾，分別儲存downsampling後結果、中繼檔、以及資料處理後結果。
3.  至../esun執行**preprocess.py**，以將../data的資料進行downsampling和資料處理轉換，並將結果儲存於../esun/data/result。
4.  至../esun執行**dataset_builder.py**，以將../esun/data/result的檔案進一步轉換模型需求的格式，並將結果儲存於../esun/data/result。
5.  建立**logs/tensorboard**路徑，並於其中建立ncku_customer_embedding資料夾，以儲存實驗產生之Tensorboard Logs。
6.  建立**checkpoint**資料夾，以儲存模型暫存檔。
7.  打開../esun執行**run_project.py**進行編輯。
           - 將TensorBoardLogger('/home/ai/work/logs/tensorboard',...)中的tensorboard路徑改為Step 5所創建的**logs/tensorboard路徑**。
           - 將ModelCheckpoint(... dirpath='./checkpoint',...)中的dirpath路徑改為Step 6的**checkpoint路徑**
8.  執行**run_project.py**以進行模型訓練。


# 如何監控訓練狀況? 

- 於terminal輸入`tensorboard --logdir [tensorboard/ncku_customer_embedding路徑]`，即可於瀏覽器開啟tensorboard查看訓練狀況(http://localhost:6006/)。


# 重要程式設定說明 

## Downsampling: 

為了加速測試，**preprocess.py**做資料處理過程中，會進一步downsample至500名users，將**preprocess.py**中進行以下修改，即可考慮所有(50K)的users。

將
```python
sampled_chids = Sample_chids(
                      'sample_chids', 
                      [chids], 
                      result_dir = os.path.join(sample_path,'sampled_chids.npy'), 
                      n_sample = 500
           ) 
```
改為
```python
sampled_chids = Sample_chids(
                      'sample_chids', 
                      [chids], 
                      result_dir = os.path.join(sample_path,'sampled_chids.npy'), 
                      n_sample = None
           ) 
```
## 如何修改模型參數? 

至run_project.py修改: 
```python
config = {
           'hidden_dims': 64, 
           'n_layers': 2, 
           'cell': 'LSTM', 
           'bi': False, 
           'dropout': 0.5
}
``` 

至dataset_builder.py修改`dense_feat`、`sparse_feat`和`USE_CHID`以決定模型所使用的**類別型特徵**、**數值型特徵**以及**是否使用顧客id做為類別型特徵**。

## 如何使preprocess.py認別其使用檔案的儲存路徑以及其產生的檔案之儲存路徑? 

可將以下preprocess.py的路徑進行調整，`origin_path`是來源資料的路徑、`sample_path`是儲存來源資料的一個downsample的版本的路徑、`tmp_path`儲存preprocess過程中中繼檔的路徑、`result_path`儲存最終檔案的路徑。

```python
origin_path = '../data'
sample_path = 'data/sample'
tmp_path = 'data/tmp'
result_path = 'data/result'
chid_file = os.path.join(origin_path, 'sample_chid.txt')
cdtx_file = os.path.join(origin_path, 'sample_zip_if_cca_cdtx0001_hist.csv')
cust_f_file = os.path.join(origin_path, 'sample_zip_if_cca_cust_f.csv')
```
