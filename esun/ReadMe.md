# Mod
1. Edit readme
2. Add install_packages.sh file
3. Add download link in notebook
4. Add create_data_folder in preprocess.py
5. Refactorize pl_module.py / run_project.py 
6. Add package versions to install_packages.sh and ReadMe.md 
7. Edit readme: 設定與執行
8. Move create_data_folder into the initialization of the ETL object. 

# 原始程式碼
```diff
! Under Construction !
```

You can check the latest sources with the command:
```
git clone git@github.com:udothemath/ncku_customer_embedding.git
```

# 安裝dependencies: 

```
sh install_packages.sh
```

## ToDo: add package version
```
pip install google-api-python-client==2.5.0
pip install oauth2client==4.1.3

pip install numpy==1.18.5
pip install pandas==1.1.4
pip install tqdm==4.54.1
pip install feather-format==0.4.1
pip install tables==3.6.1

pip install sklearn==0.0
pip install torch==1.8.1

pip install torchmetrics==0.3.2
pip install pytorch-lightning==1.3.2
pip install lightning-bolts==0.3.3

pip install tensorboard==2.4.0
```

# 如何設定與執行? 

## 1. Download Dataset from Google Drive 
* 至../data執行**download_data_from_google_drive.ipynb**進行訓練與測試資料下載

## 2. Preprocessing and Build TensorDataset 

* 先至../esun底下
* 分段執行
  * `python preprocess.py`: 將../data的資料進行downsampling和資料處理轉換，並將結果儲存於../esun/data/result。
  * `python dataset_builder.py`: 將preprocess.py的結果進一步轉換為模型所需之格式(i.e., TensorDataset)。
* 直接執行
  * `python dataset_builder.py`

## 3. 建立logging與checkpoint路徑

1.  建立**logs/tensorboard**路徑，並於其中建立ncku_customer_embedding資料夾，以儲存實驗產生之Tensorboard Logs。
2.  建立**checkpoint**資料夾，以儲存模型暫存檔。
3.  打開../esun執行**run_project.py**進行編輯。
    - 將TensorBoardLogger('/home/ai/work/logs/tensorboard',...)中的tensorboard路徑改為Step 1所創建的**logs/tensorboard路徑**。
    - 將ModelCheckpoint(... dirpath='./checkpoint',...)中的dirpath路徑改為Step 2的**checkpoint路徑**

## 4. 執行模型訓練、Debug或驗證

* 訓練: `python run_project.py -m train`
* Debug: 
  - `python run_project.py -m fastdebug` (快速執行一次validation_step和train_step)
  - `python run_project.py -m fit1batch` ([讓模型overfit一個batch](https://www.youtube.com/watch?v=nAZdK4codMk)) 
* 驗證: 
  - `python run_project.py -m test` (使用測試資料進行測試) 


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

# Test
Add my comment