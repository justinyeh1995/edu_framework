# 簡介: 

此程式框架的用途是幫助多任務實驗的協作與開發，提供了實驗的訓練、測試、Debug、前處理用的共用模組，並且支援Checkpoint，讓每次實驗產生的最佳模型可以被儲存以供測試使用，也提供logging的機制，以讓訓練過程中的模型的成效可以用Tensorboard來隨時檢視。

於每次實驗，根據我們所制定的規範創立一個全新的實驗設定資料夾，在其中定義模型、模型參數、各任務衡量指標、資料前處理方法，即可與我們的實驗共用模組進行串接整合，讓每一次的實驗都可以被簡易地複製、衡量、調整。

以下將進一步介紹此框架的 1. 安裝方法 2. 資料夾架構 3. 實驗執行方式 3. 實驗設定方法 4. 小工具 5. 範例檔說明 

此程式為beta版，若於使用中有疑問或建議，都可以隨時提供給我們。

# 資料夾架構 

以下為資料夾架構，標上 * 的檔案為實驗執行或資料下載後，才會生成的檔案或資料夾；標上 V 的為特定實驗專屬之檔案夾。
```
.
├── data                                             # 實驗資料
|    ├── source                                        # 存放原始資料 
|    |      ├──   download_data_from_google_drive.ipynb  # 從google_dirve下載原始資料用
|    |      ├── * google_drive.json                      # 串接google_drive用的api-keys，下載方式參考 download_data_from_google_drive.ipynb
|    |      ├── * sample_chid.txt                        # 原始資料
|    |      ├── ...                                        ... 
|    |      └── * sample_zip_if_cca_y.csv                # 原始資料 
|    ├── * sample                                      # 存放原始資料downsample後的資料
|    ├── *V [experiment_group_name]                     # 存放特定類型的實驗(e.g., rnn)所需之資料
|    |      ├── * tmp                                    # 中繼檔
|    |      └── * result                                 # 結果檔
|    ├── *V [experiment_name]                           # 存放特定實驗(e.g., ex1)用資料
|    |      ├── * tmp                                    # 中繼檔
|    |      └── * result                                 # 結果檔
|    ├── *V [experiment_name]                           # 存放特定實驗(e.g., ex1)用資料
|    └── ... 
|
├── common                                           # 共用模組           
|    ├── ETLBase.py                                     # 小工具 
|    ├── utils.py                                       # 小工具 
|    ├── pl_module.py                                # 內含多任務實驗共用之抽象模組 
|    └── __init__.py 
|    
├── experiments                                      # V 實驗模組 
|    ├── V [experiment_name]                           # V 特定實驗之實驗模組 
|    |      ├── V experiment_module.py                    # V 實驗設定模組                                
|    |      ├── V model.py                                # V 模型模組 
|    |      ├── V dataset_builder.py                   # V 資料前處理模組 
|    |      ├── V preprocess.py                        # V 資料前處理模組 
|    |      └── V __init__.py                        
|    └── V [experiment_name]
|    └── ...
|
├── *checkpoint                                      # 儲存模型暫存檔 
|    ├── *V [experiment_name]                          # 儲存特定實驗的模型暫存檔
|    |      ├── *V epoch964-loss0.00.ckpt                 # V 最佳模型的暫存檔
|    |      └── *V last.ckpt                              # V 最後一個epoch的模型暫存檔   
|    ├── *V [experiment_name]
|    └── ... 
|
├── *logs                                              # 存放訓練LOG: 各任務by-iteration的成效、實驗參數、模型架構圖 
|    └── tensorboard 
|           ├── *V [experiment_name]                      # 儲存特定實驗的LOG 
|           |       ├── *V version_0                        # 儲存第1次實驗的LOG 
|           |       ├── *V version_1                        # 儲存第2次實驗的LOG 
|           |       └── ... 
|           ├── *V [experiment_name]                      
|           └── ... 
├── run_project.py                                    # 主要實驗運行用檔案 (python run_project.py -m <run_mode> -e <experiment_name> [-l <log_dir>] (dflt.=logs/tensorboard)
├── requirements.txt 
└── ReadMe.md 
```

# 實驗執行方法 

# 小工具 


# 範例 

# Old ReadMe: 

## 原始程式碼
```diff
! Under Construction !
```

You can check the latest sources with the command:
```
git clone git@github.com:udothemath/ncku_customer_embedding.git
```

## 安裝dependencies: 

```
sh install_packages.sh
```

### ToDo: add package version
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

## 如何設定與執行? 

### 1. Download Dataset from Google Drive 
* 至../data執行**download_data_from_google_drive.ipynb**進行訓練與測試資料下載

### 2. Preprocessing and Build TensorDataset 

* 先至../esun底下
* 分段執行
  * `python preprocess.py`: 將../data的資料進行downsampling和資料處理轉換，並將結果儲存於../esun/data/result。
  * `python dataset_builder.py`: 將preprocess.py的結果進一步轉換為模型所需之格式(i.e., TensorDataset)。
* 直接執行
  * `python dataset_builder.py`

### 3. 建立logging與checkpoint路徑

1.  建立**logs/tensorboard**路徑，並於其中建立ncku_customer_embedding資料夾，以儲存實驗產生之Tensorboard Logs。
2.  建立**checkpoint**資料夾，以儲存模型暫存檔。
3.  打開../esun執行**run_project.py**進行編輯。
    - 將TensorBoardLogger('/home/ai/work/logs/tensorboard',...)中的tensorboard路徑改為Step 1所創建的**logs/tensorboard路徑**。
    - 將ModelCheckpoint(... dirpath='./checkpoint',...)中的dirpath路徑改為Step 2的**checkpoint路徑**

### 4. 執行模型訓練、Debug或驗證

* 訓練: `python run_project.py -m train`
* Debug: 
  - `python run_project.py -m fastdebug` (快速執行一次validation_step和train_step)
  - `python run_project.py -m fit1batch` ([讓模型overfit一個batch](https://www.youtube.com/watch?v=nAZdK4codMk)) 
* 驗證: 
  - `python run_project.py -m test` (使用測試資料進行測試) 


## 如何監控訓練狀況? 

- 於terminal輸入`tensorboard --logdir [tensorboard/ncku_customer_embedding路徑]`，即可於瀏覽器開啟tensorboard查看訓練狀況(http://localhost:6006/)。


## 重要程式設定說明 

### Downsampling: 

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
### 如何修改模型參數? 

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

### 如何使preprocess.py認別其使用檔案的儲存路徑以及其產生的檔案之儲存路徑? 

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

## Test
Add my comment
