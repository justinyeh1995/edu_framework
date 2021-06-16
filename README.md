# 簡介: 

此程式框架的用途是幫助多任務實驗的協作與開發，提供了實驗的訓練、測試、Debug、前處理用的共用模組，並且支援Checkpoint，讓每次實驗產生的最佳模型可以被儲存以供測試使用，也提供logging的機制，以讓訓練過程中的模型的成效可以用Tensorboard來隨時檢視。

於每次實驗，根據我們所制定的規範創立一個全新的實驗設定資料夾，在其中定義模型、模型參數、各任務衡量指標、資料前處理方法，即可與我們的實驗共用模組進行串接整合，讓每一次的實驗都可以被簡易地複製、衡量、調整。

以下將進一步介紹此框架的 1. [資料夾架構](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E8%B3%87%E6%96%99%E5%A4%BE%E6%9E%B6%E6%A7%8B) 2. [實驗執行方法](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E5%AF%A6%E9%A9%97%E5%9F%B7%E8%A1%8C%E6%96%B9%E6%B3%95
) 3. [範例檔說明](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E7%AF%84%E4%BE%8B%E6%AA%94%E8%AA%AA%E6%98%8E
) 4. [小工具](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E5%B0%8F%E5%B7%A5%E5%85%B7) 

此程式為beta版，若於使用中有疑問或建議，可以於[意見回饋](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E6%84%8F%E8%A6%8B%E5%9B%9E%E9%A5%8B)提供給我們，我們將會對此框架進行調整。
另外，在把實驗納入此框架的過程中，麻煩也幫我們填寫[實驗紀錄表](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E5%AF%A6%E9%A9%97%E8%A8%98%E9%8C%84%E8%A1%A8)，已方便我們追蹤進度。

## 目錄: 

- [簡介](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E7%B0%A1%E4%BB%8B)
- [資料夾架構](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E8%B3%87%E6%96%99%E5%A4%BE%E6%9E%B6%E6%A7%8B)
- [實驗執行方法](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E5%AF%A6%E9%A9%97%E5%9F%B7%E8%A1%8C%E6%96%B9%E6%B3%95)
    - [Step 1: 安裝dependencies](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-1-%E5%AE%89%E8%A3%9Ddependencies)
    - [Step 2: 下載原始資料](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-2-%E4%B8%8B%E8%BC%89%E5%8E%9F%E5%A7%8B%E8%B3%87%E6%96%99)
    - [Step 3: 測試實驗是否可執行](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-3-%E6%B8%AC%E8%A9%A6%E5%AF%A6%E9%A9%97%E6%98%AF%E5%90%A6%E5%8F%AF%E5%9F%B7%E8%A1%8C)
    - [Step 4: 建構新實驗](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-4-%E5%BB%BA%E6%A7%8B%E6%96%B0%E5%AF%A6%E9%A9%97)
        - [1) 實驗模組](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#1-%E5%AF%A6%E9%A9%97%E6%A8%A1%E7%B5%84-experiment_modulepy)
        - [2) 模型](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#2-%E6%A8%A1%E5%9E%8B-modelpy)
        - [3) 前處理](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-dataset_builderpypreprocesspy)
    - [Step 5: 執行新實驗](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-5-%E5%9F%B7%E8%A1%8C%E6%96%B0%E5%AF%A6%E9%A9%97)
        -  [1) 實驗Debug](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#1-%E5%AF%A6%E9%A9%97debug)
        -  [2) 模型訓練與測試](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#2-%E6%A8%A1%E5%9E%8B%E8%A8%93%E7%B7%B4%E8%88%87%E6%B8%AC%E8%A9%A6)
        -  [3) TensorBoard-訓練成效監控](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)
        -  [4) CPU/GPU加速](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#4-gpucpu%E5%8A%A0%E9%80%9F)
- [範例檔說明](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E7%AF%84%E4%BE%8B%E6%AA%94%E8%AA%AA%E6%98%8E)
- [小工具](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E5%B0%8F%E5%B7%A5%E5%85%B7)
- [實驗紀錄表](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E5%AF%A6%E9%A9%97%E8%A8%98%E9%8C%84%E8%A1%A8)
- [意見回饋](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#%E6%84%8F%E8%A6%8B%E5%9B%9E%E9%A5%8B)


# 資料夾架構 

以下為資料夾架構，標上 * 的檔案為實驗執行後，才會生成的檔案或資料夾；標上 V 的為特定實驗專屬之實驗檔或檔案夾；

標上🟠為須由實驗者[自行建置](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-4-%E5%BB%BA%E6%A7%8B%E6%96%B0%E5%AF%A6%E9%A9%97)之實驗檔或資料夾；標上🟢為須執行之檔案，包含[檔案下載](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-2-%E4%B8%8B%E8%BC%89%E5%8E%9F%E5%A7%8B%E8%B3%87%E6%96%99)與[實驗運行](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-5-%E5%9F%B7%E8%A1%8C%E6%96%B0%E5%AF%A6%E9%A9%97)；標上🔵為須[自行下載](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#step-2-%E4%B8%8B%E8%BC%89%E5%8E%9F%E5%A7%8B%E8%B3%87%E6%96%99)之檔案。

```
.
├── data                                                 # 實驗資料
|    ├── source                                            # 存放原始資料 
|    |      ├── 🟢download_data_from_google_drive.ipynb      # 從google_dirve下載原始資料用
|    |      ├── 🔵google_drive.json                          # 串接google_drive用的api-keys，下載方式參考 download_data_from_google_drive.ipynb
|    |      ├── 🔵sample_chid.txt                            # 原始資料
|    |      ├── 🔵sample_idx_map.npy                         # 原始資料
|    |      ├── 🔵sample_zip_if_cca_cdtx0001_hist.csv        # ...
|    |      ├── 🔵sample_zip_if_cca_cust_f.csv               # ...
|    |      └── 🔵sample_zip_if_cca_y.csv                    # 原始資料 
|    | 
|    ├── * sample                                        # 存放原始資料downsample後的資料
|    | 
|    ├── *V [experiment_group_name]                      # 存放特定類型的實驗(e.g., rnn)所需之資料
|    |      ├── * tmp                                      # 中繼檔
|    |      └── * result                                   # 結果檔
|    ├── *V [experiment_name]                            # 存放特定實驗(e.g., ex1)用資料
|    |      ├── * tmp                                      # 中繼檔
|    |      └── * result                                   # 結果檔
|    ├── *V [experiment_name]                            
|    └── ... 
|
├── common                                               # 共用模組           
|    ├── ETLBase.py                                        # 前處理用小工具 
|    ├── utils.py                                          
|    ├── pl_module.py                                    # 內含多任務實驗共用之抽象模組 
|    └── __init__.py 
|    
├── experiments                                          # V 實驗模組 
|    ├──V🟠[experiment_name]                              # V 特定實驗之實驗模組 
|    |      ├──V🟠experiment_module.py                    # V 實驗設定模組                                
|    |      ├──V🟠model.py                                # V 模型模組 
|    |      ├──V🟠dataset_builder.py                      # V 資料前處理模組 
|    |      ├──V🟠preprocess.py                           # V 資料前處理模組 
|    |      └──V🟠__init__.py                        
|    └──V🟠[experiment_name]
|    └── ...
|
├── *checkpoint                                          # 儲存模型暫存檔 
|    ├── *V [experiment_name]                              # 儲存特定實驗的模型暫存檔
|    |      ├── *V epoch964-loss0.00.ckpt                    # V 最佳模型的暫存檔
|    |      └── *V last.ckpt                                 # V 最後一個epoch的模型暫存檔   
|    ├── *V [experiment_name]
|    └── ... 
|
├── *logs                                                # 存放訓練LOG: 各任務by-iteration的成效、實驗參數、模型架構圖 
|    └── tensorboard 
|           ├── *V [experiment_name]                       # 儲存特定實驗的LOG 
|           |       ├── *V version_0                         # 儲存第1次實驗的LOG 
|           |       ├── *V version_1                         # 儲存第2次實驗的LOG 
|           |       └── ... 
|           ├── *V [experiment_name]                      
|           └── ... 
├──🟢run_project.py                                      # 實驗執行檔 
├── requirements.txt 
└── ReadMe.md 
```



# 實驗執行方法 

## Step 1: 安裝dependencies 

首先將相關套件進行安裝。

執行: 
`sh install_packages.sh`

或:
```
pip install -r requirements.txt
```

或: 

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
若要使用Nvidia GPU，須安裝GPU版本pytorch，詳情請見：https://pytorch.org。


## Step 2: 下載原始資料 

* 方法一: 至data/source執行**download_data_from_google_drive.ipynb**進行以下原始資料的下載
```
🔵sample_chid.txt                            # 原始資料
🔵sample_idx_map.npy                         # 原始資料
🔵sample_zip_if_cca_cdtx0001_hist.csv        # ...
🔵sample_zip_if_cca_cust_f.csv               # ...
🔵sample_zip_if_cca_y.csv                    # 原始資料 
```



* 方法二: 自行下載以上資料至data/source。

若採用方法一，須至google developer platform下載🔵google_drive.json，串接google_drive用的api-keys，下載方式參考 download_data_from_google_drive.ipynb。

## Step 3: 測試實驗是否可執行 

執行FastDebug: 
`python run_project.py -m fastdebug -e ex1` 


此程式會對experiments/ex1資料夾所定義之實驗進行debug。過程中會對原始資料進行前處理，並將結果與佔存檔儲存於`data/sample`、`data/rnn/tmp`、`data/rnn/result`、`data/ex1/tmp`、`data/ex1/result`，接著，資料處理完後，就會進行1個step的training和validation，以快速驗證模型、程式的運作正常。

## Step 4: 建構新實驗: 

可以複製ex1資料夾，必將其改為實驗者欲命名的實驗名稱（e.g., ex2)，並修改其中的`experiment_module.py`/`model.py`/`dataset_builder.py`/`preprocess.py`。其中`experiment_module.py`為實驗模組，`model.py`為模型，`dataset_builder.py`和`preprocess.py`為前處理程式。

以下將以ex1為範例，分別說明此三類程式的建構方式: 

### 1) 實驗模組 (`experiment_module.py`)

實驗模組必須包含三個Class: `ExperimentConfig`、`ExperimentalMultiTaskDataModule`和`ExperimentalMultiTaskModule`，`ExperimentConfig`定義了實驗名稱、實驗參數、以及最佳模型的暫存檔名稱，`ExperimentalMultiTaskDataModule`定義了資料前處理、訓練以及測試資料，`ExperimentalMultiTaskModule`定義了多任務模型、任務名稱、任務目標函數以及任務成效衡量指標。

以下將針對撰寫方式進一步說明: 

**1. ExperimentConfig**:
```python 

class ExperimentConfig:
    # @ExperimentDependent 
    best_model_checkpoint = 'epoch07-loss0.00.ckpt' # 最佳模型的暫存檔名稱

    name = experiment_name  # 實驗名稱，需和資料夾名稱相同(e.g., ex1) 

    experiment_parameters = {                    # 實驗參數 
        "model_parameters": {                      # 模型參數: 會放到model.py的參數。
            "data_independent":{                     # 和資料無關的模型參數，(e.g., hidden_dims 中間層維度, n_layers 層數, ...) 
                'hidden_dims': 64,              
                'n_layers': 2, 
                'cell': 'LSTM', 
                'bi': False
            },
            "data_dependent": {                      # 和資料相關之模型參數，(e.g., dense_dims 輸入之數值型特徵維度, use_chid 是否使用顧客ID為特徵, out_dims 模型輸出的維度)  
                'dense_dims': dense_dims, 
                'sparse_dims': sparse_dims,
                'use_chid': use_chid, 
                'out_dims': out_dims,
                'class_outputs': class_outputs 
            }
        },
        "training_parameters":{                      # 訓練參數，(e.g., dropout, warmup_epochs, annealing_cycle_epochs) 
            "dropout": 0.5, 
            "warmup_epochs": 5, 
            "annealing_cycle_epochs": 40
        }
    }
```

* 其中，dropout/warmup_epochs/annealing_cycle_epochs若不提供，預測則分別會給0.5, 5, 40。warmup_epochs和annealing_cycle_epochs會輸入以下learning rate scheduler，進行learning rate動態調整，以加速模型訓練。

```python 
LinearWarmupCosineAnnealingLR(optimizer, 
 warmup_epochs=self._warmup_epochs, 
 max_epochs=self._annealing_cycle_epochs 
)
```
**2. ExperimentalMultiTaskDataModule**:


```python 
class ExperimentalMultiTaskDataModule(BaseMultiTaskDataModule):
    # @blockPrinting
    def prepare_data(self):
        self.train_dataset = train_dataset.run()[0] 
        self.test_dataset = test_dataset.run()[0]
```

* 於prepare_data定義train_dataset和test_dataset。此兩個物件須為torch.utils.data的TensorDataset物件。

**3. ExperimentalMultiTaskModule**:

```python 
import torch.nn.functional as F
class ExperimentalMultiTaskModule(BaseMultiTaskModule):

    def config_model(self, model_parameters, dropout): # 此處引入model.py的最top-level的nn.Module，此nn.Module吃model_parameters和dropout兩個參數，後面將於model.py進一步說明建構方式。
        return MultiTaskModel(
                model_parameters,
                dropout = dropout
            )
            
    def config_task_names(self):                                 # 此處定義模型輸出所對應之任務名稱 
        return ['objmean', 'tscnt', 'label_0']
        
    def config_loss_funcs(self): 
        return [F.mse_loss, F.mse_loss, F.binary_cross_entropy]  # 此處定義各任務之目標函數 
    
    

    def config_task_metrics(self):                               # 此處定義個任務之衡量指標名稱 
        return {
            'objmean': ['mse', 'mae'], 
            'tscnt': ['mse', 'mae'], 
            'label_0': ['acc', 'auc']
        }
    
    def config_metric_calculators(self):                         # (optional) 定義個指標名稱所對應之指標計算元件。若有新指標(非mse/mae/acc/auc)才需實作此函數。
        from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy, AUROC
        return {
            'mse': lambda: MeanSquaredError(compute_on_step=False), 
            'mae': lambda: MeanAbsoluteError(compute_on_step=False), 
            'acc': lambda: Accuracy(compute_on_step=False),
            'auc': lambda: AUROC(compute_on_step=False, pos_label=1)
        }
    def batch_forward(self, batch):                              # 此處根據模型的forward方式定義一個batch的forward，以供共用模組的training/validation/testing將ground_truth和outputs套用到loss上。
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        outputs = self(x_dense, x_sparse)
        ground_truths = objmean, tscnt, label_0
        return outputs, ground_truths
```

### 2) 模型 (`model.py`)

model.py的top-level module依照nn.Module既有方法進行實作，但須注意__init__需吃`model_parameters`和`dropout`，model_parameters為一dictionary，
內容為`ExperimentConfig.experiment_parameters["model_parameters"]` 中`"data_independent"`以及`"data_dependent"`底下的所有參數。

```python 
class MultiTaskModel(torch.nn.Module):
    def __init__(self, model_parameters, dropout=0.5):
        super(MultiTaskModel, self).__init__()

        # Load parameters: 
        hidden_dims = model_parameters['hidden_dims']
        n_layers = model_parameters['n_layers'] 
        cell = model_parameters['cell'] 
        bi = model_parameters['bi'] 
        dense_dims = model_parameters['dense_dims'] 
        sparse_dims = model_parameters['sparse_dims'] 
        use_chid = model_parameters['use_chid'] 
        out_dims = model_parameters['out_dims'] 
        class_outputs = model_parameters['class_outputs']

        # Build model blocks: 
        self.rnn = ET_Rnn(
            dense_dims, 
            sparse_dims, 
            hidden_dims, 
            n_layers=n_layers, 
            use_chid=use_chid,
            cell=cell, 
            bi=bi, 
            dropout=dropout
        )
        
        self.mlps = nn.ModuleList([
            MLP(
                self.rnn.out_dim, hidden_dims=[self.rnn.out_dim // 2], out_dim=od
            ) for od in out_dims
        ])

        # Parameters used in forward 
        self.class_outputs = class_outputs

    def forward(self, *x):
        x_dense, x_sparse = x 
        logits = self.rnn(x_dense, x_sparse)
        outs = []
        for mlp, is_class in zip(self.mlps, self.class_outputs):
            out = mlp(logits)
            if is_class:
                out = torch.sigmoid(out)
            outs.append(out)
        return outs


```

### 3) 資料前處理 (`dataset_builder.py`/`preprocess.py`)

此兩程式定義了產生TensorDataset物件之資料前處理data pipeline，其使用了我們的`ETLPro`物件進行處理模組的定義並進行pipeline的串接。


於不同的實驗或模型所需之資料前處理方式不一樣，因此亦可以自行創建自己的資料前處理方式。詳細使用方式將於**範例檔介紹**說明，`ETLPro`的使用方式則會於**小工具**介紹。


## Step 5: 執行新實驗: 

當新的實驗建構完成，建議依以下順序進行debug與訓練: 

### 1) 實驗Debug:

**fastdebug**

首先，執行fastdebug，確保即使模型與實驗設定修改後，訓練與驗證皆能順利執行: 

`python run_project.py -m fastdebug -e [實驗資料夾名稱]` 

**fit1batch** 

接著，為了確保模型設計是合理的，讓模型overfit一個訓練的batch，正常的狀況，loss要能夠持續下降，loss的監控參考 [TensorBoard-訓練成效監控](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)。

`python run_project.py -m fit1batch -e [實驗資料夾名稱] (-l [log_dir])` 

執行過程中，會存放TensorBoard的Log於資料夾`./logs/tensorboard/[實驗資料夾名稱]/version_[x]`。

### 2) 模型訓練與測試

**train** 

執行訓練: 

`python run_project.py -m train -e [實驗資料夾名稱] (-l [log_dir])`  

模型的validation/training performance的監控參考 [TensorBoard-訓練成效監控](https://github.com/udothemath/ncku_customer_embedding/blob/multitask_experiment_framework/README.md#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)。

訓練過程中，表現最佳的模型以及最後一個epoch的模型暫存檔(.ckpt)會被保存於`./checkpoint/[實驗資料夾名稱]`中，每增加一個epoch，就會被更新一次。

**test**

若要用測試資料檢測模型的最終結果，可以將`./checkpoint/[實驗資料夾名稱]`中，最佳模型的.ckpt檔指定給`experiment_module.py`中`ExperimentConfig`的`best_model_checkpoint`參數，接著執行: 

`python run_project.py -m test -e [實驗資料夾名稱]`

### 3) TensorBoard-訓練成效監控 

於terminal輸入`tensorboard --logdir logs/tensorboard/[實驗資料夾名稱]`，即可於瀏覽器開啟TensorBoard查看訓練狀況(http://localhost:6006/ )。

* _Note: _若是執行fit1batch，其中val的圖表顯示的為在一個訓練的batch上衡量的成效。

若要修改log資料夾，可以於執行時，用`-l`指定新log資料夾路徑，e.g., 

`python run_project.py -m [fit1batch/train] -e [實驗資料夾名稱] -l MyDirectory`，此時，則於terminal輸入`tensorboard --logdir MyDirectory/[實驗資料夾名稱]`，即可查看該實驗結果。

若有進行實驗參數調整，可於 http://localhost:6006/#hparams 查看各實驗參數下的模型成效。

### 4) GPU/CPU加速：

至`run_project.py`設定: 

1. `pin_memory=True` 
2. `num_workers`為CPU數量 
3. `gpus`為gpu數量。


# 範例檔說明


# 小工具 

## ETLPro 

## utils.blockPrint


# 實驗記錄表
|實驗名稱|模型名稱|任務中英名稱|已建構完成實驗資料夾|fastdebug運作無誤|fit1batch運作無誤|GPU成功加速|train運作無誤|參數調整完成|最佳模型test無誤|最佳模型.ckpt路徑|負責人|
|--|:--:|--|--|--|--|--|--|--|--|--|--|
|ex1|ETRNN|下月消費總金額(objmean)、下月消費次數(tscnt)、下月是否消費(label_0)|V|V|V| |V| | | |玉山團隊|
|ex2|ETRNN|下月消費總金額(objmean)、下月消費次數(tscnt)、下月是否消費(label_0)|V| | | | | | | |玉山團隊|
|ex3|TemDMGE|下月使用紅利(bnsfg)、下月使用分期(iterm)、下月消費商店(stonc_label)-cross doamin|| | | | | | | |成大團隊|
|ex4|TemDMGE_shared|下月使用紅利(bnsfg)、下月使用分期(iterm)、下月消費商店(stonc_label)-cross doamin|| | | | | | | |成大團隊|
|ex5|TemGCN|下月使用紅利(bnsfg)-single doamin|| | | | | | | |成大團隊|
|ex6|TemGCN|下月使用分期(iterm)-single doamin|| | | | | | | |成大團隊|
|ex7|TemGCN|下月消費商店(stonc_label)-single doamin|| | | | | | | |成大團隊|
|ex8|GCN_RNN|下月使用紅利(bnsfg)-single doamin|| | | | | | | |成大團隊|
|ex9|GCN_RNN|下月使用分期(iterm)-single doamin|| | | | | | | |成大團隊|
|ex10|GCN_RNN|下月消費商店(stonc_label)-single doamin|| | | | | | | |成大團隊|

# 意見回饋 

- [1] GNN模型輸入資料的方式與先前ETRNN的使用dataloader的方式不同 (static graph可不使用data loader、dynamic graph以每月的snapshots作為不同的graph)
- [2] Multitask如果是作為訓練任務來實現的話，每個任務會有不同的loss(Recall、AUC)，不確定紀錄訓練過程是否可以同時記錄多任務的loss
- 
-  


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
