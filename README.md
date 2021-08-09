

# 簡介: 

此程式框架的用途是幫助多任務實驗的協作與開發，提供了實驗的訓練、測試、Debug、前處理用的共用模組，並且支援Checkpoint，讓每次實驗產生的最佳模型可以被儲存以供測試使用，也提供logging的機制，以讓訓練過程中的模型的成效可以用Tensorboard來隨時檢視。

於每次實驗，根據我們所制定的規範創立一個全新的實驗設定資料夾，在其中定義模型、模型參數、各任務衡量指標、資料前處理方法，即可與我們的實驗共用模組進行串接整合，讓每一次的實驗都可以被簡易地複製、衡量、調整。

以下將進一步介紹此框架的 1. [資料夾架構](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E8%B3%87%E6%96%99%E5%A4%BE%E6%9E%B6%E6%A7%8B) 2. [實驗執行方法](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E5%AF%A6%E9%A9%97%E5%9F%B7%E8%A1%8C%E6%96%B9%E6%B3%95
) 3. [範例檔說明](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E7%AF%84%E4%BE%8B%E6%AA%94%E8%AA%AA%E6%98%8E
) 4. [資料前處理小工具](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E5%B0%8F%E5%B7%A5%E5%85%B7) 

此程式為beta版，若於使用中有疑問或建議，可以於[意見回饋](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E6%84%8F%E8%A6%8B%E5%9B%9E%E9%A5%8B)提供給我們，我們將會對此框架進行調整。
另外，在把實驗納入此框架的過程中，麻煩也幫我們填寫[實驗紀錄表](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E5%AF%A6%E9%A9%97%E8%A8%98%E9%8C%84%E8%A1%A8)，已方便我們追蹤進度。

## 目錄: 

- [簡介](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E7%B0%A1%E4%BB%8B)
- [資料夾架構](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E8%B3%87%E6%96%99%E5%A4%BE%E6%9E%B6%E6%A7%8B)
- [實驗執行方法](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E5%AF%A6%E9%A9%97%E5%9F%B7%E8%A1%8C%E6%96%B9%E6%B3%95)
    - [Step 1: 安裝dependencies](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-1-%E5%AE%89%E8%A3%9Ddependencies)
    - [Step 2: 下載原始資料](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-2-%E4%B8%8B%E8%BC%89%E5%8E%9F%E5%A7%8B%E8%B3%87%E6%96%99)
    - [Step 3: 測試實驗是否可執行](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-3-%E6%B8%AC%E8%A9%A6%E5%AF%A6%E9%A9%97%E6%98%AF%E5%90%A6%E5%8F%AF%E5%9F%B7%E8%A1%8C)
    - [Step 4: 建構新實驗](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-4-%E5%BB%BA%E6%A7%8B%E6%96%B0%E5%AF%A6%E9%A9%97)
        - [1) 實驗模組](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#1-%E5%AF%A6%E9%A9%97%E6%A8%A1%E7%B5%84-experiment_modulepy)
        - [2) 模型](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#2-%E6%A8%A1%E5%9E%8B-modelpy)
        - [3) 前處理](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-dataset_builderpypreprocesspy)
    - [Step 5: 執行新實驗](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-5-%E5%9F%B7%E8%A1%8C%E6%96%B0%E5%AF%A6%E9%A9%97)
        -  [1) 實驗Debug](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#1-%E5%AF%A6%E9%A9%97debug)
        -  [2) 模型訓練與測試](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#2-%E6%A8%A1%E5%9E%8B%E8%A8%93%E7%B7%B4%E8%88%87%E6%B8%AC%E8%A9%A6)
        -  [3) TensorBoard-訓練成效監控](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)
        -  [4) CPU/GPU加速]()
- [範例檔說明](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E7%AF%84%E4%BE%8B%E6%AA%94%E8%AA%AA%E6%98%8E)
- [小工具](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E5%B0%8F%E5%B7%A5%E5%85%B7)
    - 1) [資料前處理工具: ETLBase](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86%E5%B7%A5%E5%85%B7-etlbase)
        - 1.1) [參數設定 (PipeConfigBuilder)](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#1-%E5%8F%83%E6%95%B8%E8%A8%AD%E5%AE%9A-pipeconfigbuilder)
        - 1.2) [前處理串接方式 (PipelineBuilder)](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#2-%E5%89%8D%E8%99%95%E7%90%86%E4%B8%B2%E6%8E%A5%E6%96%B9%E5%BC%8F-pipelinebuilder)
        - 1.3) [於.py定義前處理模組](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#3-%E6%96%BCpy%E5%AE%9A%E7%BE%A9%E5%89%8D%E8%99%95%E7%90%86%E6%A8%A1%E7%B5%84)
        - 1.4) [執行前處理並取得運算結果](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#4-%E5%9F%B7%E8%A1%8C%E5%89%8D%E8%99%95%E7%90%86%E4%B8%A6%E5%8F%96%E5%BE%97%E9%81%8B%E7%AE%97%E7%B5%90%E6%9E%9C)
        - 1.5) [中繼檔暫存功能](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#5-%E4%B8%AD%E7%B9%BC%E6%AA%94%E6%9A%AB%E5%AD%98%E5%8A%9F%E8%83%BD)
        - 1.6) [Dependency視覺化介紹](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#6-dependency%E8%A6%96%E8%A6%BA%E5%8C%96%E4%BB%8B%E7%B4%B9)
    - 2) [blockPrinting](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#utilsblockprint)



# 資料夾架構 
以下為資料夾架構，標上 * 的檔案為實驗執行後，才會生成的檔案或資料夾；標上 V 的為特定實驗專屬之實驗檔或檔案夾；

標上🟠為須由實驗者[自行建置](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-4-%E5%BB%BA%E6%A7%8B%E6%96%B0%E5%AF%A6%E9%A9%97)之實驗檔或資料夾；標上🟢為須執行之檔案，包含[檔案下載](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-2-%E4%B8%8B%E8%BC%89%E5%8E%9F%E5%A7%8B%E8%B3%87%E6%96%99)與[實驗運行](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-5-%E5%9F%B7%E8%A1%8C%E6%96%B0%E5%AF%A6%E9%A9%97)；標上🔵為須[自行下載](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#step-2-%E4%B8%8B%E8%BC%89%E5%8E%9F%E5%A7%8B%E8%B3%87%E6%96%99)之檔案。

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
|    |      ├──V🟠preprocess_operators.py                 # V 資料前處理模組 (see ex3)
|    |      ├──V🟠config_pipeline.py                      # V 資料前處理模組 (see ex3)
|    |      ├──V🟠connect_pipeline.py                     # V 資料前處理模組 (see ex3)
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

可以複製[ex3](https://github.com/udothemath/ncku_customer_embedding/tree/enhance_preprocess_module/experiments/ex3)資料夾，必將其改為實驗者欲命名的實驗名稱（e.g., ex4)，並修改其中的`experiment_module.py`/`model.py`/`config_pipeline.py`/`connect_pipeline.py`/`preprocess_operators.py`。其中`experiment_module.py`為實驗模組，`model.py`為模型，`config_pipeline.py`/`connect_pipeline.py`/`preprocess_operators.py`為前處理程式。

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

### 3) 資料前處理 (`preprocess_operators.py`/`connect_pipeline.py`/`config_pipeline.py`)

此三個程式定義了產生TensorDataset物件之資料前處理data pipeline，其使用了我們的`common/ETLBase.py`的`PipeConfigBuilder`物件進行處理模組的參數定義並用`PipeBuilder`進行串接的定義。

詳細使用方式於**小工具**介紹。


## Step 5: 執行新實驗: 

當新的實驗建構完成，建議依以下順序進行debug與訓練: 

### 1) 實驗Debug:

**fastdebug**

首先，執行fastdebug，確保即使模型與實驗設定修改後，訓練與驗證皆能順利執行: 

`python run_project.py -m fastdebug -e [實驗資料夾名稱]` 

**fit1batch** 

接著，為了確保模型設計是合理的，讓模型overfit一個訓練的batch，正常的狀況，loss要能夠持續下降，loss的監控參考 [TensorBoard-訓練成效監控](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)。

`python run_project.py -m fit1batch -e [實驗資料夾名稱] (-l [log_dir])` 

執行過程中，會存放TensorBoard的Log於資料夾`./logs/tensorboard/[實驗資料夾名稱]/version_[x]`。

### 2) 模型訓練與測試

**train** 

執行訓練: 

`python run_project.py -m train -e [實驗資料夾名稱] (-l [log_dir])`  

模型的validation/training performance的監控參考 [TensorBoard-訓練成效監控](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/README.md#3-tensorboard-%E8%A8%93%E7%B7%B4%E6%88%90%E6%95%88%E7%9B%A3%E6%8E%A7)。

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

```
python run_project.py -m [fit1batch/train] -e [實驗資料夾名稱] -g [GPU數量] -c [使用的cpu worker數量]
```

# 範例檔說明


# 小工具 

## 資料前處理工具: ETLBase 

為了能讓資料轉換為能夠輸入模型的形式，實驗建置過程中，往往會需要耗費許多的心力來進行資料的前處理，而對於不同的模型版本，又有可能會有相應的不一樣的前處理方式，隨著實驗的增加，前處理的程式也相應得變得越來越難以維護。另外，建立前處理程式的過程中，往往涉及到大量冗長的資料轉換，因此在開發過程中也容易因資料轉換而耽誤了開發時程。

因此，我們希望透過提供簡易好用的前處理工具，不只讓前處理程式更易於理解，也可以開發更快速。此前處理工具可以透過視覺化的方式，將前處理過程中的模塊、模塊的輸入、輸出，以及模塊之間的串連方式，以[有向圖(DAG)](https://zh.wikipedia.org/wiki/File:Tred-G.svg)的方式呈現，讓前處理的步驟與邏輯可以一目了然。另外，此工具也加入了資料中繼檔暫存功能，讓前處理過程中的中間產物，可以被以檔案的方式儲存起來，讓後續使用此中間產物的處理模塊可以快速仔入，進行後續模塊的調整。

此工具主要分為參數設定模組 PipeConfigBuilder 和 串接模組PipelineBuilder 這兩塊，前者用來設定前處理會用到的參數，例如window size、類別或數值型因子的欄位名稱等等，後者則是用來串接前處理模塊，以下我們將對此工具的使用方式進行簡單說明，詳細使用方式請參考[Jupyter Notebook - Tutorial of Pipeline Tools.ipynb](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/Tutorial%20of%20Pipeline%20Tools.ipynb)。

### 1) 參數設定 (PipeConfigBuilder)


假設前處理涉及兩個參數a,b，分別設定為1,2，可以用以下方是設定: 
```python
from common.ETLBase import PipeConfigBuilder
config = PipeConfigBuilder()
config.setups(a=1,b=2)
```
設定完成後，即可用view來呈現設定狀態: 

```python
config.view(summary=False)
```
![alt text](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/image/config.svg)


### 2) 前處理串接方式 (PipelineBuilder)

接著可以開始來定義前處理方式。

* 首先我們先透過以下方是建立好 PipelineBuilder: 
```python
from common.ETLBase import PipelineBuilder
pipe = PipelineBuilder(config)
```
PipelineBuilder帶入config，代表config中所建立的那些參數(a,b)，及可以在前處理程式串接過程中被取用。 

* 接著我們可以去定義資料處理模塊，舉例來說我們希望有一個可以把a,b進行相加的模塊: 
```python
@pipe._func_
def plus_a_b(a=1,b=2):
    return a+b
``` 
如此我們即可以把PipelineBuilder任別出此模塊。

* 最後進行模塊串接，假設我們想要讓c = a + b, d = a + c, e = d + d, f = b + d，我們可以以下面的方式進行串接: 

```python 
pipe.setup_connection('c = plus_a_b(a=a,b=b)')
pipe.setup_connection('d = plus_a_b(a=a,c)')
pipe.setup_connection('e = plus_a_b(d,d)')
pipe.setup_connection('f = plus_a_b(b,d)')
```

注意: 帶入setup_connection的python字串請勿加入換行字符\n，或使用expression來定義參數，如: `c=plus_a_b(a=(1*2), b=(6*9))`，PipeConfigBuilder的setup中定義或是出現於先前定義之setup_connection的output。

接著使用view即可呈現整張串接的結果: 

```python 
pipe.view(summary=False)
```
![alt text](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/image/pipe.svg)

### 3) 於.py定義前處理模組: 

前處理模塊可統一定義於一個.py中，並以以下方是載入PipelineBuilder中: 

```python 
from experiments.ex3.config_pipeline import config
pipe = PipelineBuilder(config, func_source='experiments.ex3.preprocess_operators')
``` 
如以上範例所式，此方式可以載入experiments/ex3/preprocess_operators.py中的所有函式作為串接的模塊使用。

一樣使用view即可檢視串接樣貌: 
```python 
pipe.view(summary=False)
```
![alt text](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/image/whole.svg)

### 4) 執行前處理並取得運算結果: 

我們所設計的工具，在定義資料串接方式時，前處理只會進行串接，並不會執行計算。
**但是**在開發前處理的過程中，常常會需要檢視前處理過程中的中繼產物，透過以下方法即可將前處理進行計算並取得某一模塊的輸出結果: 

```
pipe.f.get(verbose=True)
>> 6
```
例如我們想要取得上面pipe中所得之f的值，即可用get來取得，此時所有f所依賴的前處理模塊皆會進行執行。


### 5) 中繼檔暫存功能: 

若要使前處理重複使用的中繼產物可以更快被取得，我們提供暫存功能: 

```
pipe.setup_connection(
    'df_input, feature_map = extract_feature_cols_and_encode_categoricals(df_cdtx, numeric_cols=numeric_cols, category_cols=category_cols)',
    result_dir=[
                'df_input.feather',
                'feature_map.npy'
            ]
)
```

舉例來說，上面的extract_feature_cols_and_encode_categoricals函數會輸出兩個暫存檔，且此兩個檔案都會在後續資料處理被大量使用。為了減少開發時間，可以在result_dir給其各自的儲存檔名進行暫存，當程式執行到此函數時，其結果即會被自動儲存，下次執行時，即會直接載入所暫存的結果進行後續計算。

目前支援的格式有.feather/.h5/.npy三種格式，.feather和.h5為儲存pandas.DataFrame用的格式、.npy則是用來儲存numpy.array用的格式。

注意: 若要重從新執行此模塊的計算，須把暫存檔刪除才會重新執行，並產製結果，否則預設為直接使用暫存的結果。


### 6) Dependency視覺化介紹: 

我們亦提供了Hightlight Dependency的功能，舉例來說，透過以下方式即可把圖中，split_data所依賴的模組與資料產物都標住處來。
```
pipe.view_dependency('split_data', summary=False)
```
![alt text](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/image/dependency.svg)
詳細視覺化的進階操作請參考: [Jupyter Notebook - Tutorial of Pipeline Tools.ipynb](https://github.com/udothemath/ncku_customer_embedding/blob/enhance_preprocess_module/Tutorial%20of%20Pipeline%20Tools.ipynb)



## utils.blockPrint
用來把函數中會print出來的資訊都影藏起來。

使用方法: 

```python 
from common.utils import blockPrinting

@blockPrinting
def function_to_block():
      print('message to be blocked')
```



# 實驗記錄表
|實驗名稱|模型名稱|任務中英名稱|已建構完成實驗資料夾|fastdebug運作無誤|fit1batch運作無誤|train運作無誤|參數調整完成|最佳模型test無誤|最佳模型.ckpt路徑|
|--|:--:|--|--|--|--|--|--|--|--|
|ex1|ETRNN|下月消費總金額(objmean)、下月消費次數(tscnt)、下月是否消費(label_0)|V|V|V| | | | |

# 意見回饋 

- [1] GNN模型輸入資料的方式與先前ETRNN的使用dataloader的方式不同 (static graph可不使用data loader、dynamic graph以每月的snapshots作為不同的graph)
- [2] Multitask如果是作為訓練任務來實現的話，每個任務會有不同的loss(Recall、AUC)，不確定紀錄訓練過程是否可以同時記錄多任務的loss

# MODIFICATION: 

## New Preprocessing Model 
- [X] 設計新preprocess module (based on [pyflow-viz](https://pypi.org/project/pyflow-viz/)) 幫助data pipeline的視覺化。
      - [X] 視覺化完整 pipeline 
      - [X] 視覺化dependency 
- [X] 把此新preprocess module打包進ex3 作為使用範例
      - [X] 建立 ex3
      - [X] 把preprocessing functions 放進去
- [X] refine DataNode and SelectResult:
      - [X] implement get method on DataNode and SelectResult  
      - [X] allow passing of verbose variable. 
- [X] allow visualization of configuration variable. 
- [X] let DataNode takes kargs with ETLPro class argument 
- [X] build preprocess configuration object. 
    - [X] Using pprint to make config values better visualized in the DAG graph (given a max-line-length) 
- [X] 任何etl程式都可以被以圖形化的方式呈現。(只要function定義好、接好、config也定義好即可) 
- [X] 用 exec() 讓pipeline的撰寫和config的assignment可以更自然。 HARD!! 
    - [X] Allow all functions to be inserted into the PipelineBuilder in one go by 1. globals() 2. from package_name import * 
    - [ ] Allow object function to be inserted too. 
- [X] Use the setup_connection on the current ex1 pipeline. 
- [X] 放置data pipeline視覺化範例
- [ ] 讓此工具完整取代 experiment_module.py 中的 preprocessing. 
- [ ] Scan over the code and switch public func/vars to private ones. 

# ISSUES:
- [ ] 行內相容性問題 
      - [ ] cpu 環境 
      - [ ] gpu 環境 
- [ ] colab相容性問題 
