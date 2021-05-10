# GNN


## 資料
檔案過大無法上傳，改以雲端硬碟放置檔案:[點我下載](https://drive.google.com/drive/folders/1RuglR5poPy7zi3AkXP-66fwLSBJOOnOt?usp=sharing)\
請將檔案放置到 `./Junwei_0420/data/sample`\
並在 `./Junwei_0420/data` 底下創建新資料夾 `preprocessed`\
創建完成後，執行`preprocess.py`
```shell

python3 preprocess.py

```

## Pretraining Model
- GCN_GRU
- SingleGCN_GRU


## Pretraining Task
預訓練任務共有三種，分別是：
- Link Prediction
- Node Classification
    1. label_0
    2. label_mul

以上的預訓練任務程式碼檔案名為：`Pretrain_{Pretraining Model}_{Pretraining Task}.ipynb`

模型訓練完成後，必須將模型權重(model weights)進行儲存\
因執行\b{Downstream Task}時，得將儲存下來的模型權重進行載入\
故在執行\b{Pretraining Task}時，需注意`weights_path`模型權重儲存的路徑與名稱。

## Downstream Task (.ipynb)
下游任務共有五種，分別是：
-  objsum： 消費總金額(回歸)
-  tscnt：交易次數(回歸)
-  spcnt：交易商店種類數量(回歸)
-  label_0：下個月是否消費(分類)
-  label_mul：交易總金額區間(分類)

以上的預訓練任務程式碼檔案名為：`Downstream_{Pretraining Model}_{Downstream Task}.ipynb`

執行\b{Downstream Task}時，得將\b{Pretraining Task}儲存下來的模型權重進行載入\
需注意`weights_path`模型權重載入的路徑與名稱。
