# GNN


## 資料
由於檔案過大無法上傳，改以雲端硬碟放置檔案:[點我下載](https://drive.google.com/drive/folders/1RuglR5poPy7zi3AkXP-66fwLSBJOOnOt?usp=sharing)\
請將檔案放置到 ./Junwei_0420/data/sample \
並在 ./Junwei_0420/data 底下創建新資料夾 "preprocessed" \
創建完成後，執行preprocess.py
```shell

python3 preprocess.py

```

## Pretrain Model
- GCN+GRU
- SingleGCN+GRU


## Pretrain Task (.ipynb)
預訓練任務共有三種，分別是：
- Link Prediction
- Node Classification
    1. label_0
    2. label_mul

## Downstream Task (.ipynb)
下游任務共有五種，分別是：
-  objsum： 消費總金額(回歸)
-  tscnt：交易次數(回歸)
-  spcnt：交易商店種類數量(回歸)
-  label_0：下個月是否消費(分類)
-  label_mul：交易總金額區間(分類)


