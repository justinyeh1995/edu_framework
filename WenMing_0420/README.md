 **莊文明 RE6094036@gs.ncku.edu.tw**

## 資料
訓練資料與測試資料，[下載](https://drive.google.com/drive/folders/1DZH1UGQjaZemjGc94v1RXwNrCSWIaHgF?usp=sharing)

## 檔案說明
1. 標頭數字，1_ : 資料前處理，2_ :預訓練模型，3_ :DownStreemTask
2. 模型：ETRnn、XGBoost
3. 五種任務：
-  objsum： 消費總金額(回歸)
-  tscnt：交易次數(回歸)
-  spcnt：交易商店種類數量(回歸)
-  label_0：下個月是否消費(分類)
-  label_mul：交易總金額區間(分類)
4. 新增 ranking loss 訓練方式。
