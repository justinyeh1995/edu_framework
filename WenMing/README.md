 **莊文明 RE6094036@gs.ncku.edu.tw**

## 目標
- Pretrain -> 生成用戶Embedding。
- DownStream -> 預測用戶每月消費金額。

## 檔案說明
1. 標頭數字為執行的優先順序。
1. `0_sample_file_by_chid.ipynb` -> 依據 Sample 的 chid 從 raw data 抓出資料。
 (下載 [SampleData](https://drive.google.com/drive/folders/1Vw6jKoEhqmnmvbxh-kqh4xY-o2Ymr1d5?usp=sharing)，並放入 `./data/sample_50k/` 中。)
1. `1_data_process_*.ipynb` -> DownStream task 的資料前處理。(Normal: MLP, LinearSVR)
1. `2_*.ipynb` -> 建立DownStream task 模型。
1. `3_result.ipynb` -> 將 DownStream task 的 testing data 結果做評估及可視化。
1. `node2Vec_edge.ipynb` -> 使用2018年資料做Graph，訓練node2vec。
1. `node2Vec_edge_sw.ipynb` -> 使用Slideing Window，每個月建立一次Graph(一次一年的資料)，訓練多個node2vec。

## Model
- mlp.py -> Multilayer perceptron
- etrnn.py -> [E.T.-RNN: Applying Deep Learning to Credit Loan Applications](https://arxiv.org/pdf/1911.02496.pdf)
- node2vec.py -> 修改 [torch-geometric node2vec](https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/models/node2vec.py)，positive pair 額外加上預測平均消費金額的任務。
