# ncku_customer_embedding

## HackMD 
[進度更新連結](https://hackmd.io/@udothemath/ncku_embedding_ext)

## 環境設定
* pytorch 設定

```shell

pip uninstall -y torch
pip uninstall -y torch-scatter
pip uninstall -y torch-sparse
pip uninstall -y torch-cluster
pip uninstall -y torch-spline-conv
pip uninstall -y torch-geometric

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric

```

# sample顧客ID方式
從2018年的信用卡請款檔獨有的顧客ID中，抽取50000名顧客來作為後續的實驗資料。
(使用2018年的信用卡請款檔原因在於使用2018年的資料進行pre-train task時，確保顧客在2018年間是有跟商店建立互動關係的)


## 資料夾描述
* data -> sample出來的顧客交易紀錄檔與特徵檔

    由於檔案過大無法上傳，改以雲端硬碟放置檔案:[點我下載](https://drive.google.com/drive/folders/1Vw6jKoEhqmnmvbxh-kqh4xY-o2Ymr1d5?usp=sharing)

* Pre-train -> 預訓練模型：Node2Vec、GCN

* Downstream -> 下游任務模型：MLP

* GCN+RNN -> GCN+RNN (pretrain-model) + MLP (downstream model) 程式已包含pre-train與downstream
