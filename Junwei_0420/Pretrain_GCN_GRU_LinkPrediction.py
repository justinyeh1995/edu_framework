import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from torch_geometric.nn import GAE
from torch_geometric.utils import is_undirected, to_undirected
from sklearn.preprocessing import MinMaxScaler

from model.GCN_GRU import GCN_GRU
from utils import *

shop_col = 'stonc_6_label'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    
    embedding_dim = 64
    epochs = 200
    learning_rate = 0.0001

    edges_path = './data/edges/edges_stonc6.pkl'
    weight_path = 'Model_GCN_Link_Prediction'

    data_path = './data'

    chid_dict_file = 'sample/sample_50k_idx_map.npy'
    cdtx_file = 'sample/sample_50k_cdtx.csv'
    cust_file = 'preprocessed/df_cust.csv'

    chid_path = os.path.join(data_path, chid_dict_file)
    cdtx_path = os.path.join(data_path, cdtx_file)
    cust_path = os.path.join(data_path, cust_file)
    
    df_cdtx, df_cust, n_users, n_shops = read_sample_files(cdtx_path,
                                                       cust_path,
                                                       chid_path,
                                                       shop_col)
    
    list_months = sorted(df_cdtx.csmdt.unique())
    
    edge_dict = {}
    for i in tqdm(list_months):
        edge_pairs = df_cdtx[df_cdtx.csmdt==i][['chid', shop_col]].copy()
        edge_pairs.drop_duplicates(ignore_index=True, inplace=True)
        edge_pairs = torch.LongTensor(edge_pairs.to_numpy().T)

        if not is_undirected(edge_pairs):
            edge_pairs = to_undirected(edge_pairs)
    
        edge_dict[i] = edge_pairs
        
    neg_edges_dict = {}
    for i in tqdm(list(edge_dict.values())[12:12+12]):
        neg_edges_dict[i] = sample_neg_edges(i, n_users+n_shops, n_users)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GAE(GCN_GRU(input_dim, embedding_dim)).to(device)
    x = []
    y = []
    for i in list_months:
        cust_features = df_cust[df_cust.data_dt==i].iloc[:,2:].to_numpy()
        cust_features = torch.Tensor(cust_features)
        shop_features = torch.zeros(n_shops, cust_features.shape[1])

        temp_y = kind_y[df_cust.data_dt==i]
        x.append(torch.cat([cust_features, shop_features], 0).to(device))
        y.append(torch.Tensor(temp_y).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pos_edge_index = [i.to(device) for i in edge_dict.values()]
    neg_edges_index = [i.to(device) for i in neg_edges_dict.values()]
    
    def train():
        model.train()
        loss_ = 0
        for i in tqdm(range(10)):
            optimizer.zero_grad()
            z = model.encode(x[i:i+12], pos_edge_index[i:i+12])
            loss = model.recon_loss(z, pos_edge_index[i+12], neg_edges_index[i])
            loss_ += loss.item()
            loss.backward()
            optimizer.step()
        return loss_/10
    
    def test(pos_edge_index, neg_edge_index):
        model.eval()
        total_auc = 0
        total_ap = 0

        for i in range(10,12):

            with torch.no_grad():
                z = model.encode(x[i:i+12], pos_edge_index[i:i+12])
                auc, ap = model.test(z, pos_edge_index[i+12], neg_edges_index[i])

                total_auc += auc
                total_ap += ap

        return total_auc/2, total_ap/2
    
    for epoch in range(epochs):
        loss = train()

        auc, ap = test(pos_edge_index, neg_edges_index)
        print('Epoch: {:03d}, Train Loss:{:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch+1, loss, auc, ap))
        
    torch.save(model.encoder.state_dict(), weight_path)
    
    
if __name__ == '__main__':
    main()
    
    