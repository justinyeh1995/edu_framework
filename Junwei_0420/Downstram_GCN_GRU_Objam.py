import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
from torch_geometric.nn import GAE
from torch_geometric.utils import is_undirected, to_undirected

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from model.GCN_GRU import GCN_GRU, Decoder
from utils import *

def main():
    shop_col = 'stonc_6_label'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    embedding_dim = 64
    epochs = 1000
    learning_rate = 0.0001
    batch_size = 1000

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
    ma = df_cdtx.groupby(['chid', 'csmdt']).objam.sum().max()

    edge_dict = {}
    for i in tqdm(list_months):
        edge_pairs = df_cdtx[df_cdtx.csmdt==i][['chid', shop_col]].copy()
        edge_pairs.drop_duplicates(ignore_index=True, inplace=True)
        edge_pairs = torch.LongTensor(edge_pairs.to_numpy().T)

        if not is_undirected(edge_pairs):
            edge_pairs = to_undirected(edge_pairs)

        edge_dict[i] = edge_pairs

    input_dim = df_cust.shape[1]-2

    model = GAE(GCN_GRU(input_dim, embedding_dim)).to(device)
    x = []
    y = []
    for i in list_months:
        cust_features = df_cust[df_cust.data_dt==i].iloc[:,2:].to_numpy()
        cust_features = torch.Tensor(cust_features)
        shop_features = torch.zeros(n_shops, cust_features.shape[1])

        temp_y = df_cust[df_cust.data_dt==i][['objam']].to_numpy()
        x.append(torch.cat([cust_features, shop_features], 0).to(device))
        y.append(torch.Tensor(temp_y).to(device))

    pos_edge_index = [i.to(device) for i in edge_dict.values()]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.encoder.load_state_dict(torch.load(weight_path))
    model.eval()

    model.decoder = Decoder(embedding_dim, 1)
    model.to(device)

    criterion = torch.nn.MSELoss()

    def train():
        model.train()
        train_output = np.array([])
        train_y = np.array([])
        for i in tqdm(range(10)):

            train_dataset = TensorDataset(y[i+12])
            train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size)
            for j, true_y in enumerate(train_loader):
                optimizer.zero_grad()
                z = model.encode(x[i:i+12], pos_edge_index[i:i+12])
                output = model.decode(z[j*batch_size:(j+1)*batch_size])
                loss = criterion(output, true_y[0])
                train_output = np.concatenate([train_output, output.cpu().detach().numpy().reshape(-1)])
                train_y = np.concatenate([train_y, true_y[0].cpu().detach().numpy().reshape(-1)])

                loss.backward(retain_graph=True)
                optimizer.step()


        return loss/10, train_output, train_y

    def test():
        model.eval()
        test_output = np.array([])
        test_y = np.array([])

        for i in range(10,12):

            with torch.no_grad():
                z = model.encode(x[i:i+12], pos_edge_index[i:i+12])
                output = model.decode(z[:n_users])

                test_output = np.concatenate([test_output, output.cpu().detach().numpy().reshape(-1)])
                test_y = np.concatenate([test_y, y[i+12].cpu().detach().numpy().reshape(-1)])

        return test_output, test_y

    for epoch in range(1, 400 + 1):
        loss, train_output, train_y  = train()

        test_output, test_y = test()

        train_RMSE = mean_squared_error(train_output*ma, train_y*ma, squared=False)
        test_RMSE = mean_squared_error(test_output*ma, test_y*ma, squared=False)

        train_MAE = mean_absolute_error(train_output*ma, train_y*ma)
        test_MAE = mean_absolute_error(test_output*ma, test_y*ma)

        print(f'epoch:{epoch+1}\ntrain loss:{train_RMSE:.0f},test loss:{test_RMSE:.0f}\ntrain MAE(mean):{train_MAE:.0f},test MAE(mean):{test_MAE:.0f}')

    
    
if __name__ == '__main__':
    main()
