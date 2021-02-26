import torch
from torch import nn
import numpy as np

from time import time 
from scipy.sparse import csr_matrix
import scipy.sparse as sp

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class LightGCN(torch.nn.Module):
    def __init__(self, out_channels, n_user, n_shop, n_layers, edge_index):
        super(LightGCN, self).__init__()
        
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=n_user, embedding_dim=out_channels)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=n_shop, embedding_dim=out_channels)
        
        torch.nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        
        self.n_layers = n_layers
        self.Graph = self.getSparseGraph(n_user, n_shop, edge_index[0], edge_index[1])
        #self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        #self.conv2 = GCNConv(2 * out_channels, 2 * out_channels, cached=True)
        #self.conv3 = GCNConv(2 * out_channels, out_channels, cached=True)
        #self.embedding_dict = torch.nn.ModuleDict({cate_col:torch.nn.Embedding(cate_dim,
                                                                               #out_channels)
                                                   #for cate_col, cate_dim in category_dims.items()})

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def getSparseGraph(self, n_users, m_items, trainUser, trainItem):

        s = time()
        adj_mat = sp.dok_matrix((n_users + m_items, n_users + m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem-n_users)),
                                      shape=(n_users, m_items)).tolil()
        adj_mat[:n_users, n_users:] = R
        adj_mat[n_users:, :n_users] = R.T
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time()
        print(f"costing {end-s}s, saved norm_mat...")

        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(device)
        print("don't split the matrix")

        return Graph
    
    def forward(self, edge_index):
        
        #category_embeddings = [self.embedding_dict[cate_col](x[:,cate_idx].long()) for cate_col, cate_idx in
                               #category_dict.items()]
        #category_embeddings = torch.cat(category_embeddings, -1)
        
        #numeric_idx = torch.Tensor(list(numeric_dict.values())).long()
        
        #x = torch.cat([category_embeddings, x[:,numeric_idx]], -1)
        
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        g_droped = self.Graph
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        
        return light_out