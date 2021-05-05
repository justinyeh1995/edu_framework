import numpy as np
import pandas as pd
import torch
import math
from torch_sparse import coalesce
from torch_geometric.utils import is_undirected, to_undirected

def read_sample_files(sample_cdtx_file, sample_cust_file, sample_chid_dict, shop_col):
    
    print('Start reading cdtx file...')
    df_cdtx = pd.read_csv(sample_cdtx_file) 
    df_cdtx.sort_values('csmdt') # sort by date
    print('Finish reading cdtx file !')
    
    print('Start reading cust file...')
    df_cust = pd.read_csv(sample_cust_file)
    df_cust.drop_duplicates(ignore_index=True, inplace=True) # drop duplicate row
    print('Finish reading cust file !')
    
    idx_map = np.load(sample_chid_dict, allow_pickle=True).tolist()

    n_users = df_cdtx.chid.nunique()
    n_shops = df_cdtx[shop_col].nunique()
    
    for i , j in enumerate(sorted(df_cdtx[shop_col].unique())):
        idx_map[j] = i+n_users
    
    print('Start maping encodeing...')
    df_cdtx.chid = df_cdtx.chid.map(idx_map)
    df_cdtx[shop_col] = df_cdtx[shop_col].map(idx_map)

    df_cdtx.csmdt = df_cdtx.csmdt.apply(lambda x: x[:8]+'01')
    df_cdtx.objam = df_cdtx.objam.apply(lambda x: int(x))
    
    print('Finish !!')
    return df_cdtx, df_cust, n_users, n_shops

def train_test_split_edges(data, n_user, val_ratio=0.05, test_ratio=0.1, weighted = False):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.
    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)
    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    
    r, c = data.train_pos_edge_index
    r, c = torch.cat([r, c], dim=0), torch.cat([c, r], dim=0)
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    
    if weighted == True:
        weights = data.edge_weights
        data.edge_weights = None
        weights = weights[mask]
        weights = weights[perm]
        data.val_pos_edge_weight = weights[:n_v]
        data.test_pos_edge_weight = weights[n_v:n_v + n_t]
        data.train_pos_edge_weight = torch.cat([weights[n_v + n_t:],weights[n_v + n_t:]], dim=0)
        data.train_pos_edge_index, data.train_pos_edge_weight = coalesce(data.train_pos_edge_index,
                                                                        data.train_pos_edge_weight,
                                                                        num_nodes, num_nodes)
    else:
        data.train_pos_edge_index, _ = coalesce(data.train_pos_edge_index, None, num_nodes, num_nodes)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask[:n_user, n_user:].nonzero(as_tuple=False).t()
    neg_col = neg_col+n_user
    
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)
    
    return data

def sample_neg_edges(pos_edges, num_nodes, n_user):
    row , col = pos_edges
    mask = row < col
    row, col = row[mask], col[mask]
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    neg_row, neg_col = neg_adj_mask[:n_user, n_user:].nonzero(as_tuple=False).t()
    neg_col = neg_col+n_user
    perm = torch.randperm(row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    neg_edge_index = torch.cat([neg_row.view(1,-1), neg_col.view(1,-1)],0)
    
    return to_undirected(torch.LongTensor(neg_edge_index))

