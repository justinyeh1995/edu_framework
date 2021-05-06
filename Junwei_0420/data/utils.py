import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time

def read_and_convert(file_path, sortby = None):
    t0 = time()

    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    print(f'Read {file_path}\ncost time : {time()-t0}')

    t0 = time()
    np_data = np.array(list(map(lambda x:list(x.values()), json_data.values())))
    
    print(f'轉換成np array cost time : {time() - t0}')
    
    column_names = list(json_data['0'].keys())

    t0 = time()
    if sortby:
        df = pd.DataFrame(np_data, columns = column_names).sort_values(by=[sortby])
    else:
        df = pd.DataFrame(np_data, columns = column_names)
        
    print(f'轉換成Data Frame cost time : {time() - t0}')
    
    return df

def plus_month(date, n):
    month = int(date[5:7])+n
    if month > 12:
        year = int(date[:4])+1
        month -= 12
        new_date = f'{year:04d}-{month:02d}{date[7:]}'
    else:
        new_date = f'{date[:5]}{month:02d}{date[7:]}'
        
    return new_date

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