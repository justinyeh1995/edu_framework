import torch
from torch_geometric.nn import GCNConv

class MLP(torch.nn.Module):
    def __init__(self, category_dims, layer_dims, embedding_dim):
        super(MLP, self).__init__()
        self.layer_dims = layer_dims
        Linear_blokcs = [self.Linear_block(in_dim, out_dim)
                         for in_dim, out_dim in zip(self.layer_dims, self.layer_dims[1:])]
        self.model = torch.nn.Sequential(*Linear_blokcs)
        self.embedding_dict = torch.nn.ModuleDict({cate_col:torch.nn.Embedding(cate_dim,
                                                                               embedding_dim)
                                                   for cate_col, cate_dim in category_dims.items()})

    def Linear_block(self, in_dim, out_dim):
        block = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim),
                                    torch.nn.ReLU())
        return block

    def forward(self, x, category_dict, numeric_dict):
        
        category_embeddings = [self.embedding_dict[cate_col](x[:,cate_idx].long()) for cate_col, cate_idx in
                               category_dict.items()]
        category_embeddings = torch.cat(category_embeddings, -1)
        
        numeric_idx = torch.Tensor(list(numeric_dict.values())).long()
        
        x = torch.cat([category_embeddings, x[:,numeric_idx]], -1)
        
        x = self.model(x)
        
        return x
    
class MLP_with_pretrain(torch.nn.Module):
    def __init__(self, category_dims, layer_dims, embedding_dim):
        super(MLP_with_pretrain, self).__init__()
        self.layer_dims = layer_dims
        Linear_blokcs = [self.Linear_block(in_dim, out_dim)
                         for in_dim, out_dim in zip(self.layer_dims, self.layer_dims[1:])]
        self.model = torch.nn.Sequential(*Linear_blokcs)
        self.embedding_dict = torch.nn.ModuleDict({cate_col:torch.nn.Embedding(cate_dim,
                                                                               embedding_dim)
                                                   for cate_col, cate_dim in category_dims.items()})

    def Linear_block(self, in_dim, out_dim):
        block = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim),
                                    torch.nn.ReLU())
        return block

    def forward(self, x, user_embedding, category_dict, numeric_dict):
    
        category_embeddings = [self.embedding_dict[cate_col](x[:,cate_idx].long()) for cate_col, cate_idx in
                               category_dict.items()]
        category_embeddings = torch.cat(category_embeddings, -1)
        
        numeric_idx = torch.Tensor(list(numeric_dict.values())).long()
        
        x = torch.cat([category_embeddings, x[:,numeric_idx], user_embedding], -1)
        
        x = self.model(x)
        
        return x

class Whole_model_Node2Vec(torch.nn.Module):
    def __init__(self, pre_train_model, down_stream_model):
        super(Whole_model_Node2Vec, self).__init__()
        self.num_node = pre_train_model.adj.sizes()[0]
        self.pre_train_model = pre_train_model
        
        self.down_stream_model = down_stream_model

    
    def forward(self, x, category_dict, numeric_dict):
        user_embedding = self.pre_train_model(torch.arange(self.num_node))[x[:,0].long()]
        
        output = self.down_stream_model(x, user_embedding, category_dict, numeric_dict)
        
        return output

class Whole_model_GCN(torch.nn.Module):
    def __init__(self, pre_train_model, down_stream_model):
        super(Whole_model_GCN, self).__init__()
        
        self.pre_train_model = pre_train_model
        
        self.down_stream_model = down_stream_model

    
    def forward(self, x, x_feature, edge_index, category_dict, numeric_dict):
        user_embedding = self.pre_train_model(x_feature, edge_index, category_dict, numeric_dict)[x[:,0].long()]
        
        output = self.down_stream_model(x[:,1:], user_embedding, category_dict, numeric_dict)
        
        return output
    
class Whole_model_LGCN(torch.nn.Module):
    def __init__(self, pre_train_model, down_stream_model):
        super(Whole_model_LGCN, self).__init__()
        
        self.pre_train_model = pre_train_model
        
        self.down_stream_model = down_stream_model

    
    def forward(self, x, category_dict, numeric_dict):
        user_embedding = self.pre_train_model(1)[x[:,0].long()]
        
        output = self.down_stream_model(x[:,1:], user_embedding, category_dict, numeric_dict)
        
        return output

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, category_dims, window_size=12):
        super(Encoder, self).__init__()
        self.window_size = window_size
        self.embedding_dict = torch.nn.ModuleDict({cate_col:torch.nn.Embedding(cate_dim,
                                                                               out_channels)
                                                   for cate_col, cate_dim in category_dims.items()})
        self.gcn1_dict = torch.nn.ModuleDict({str(i):GCNConv(in_channels, 2 * out_channels, cached=True)
                                                   for i in range(window_size)})
        self.gcn2_dict = torch.nn.ModuleDict({str(i):GCNConv(2 * out_channels, out_channels, cached=True)
                                                   for i in range(window_size)})
        
        self.rnn = torch.nn.GRU(out_channels, out_channels, 1, batch_first=True)
        

    def forward(self, x, edge_index, category_dict, numeric_dict):
        new_x = []
        numeric_idx = torch.LongTensor(list(numeric_dict.values()))
        for feature in x:
            category_embeddings = [self.embedding_dict[cate_col](feature[:,cate_idx].long()) for cate_col, cate_idx in 
                                   category_dict.items()]
            category_embeddings = torch.cat(category_embeddings, -1)
            new_x.append(torch.cat([category_embeddings, feature[:,numeric_idx]], -1))
        
        gcn_embeddings = [self.gcn1_dict[str(i)](new_x[i], edge_index[i]).relu() for i in range(self.window_size)]
        gcn_embeddings = [self.gcn2_dict[str(i)](gcn_embeddings[i], edge_index[i]).unsqueeze(1) for i in range(self.window_size)]
        gcn_embeddings = torch.cat(gcn_embeddings, 1)
        
        _ ,gcn_embeddings = self.rnn(gcn_embeddings)
        
        return gcn_embeddings.squeeze(0)
    
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, category_dims):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        self.embedding_dict = torch.nn.ModuleDict({cate_col:torch.nn.Embedding(cate_dim,
                                                                               out_channels)
                                                   for cate_col, cate_dim in category_dims.items()})


    def forward(self, x, edge_index, category_dict, numeric_dict):
        category_embeddings = [self.embedding_dict[cate_col](x[:,cate_idx].long()) for cate_col, cate_idx in
                               category_dict.items()]
        category_embeddings = torch.cat(category_embeddings, -1)
        
        numeric_idx = torch.Tensor(list(numeric_dict.values())).long()
        
        x = torch.cat([category_embeddings, x[:,numeric_idx]], -1)
        
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        return x
    
class LinearDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearDecoder, self).__init__()
        self.layer1 = torch.nn.Linear(in_channels, 2 * in_channels)
        self.layer2 = torch.nn.Linear(2 * in_channels, out_channels)

    def forward(self, z, batch_r, batch_c):
        
        x = torch.cat([z[batch_r], z[batch_c]],1)
        
        x = self.layer1(x).relu()
        x = self.layer2(x)
        
        return x