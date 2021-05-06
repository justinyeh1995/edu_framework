import torch
from torch_geometric.nn import GCNConv

class GCN_GRU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, entity_dim, emb_dims, window_size=12):
        super(GCN_GRU, self).__init__()
        self.window_size = window_size
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(x, entity_dim) for x in emb_dims])
        self.gcn1_dict = torch.nn.ModuleDict({str(i):GCNConv(in_channels, out_channels*2)
                                                   for i in range(window_size)})
        self.gcn2_dict = torch.nn.ModuleDict({str(i):GCNConv(out_channels*2, out_channels)
                                                   for i in range(window_size)})
        
        self.rnn = torch.nn.GRU(out_channels, out_channels, 1, batch_first=False)
        

    def forward(self, cat_x, num_x, edges, edge_weights=None):
        x = []
        for month in range(self.window_size):
            x_ = [emb_layer(cat_x[month][:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x_ = torch.cat(x_, -1)
            x_ = torch.cat([x_, num_x[month]], -1)
            x.append(x_)
        
        if edge_weights == None:
            gcn_embeddings1 = [self.gcn1_dict[str(i)](x[i], edges[i]).relu() for i in range(self.window_size)]
            gcn_embeddings2 = [self.gcn2_dict[str(i)](gcn_embeddings1[i], edges[i]).unsqueeze(0) for i in range(self.window_size)]
            gcn_embeddings = torch.cat(gcn_embeddings2, 0)
        else:
            gcn_embeddings1 = [self.gcn1_dict[str(i)](x[i], edges[i], edge_weights[i]).relu() for i in range(self.window_size)]
            gcn_embeddings2 = [self.gcn2_dict[str(i)](gcn_embeddings1[i], edges[i], edge_weights[i]).unsqueeze(0) for i in range(self.window_size)]
            gcn_embeddings = torch.cat(gcn_embeddings2, 0)
            
        _ ,gcn_embeddings = self.rnn(gcn_embeddings)
        
        return gcn_embeddings.squeeze(0)
    
class SingleGCN_GRU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, entity_dim, emb_dims, window_size=12):
        super(SingleGCN_GRU, self).__init__()
        self.window_size = window_size
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(x, entity_dim) for x in emb_dims])
        self.gcn1 = GCNConv(in_channels, out_channels*2)
        self.gcn2 = GCNConv(out_channels*2, out_channels)
        
        self.rnn = torch.nn.GRU(out_channels, out_channels, 1, batch_first=False)
        

    def forward(self, cat_x, num_x, edges, edge_weights=None):
        x = []
        for month in range(self.window_size):
            x_ = [emb_layer(cat_x[month][:, i]) for i, emb_layer in enumerate(self.emb_layers)]
            x_ = torch.cat(x_, -1)
            x_ = torch.cat([x_, num_x[month]], -1)
            x.append(x_)
        
        if edge_weights == None:
            gcn_embeddings = [self.gcn1(x[i], edges[i]).relu() for i in range(self.window_size)]
            gcn_embeddings = [self.gcn2(gcn_embeddings[i], edges[i]).unsqueeze(0) for i in range(self.window_size)]
            gcn_embeddings = torch.cat(gcn_embeddings, 0)
        else:
            gcn_embeddings = [self.gcn1(x[i], edges[i], edge_weights[i]).relu() for i in range(self.window_size)]
            gcn_embeddings = [self.gcn2(gcn_embeddings[i], edges[i], edge_weights[i]).unsqueeze(0) for i in range(self.window_size)]
            gcn_embeddings = torch.cat(gcn_embeddings, 0)
            
        _, gcn_embeddings = self.rnn(gcn_embeddings)
        
        return gcn_embeddings.squeeze(0)


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        
        self.layer1 = torch.nn.Linear(in_channels, in_channels//2)
        self.layer2 = torch.nn.Linear(in_channels//2, out_channels)
        

    def forward(self, x):
        
        x = self.layer1(x).relu()
        
        return self.layer2(x)
    
class InnerDecoder(torch.nn.Module):
    def __init__(self):
        super(InnerDecoder, self).__init__()
        self.relu = torch.nn.ReLU6()

    def forward(self, user_embeddings, shop_embeddings):
        
        x = torch.mm(user_embeddings, shop_embeddings.T)
        x = self.relu(x)
        return x.sum(-1)