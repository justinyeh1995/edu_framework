import torch
from torch_geometric.nn import GCNConv

class GCN_GRU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, window_size=12):
        super(GCN_GRU, self).__init__()
        self.window_size = window_size

        self.gcn1_dict = torch.nn.ModuleDict({str(i):GCNConv(in_channels, out_channels*2)
                                                   for i in range(window_size)})
        self.gcn2_dict = torch.nn.ModuleDict({str(i):GCNConv(out_channels*2, out_channels)
                                                   for i in range(window_size)})
        
        self.rnn = torch.nn.GRU(out_channels, out_channels, 1, batch_first=False)
        

    def forward(self, x, edges):
        
        gcn_embeddings1 = [self.gcn1_dict[str(i)](x[i], edges[i]).relu() for i in range(self.window_size)]
        gcn_embeddings2 = [self.gcn2_dict[str(i)](gcn_embeddings1[i], edges[i]).unsqueeze(0) for i in range(self.window_size)]
        gcn_embeddings = torch.cat(gcn_embeddings2, 0)
        
        _ ,gcn_embeddings = self.rnn(gcn_embeddings)
        
        return gcn_embeddings.squeeze(0)

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        
        self.layer1 = torch.nn.Linear(in_channels, in_channels//2)
        self.layer2 = torch.nn.Linear(in_channels//2, 1)
        

    def forward(self, x):
        
        x = self.layer1(x).relu()
        
        return self.layer2(x)