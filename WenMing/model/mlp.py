import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, out_dim=1):
        super(MLP, self).__init__()
        input_dims = dense_dims + sum([ed for sf, fd, ed in sparse_dims])
        hidden_dims = [input_dims] + hidden_dims
        
        self.embedding_dict = nn.ModuleDict({
            sf:nn.Embedding(fd, ed, padding_idx=0) for sf, fd, ed in sparse_dims
        })
        
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(idim, odim), 
                nn.ReLU()
            ) for idim, odim in zip(hidden_dims[:-1], hidden_dims[1:])
        ])
        
        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)
        
    def init_embedding(self):
        for embed in self.embedding_dict.values():
            embed.reset_parameters()
            #embed.weight.data.uniform_(-1, 1)
            
    def forward(self, x_dense, x_sparse, sparse_feature):
        x = torch.cat([x_dense]+[self.embedding_dict[sp](x_sparse[:,i]) for i, sp in enumerate(sparse_feature)], dim=-1)
        out = self.layers(x)
        out = self.out_layer(out)
        
        return out