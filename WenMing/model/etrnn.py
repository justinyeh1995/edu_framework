import torch
from torch import nn
from torch.autograd import Variable

class ET_Rnn(torch.nn.Module):
    def __init__(self, dense_dims, sparse_dims, hidden_dims, out_dim=1, n_layers=1, use_chid=True, cell='GRU', bi=False, dropout=0, device='cpu'):
        super(ET_Rnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.cell = cell
        self.use_chid = use_chid
        self.device = device
        self.bi = bi
        
        self.input_dims = dense_dims + sum([ed for fd, ed in sparse_dims])
        self.embedding_list = nn.ModuleList([nn.Embedding(fd, ed, padding_idx=0) for fd, ed in sparse_dims])
        
        if use_chid:
            self.dropout = nn.Dropout(p=dropout)
            self.user_layer = nn.Linear(sparse_dims[0][1], sparse_dims[0][1])        
        
        if self.cell == 'LSTM':
            self.rnn = nn.LSTM(self.input_dims, hidden_dims, n_layers, batch_first=True, bidirectional=bi, dropout=0)
        elif self.cell == 'GRU':
            self.rnn = nn.GRU(self.input_dims, hidden_dims, n_layers, batch_first=True, bidirectional=bi, dropout=0)  
        
        self.out_layer = nn.Linear(hidden_dims*(bi+1), out_dim)
        
        self.init_embedding()
        
    def init_embedding(self):
        for embed in self.embedding_list:
            embed.reset_parameters()
            #embed.weight.data.uniform_(-1, 1)

    def init_hidden(self, x):
        if self.cell == 'LSTM':
            hidden = Variable(torch.zeros(self.n_layers*(self.bi+1), x.size(0), self.hidden_dims)).to(self.device)
            context = Variable(torch.zeros(self.n_layers*(self.bi+1), x.size(0), self.hidden_dims)).to(self.device)
            ret = (hidden, context)
        elif self.cell == 'GRU':
            hidden = Variable(torch.zeros(self.n_layers*(self.bi+1), x.size(0), self.hidden_dims)).to(self.device)
            ret = hidden
        
        return ret
            
    def forward(self, x_dense, x_sparse):
        if self.use_chid:
            user_embed = self.user_layer(self.embedding_list[0](x_sparse[:,:,0]))
            x = torch.cat([x_dense]+[user_embed]+[self.dropout(embed(x_sparse[:, :, i+1]))
                                         for i, embed in enumerate(self.embedding_list[1:])], dim=-1)
        else:
            x = torch.cat([x_dense]+[embed(x_sparse[:, :, i]) for i, embed in enumerate(self.embedding_list[:])], dim=-1)
        
        self.hidden = self.init_hidden(x)
        logits, self.hidden = self.rnn(x, self.hidden)
        out = self.out_layer(logits[:, -1])
        
        return out