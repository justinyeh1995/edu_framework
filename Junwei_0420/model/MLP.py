import torch

class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, USE_CHID=False, n_user=None):
        super(MLP, self).__init__()
        self.a = USE_CHID
        if USE_CHID:
            self.user_embed = torch.nn.Embedding(n_user, 64)
            in_channels = in_channels+64-1
        self.layer1 = torch.nn.Linear(in_channels,  in_channels//2)
        self.layer2 = torch.nn.Linear(in_channels//2, out_channels)
        
        
        
    def forward(self, x):
        if self.a:
            user_embedding = self.user_embed(x[:,0].long())
            x = torch.cat([user_embedding, x[:,1:]], -1)
        x = self.layer1(x).relu()
        return self.layer2(x)