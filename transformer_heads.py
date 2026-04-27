import torch 


class FeedForward(torch.nn.Module):

    def __init__(self, n_embed):

        super().__init__()
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(n_embed, 2*n_embed), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*n_embed, n_embed))
        
        self.dropout = torch.nn.Dropout(p=0.1)

    
    def forward(self, x):

        out = self.next(x)
        out = self.dropout(out)
        return out 
