import torch.nn as nn

class NS_internal_percept(nn.Module):
    def __init__(self, latent_dim=8, output_dim=3):
        super(NS_internal_percept, self).__init__()

        self.neural_scope = nn.Sequential(
            nn.Linear(latent_dim, output_dim),
        )
        self.neural_scope[0].weight.data.fill_(0)
        self.neural_scope[0].bias.data.fill_(0)

    def forward(self, internal_percept):
        return self.neural_scope(internal_percept.permute(0,2,3,1)).permute(0,3,1,2)