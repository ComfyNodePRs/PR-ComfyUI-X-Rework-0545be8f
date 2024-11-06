from torch import nn

# Define the LoRA class
class LoRA(nn.Module):
    def __init__(self, original_dim, rank):
        super(LoRA, self).__init__()
        self.rank = rank
        self.down_proj = nn.Linear(original_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, original_dim, bias=False)

    def forward(self, x):
        return self.up_proj(self.down_proj(x))