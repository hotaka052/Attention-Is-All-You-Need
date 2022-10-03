import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, dim_ff)
        self.linear_2 = nn.Linear(dim_ff, d_model)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.linear_2(x)

        return x
