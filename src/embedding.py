

# #+RESULTS:

from torch.nn import Module , Sequential
from torch import Tensor
import torch.nn as nn
import torch as pt

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dimension: int , hidden : int , out:int):
        super().__init__()
        self.dim: int = dimension
        self.net = nn.Sequential(
            nn.Linear(self.dim * 2 , hidden ),
            nn.ReLU(),
            nn.Linear(hidden , out )
        )


    def embedd(self, t:Tensor):
        # Create embedding frequencies
        i = pt.arange(self.dim, device=t.device)
        freqs = 10000 ** (-2 * i / self.dim)
        # Compute sin and cos embeddings
        embedded_sin = pt.sin(t[:, None] * freqs[None, :])
        embedded_cos = pt.cos(t[:, None] * freqs[None, :])
        return pt.cat([embedded_sin, embedded_cos], dim=1)

    def forward(self, t):
        input_tensor = self.embedd(t)
        output_tensor = self.net(input_tensor)
        return output_tensor
