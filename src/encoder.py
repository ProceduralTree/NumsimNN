

# #+RESULTS:

from torch.nn import Module , Sequential
import torch.nn as nn
import torch as pt

class SinusoidalEmbedding(Module):
    def __init__(self, dimension: int , hidden : int , out:int):
        self.dim: int = dimension
        self.layer = Sequential(
            nn.Linear(self.dim * 2 , hidden ),
            nn.ReLU(),
            nn.Linear(hidden , out )
        )

        pass

    def embedd(self, t):
        arange = pt.linspace(0, 1.0, self.dim)
        embedded_sin = pt.sin(pt.outer(t, 1e-4**arange))
        embedded_cos = pt.cos(pt.outer(t, 1e-4**arange))
        return pt.cat([embedded_sin, embedded_cos])

    def forward(self, t):
        input_tensor = self.embedd(t)
        output_tensor = self.layer(input_tensor)
        return output_tensor
