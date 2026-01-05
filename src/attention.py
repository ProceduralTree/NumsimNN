from torch import Tensor
from torch import nn
from typing import Tuple
import torch as pt
from torch import nn

class MultiheadAttention(nn.Module):
    def __init__(self , in_ch , out_ch , heads):
        super().__init__()
        self.ch = in_ch
        # ensure propper divisibility 
        self.dim = in_ch // heads
        assert self.dim != 0
        self.heads = heads 
        self.feature_map = nn.Conv2d(in_ch , self.dim * self.heads * 3 , kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv2d(self.heads * self.dim, in_ch, kernel_size=1)
        self.norm = nn.GroupNorm(1,in_ch)
        return None

    def forward(self , x:Tensor):
        B , C , H , W = x.shape
        x = self.norm(x)
        
        QKV : Tuple[Tensor,Tensor,Tensor]  = self.feature_map(x).view(B,  3 * self.heads , self.dim ,-1).chunk(3 , dim=1) # type: tuple[Tensor, Tensor, Tensor]
        Q,K,V = QKV
        Q:Tensor =Q.permute(0,1,3,2)# B,Heads , HW , dim
        K:Tensor =K.permute(0,1,2,3)# B,Heads , dim , HW
        V:Tensor =V.permute(0,1,3,2)# B,Heads , HW, dim

        dot : Tensor= Q @ K # B , Heads , HW , HW
        attention: Tensor = self.softmax(dot*C**-0.5)
        values = attention @ V # B , Heads , HW , dim
        values = values.permute(0,1,3,2)
        values = values.reshape(B,self.heads * self.dim , H,W)
        return self.proj(values)
        #return self.proj(values)

from torch import nn
from torch import Tensor
class SpatialAttention(nn.Module):
    def __init__(self , in_ch , out_ch , heads):
        super().__init__()
        self.meanpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self,x:Tensor):
        input = pt.cat([x.mean((-1,-2)) , x.max((-1,-2))])
