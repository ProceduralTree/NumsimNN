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
class ChannelAttention(nn.Module):
    def __init__(self , channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels , channels//2, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 2 , channels ,1),
        )
        self.meanpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.actv = nn.Sigmoid()

    def forward(self,x:Tensor):
        mean_attention = self.mlp(self.meanpool(x))
        max_attention = self.mlp(self.maxpool(x))
        attention = self.actv(max_attention + mean_attention)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self , channels):
        super().__init__()
        kernel_size=7
        self.conv = nn.Sequential(
            nn.Conv2d(2 ,1 , kernel_size , padding=(kernel_size-1)//2),
        )
        self.actv = nn.Sigmoid()
        
    def forward(self,x:Tensor):
        mean_pool = pt.mean(x,dim=1 , keepdims=True)
        max_pool = pt.max(x,dim=1 , keepdims=True).values
        pool = pt.cat((mean_pool , max_pool) , dim=1)
        attention = self.conv(pool)
        attention = self.actv(attention)
        return x * attention
class ConvBlockAttention(nn.Module):
    def __init__(self , channels):
        super().__init__()
        self.attention = nn.Sequential(
            ChannelAttention(channels),
            SpatialAttention(channels),
        )
    
    def forward(self,x:Tensor):
        return self.attention(x)
