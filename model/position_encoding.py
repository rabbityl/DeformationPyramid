import math
import torch
from torch import nn


def sinusoidal (pos,  m=1) :
    if m > 0:
        x_position, y_position, z_position = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
        mul_term =  (2 ** torch.arange( m ,device=pos.device)).reshape(1,-1)
        sinx = torch.sin(x_position * mul_term)  #
        cosx = torch.cos(x_position * mul_term)
        siny = torch.sin(y_position * mul_term)
        cosy = torch.cos(y_position * mul_term)
        sinz = torch.sin(z_position * mul_term)
        cosz = torch.cos(z_position * mul_term)
        position_code = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        # x = torch.cat([ pos, position_code], dim= -1 )
        x = position_code
        return x
    else :
        return pos



