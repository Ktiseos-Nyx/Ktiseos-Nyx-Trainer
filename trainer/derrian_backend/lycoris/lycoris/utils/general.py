import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math

def product(xs: list[int | float]):
    res = 1
    for x in xs:
        res *= x
    return res

