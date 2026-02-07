import torch
from torch import Tensor, nn
import torch.nn.functional as F

# LoRA Dropout as a Sparsity Regularizer for Overfitting Control
def lora_dropout_down(down: Tensor, x: Tensor, dropout_prob=0.5):
    """ A = A · diag(mA), mA ∼ Bern(1 − p)"""
    mask = torch.bernoulli(
        torch.ones(down.shape[1], device=down.device) * (1 - dropout_prob)
    )

    # Apply input dimension mask (columns of down-projection)
    lx = x @ (down * mask.view(1, -1)).t()
    return lx

def lora_dropout_up(up: Tensor, x: Tensor, dropout_prob=0.5):
    """ B = B⊤ · diag(mB )⊤ , mB ∼ Bern(1 − p)"""
    mask = torch.bernoulli(
        torch.ones(up.shape[0], device=up.device) * (1 - dropout_prob)
    )

    # Apply output dimension mask (rows of up-projection)
    lx = x @ (up * mask.view(-1, 1)).t()
    return lx

class AID(nn.Module):
    def __init__(self, dropout_prob=0.9):
        super(AID, self).__init__()
        self.p = dropout_prob

    def forward(self, x):
        if self.training:
            # Generate masks for positive and negative values
            pos_mask = torch.bernoulli(torch.full_like(x, self.p))
            neg_mask = torch.bernoulli(torch.full_like(x, 1 - self.p))

            # Apply masks to positive and negative parts
            pos_part = F.relu(x) * pos_mask
            neg_part = F.relu(-x) * neg_mask * -1

            return pos_part + neg_part
        else:
            # During testing, use modified leaky ReLU with coefficient p
            return self.p * F.relu(x) + (1 - self.p) * F.relu(-x) * -1

def product(xs: list[int | float]):
    res = 1
    for x in xs:
        res *= x
    return res

