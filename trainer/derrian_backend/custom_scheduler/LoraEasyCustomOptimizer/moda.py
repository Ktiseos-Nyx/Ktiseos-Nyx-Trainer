import math
import torch
import logging

def _default_primal_momentum1(k):
    return 1 / (k + 2)


class MODA(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        dual_momentum1=0,
        dual_momentum2=0,
        primal_momentum1=None,
        norm: str = "Spectral",
        norm_kwargs: dict = None,
        **kwargs,
    ):
        
        # Loop over the keys in the kwargs dictionary
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        if primal_momentum1 is None:
            primal_momentum1 = _default_primal_momentum1
        if norm_kwargs is None:
            norm_kwargs = {}

        if norm not in norm_dict:
            raise ValueError(f"Invalid norm: {norm}")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if dual_momentum1 < 0.0 or dual_momentum2 < 0.0:
            raise ValueError(f"Invalid momentum value: {dual_momentum1}, {dual_momentum2}")

        defaults = dict(
            lr=lr,
            k=0,
            dual_momentum1=dual_momentum1,
            dual_momentum2=dual_momentum2,
            primal_momentum1=primal_momentum1,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            k = group["k"]
            primal_momentum1 = group["primal_momentum1"]
            dual_momentum1 = group["dual_momentum1"]
            dual_momentum2 = group["dual_momentum2"]
            norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "mom_buff" not in state:
                    state["mom_buff"] = torch.clone(g)
                    state["z0"] = torch.clone(p.data)
                mom_buff = state["mom_buff"]
                z = state["z0"]

                mom_buff.mul_(1 - dual_momentum1).add_(g, alpha=dual_momentum1)
                if dual_momentum2 != 0:
                    mom_buff = mom_buff.mul(1 - dual_momentum2).add(g, alpha=dual_momentum2)

                update = norm_backend.lmo(mom_buff)
                z = z.add(update, alpha=lr * (k+2))
                p_mom1 = primal_momentum1(k) if callable(primal_momentum1) else primal_momentum1
                p.data.mul_(1 - p_mom1).add_(z, alpha=p_mom1)

            group["k"] = k + 1




@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    The quintic iteration coefficients are selected to maximize the slope at zero.
    This does not produce exactly UV^T, but an empirically useful approximation.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Norm:
    def lmo(self, g):
        raise NotImplementedError


class Spectral(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        g *= (d_out / d_in) ** 0.5
        return -g


class Sign(Norm):
    def __init__(self, zero_init=False):
        self.zero_init = zero_init

    def lmo(self, g):
        _, in_channels = g.shape
        return -(1 / in_channels) * torch.sign(g)



class ColNorm(Norm):
    """
    Column-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use True only for non-input layers.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=False, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = 1/math.sqrt(g.size(0))*torch.sqrt(torch.sum(g ** 2, dim=0, keepdim=True))
        if self.normalized:
            rms_values *= g.size(1)
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return -g


class RowNorm(Norm):
    """
    Row-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use False only for the input layer.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=True, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = torch.sqrt(torch.sum(g ** 2, dim=-1, keepdim=True))
        if self.normalized:
            rms_values *= math.sqrt(g.size(-1))
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return -g


class BiasRMS(Norm):
    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return -g



class SpectralConv(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        if g.ndim == 3:    # Conv1d
            out_channels, in_channels, k = g.shape
            g *= (out_channels / in_channels)**0.5 / k
        elif g.ndim == 4:   # Conv2d
            out_channels, in_channels, k, _ = g.shape
            g *= (out_channels / in_channels)**0.5 / (k ** 2)
        return -g
    

class Sinkhorn(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        eps = 1e-8
        for _ in range(self.steps):
            # Row Norm
            row_rms = torch.sqrt(torch.mean(g**2, dim=1, keepdim=True) + eps)
            g = g / row_rms

            # Column Norm
            col_rms = torch.sqrt(torch.mean(g**2, dim=0, keepdim=True) + eps)
            g = g / col_rms
        
        return -g


norm_dict = {
    "Spectral": Spectral,
    "SpectralConv": SpectralConv,
    "Sign": Sign,
    "ColNorm": ColNorm,
    "RowNorm": RowNorm,
    "BiasRMS": BiasRMS,
    "Sinkhorn": Sinkhorn,
}