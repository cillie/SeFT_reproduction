import torch
from torch import nn
from torch.nn.parameter import Parameter


class PositionalEncoding(nn.Module):
    def __init__(self, max_time=20000, n_dim=10):
        super().__init__()
        self.max_time = max_time
        self.n_dim = n_dim
        self._num_timescales = self.n_dim // 2
        self.timescales = Parameter(self.get_timescales(), requires_grad=False)

    def get_timescales(self):
        timescales = self.max_time ** torch.linspace(0, 1, self._num_timescales)
        return timescales

    def forward(self, times):
        scaled_time = times / self.timescales[None, None, :]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return signal


class SetAttentionLayer(nn.Module):
    dense_options = {"activation": "relu", "kernel_initializer": "he_uniform"}

    def __init__(
        self,
        n_layers=2,
        width=128,
        latent_width=128,
        aggregation_function="mean",
        dot_prod_dim=64,
        n_heads=4,
        attn_dropout=0.3,
    ):
        super().__init__()
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi =