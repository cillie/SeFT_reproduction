from itertools import chain
from torch.nn import Linear


def build_linear_dropout_model(n_layers, width, dropout, linear_kwargs, name=None):
    if dropout > 0.0:
        layers = list(chain(*()))
