import numpy as np
import torch
import torch.nn as nn
import dgl.nn
from torch.nn.utils import spectral_norm as sn

class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim=128,
                 hidden_layer=3,
                 activation_first=False,
                 activation='relu',
                 use_spectral_norm=False):
        super(MLP, self).__init__()
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise Exception('Only support: relu, leaky_relu, sigmoid, tanh as non-linear activation')

        mlp_layer = []
        for l in range(hidden_layer):
            if l != hidden_layer-1 and l != 0:
                if use_spectral_norm:
                    mlp_layer += [sn(nn.Linear(hidden_dim, hidden_dim)), act_fn]
                else:
                    mlp_layer += [nn.Linear(hidden_dim, hidden_dim), act_fn]
            elif l == 0:
                if not activation_first:
                    if use_spectral_norm:
                        mlp_layer += [sn(nn.Linear(in_feats, hidden_dim)), act_fn]
                    else:
                        mlp_layer += [nn.Linear(in_feats, hidden_dim), act_fn]
                else:
                    if use_spectral_norm:
                        mlp_layer += [act_fn, sn(nn.Linear(in_feats, hidden_dim)), act_fn]
                    else:
                        mlp_layer += [act_fn, nn.Linear(in_feats, hidden_dim), act_fn]
            else:
                if use_spectral_norm:
                    mlp_layer += [sn(nn.Linear(hidden_dim, out_feats))]
                else:
                    mlp_layer += [nn.Linear(hidden_dim, out_feats)]
        self.mlp_layer = nn.Sequential(*mlp_layer)

    def forward(self, feat):
        return self.mlp_layer(feat)

