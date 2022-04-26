import torch
import dgl
from pytorch3d.ops import knn_points
import frnn
from torch.nn.utils import spectral_norm as sp_norm

from gcn_lib.nn import MLP
from typing import List, Optional, Tuple
from pointnet2_ops.pointnet2_utils import grouping_operation
# the main purpose of this module is to support multi-batch training more efficiently


def knn_query(k, xyz1, xyz2=None):
    if xyz2 is None:
        xyz2 = xyz1
    dist, nbr_idxs, _ = knn_points(
        xyz1, xyz2,
        K=k,
        return_nn=False,
        return_sorted=True
    )
    return dist, nbr_idxs


def ball_query(radius, sample, xyz1, xyz2=None, knn_padding=True):
    if xyz2 is None:
        xyz2 = xyz1
    # If the ball neighborhood points are less than nsample,
    # than use the knn neighborhood points
    dist, nbr_idxs, _, _ = frnn.frnn_grid_points(
        xyz1, xyz2,
        K=sample,
        r=radius,
        grid=None, return_nn=False, return_sorted=True
    )

    if knn_padding:
        _, knn_nbr_idxs, _ = knn_points(
            xyz1, xyz2,
            K=sample,
            return_nn=False,
            return_sorted=True
        )
        nbr_idxs[nbr_idxs == -1] = knn_nbr_idxs[nbr_idxs == -1]
    return dist, nbr_idxs


class Dilated(torch.nn.Module):
    """
    Find dilated neighbor from neighbor list
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Dilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self,
                edge_index                 # [B, N, k]
                ):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index.view(2, -1, randnum)
                return edge_index.view(2, -1)
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index


class DilatedKnnGraph(torch.nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = Dilated(k, dilation, stochastic, epsilon)

    def forward(self, x):
        """
        x: [B, N, C]
        """
        _, knn_idx = knn_query(self.k, x)
        knn_idx = self._dilated(knn_idx)
        return knn_idx


def build_shared_mlp(mlp_spec: List[int], norm: str = 'batch', sn: bool = False):
    layers = []
    use_bias = not norm in ['batch', 'ins']
    for i in range(1, len(mlp_spec)):
        if sn:
            layers.append(
                sp_norm(torch.nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not use_bias))
            )
        else:
            layers.append(
                torch.nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not use_bias)
            )

        if norm == 'batch':
            layers.append(torch.nn.BatchNorm2d(mlp_spec[i]))
        elif norm == 'ins':
            layers.append(torch.nn.InstanceNorm2d(mlp_spec[i]))
        elif norm == 'none':
            pass
        else:
            raise Exception(f'Unsupported normalization: {norm}')

        layers.append(torch.nn.LeakyReLU(0.2))

    return torch.nn.Sequential(*layers)


def conv_bn_layer(in_feat, out_feat, act=False, norm='batch', sn=False):
    layers = []
    use_bias = not norm in ['batch', 'ins']
    if sn:
        layers.append(
            sp_norm(torch.nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=not use_bias))
        )
    else:
        layers.append(
            torch.nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=not use_bias)
        )

    if norm == 'batch':
        layers.append(torch.nn.BatchNorm2d(out_feat))
    elif norm == 'ins':
        layers.append(torch.nn.InstanceNorm2d(out_feat))
    elif norm == 'none':
        pass
    else:
        raise Exception(f'Unsupported normalization: {norm}')

    if act:
        layers.append(torch.nn.LeakyReLU(0.2))

    return torch.nn.Sequential(*layers)


class EdgeConv(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 k=9,
                 dilation=1,
                 mlp_layer=True,
                 aggregate='max',
                 bn=True,        # use batch norm or not
                 insn=False,     # use instance norm or not
                 sn=False,       # use spectral norm or not
                 **kwargs):
        super(EdgeConv, self).__init__()

        if insn and bn:
            raise Exception('Cant use batch normalization and instance normalization at the same time')
        if bn:
            self.norm = 'batch'
        elif insn:
            self.norm = 'ins'
        else:
            self.norm = 'none'

        self.k = k // dilation
        self.dilated_knn_graph = DilatedKnnGraph(k, dilation, **kwargs)

        self.edge_affine = conv_bn_layer(in_feat, out_feat//2, act=True, norm=self.norm, sn=sn)
        self.node_affine = conv_bn_layer(in_feat, out_feat//2, act=True, norm=self.norm, sn=sn)

        if mlp_layer:
            self.mlp = build_shared_mlp([out_feat//2, out_feat//2, out_feat], norm=self.norm, sn=sn)
        else:
            self.mlp = conv_bn_layer(out_feat//2, out_feat, norm=self.norm, sn=sn, act=False)

        if aggregate == 'sum':
            self.aggregate_fn = lambda y: torch.sum(y, dim=-1, keepdim=True)
        elif aggregate == 'max':
            self.aggregate_fn = lambda y: torch.max(y, dim=-1, keepdim=True)[0]
        elif aggregate == 'min':
            self.aggregate_fn = lambda y: torch.min(y, dim=-1, keepdim=True)[0]
        elif aggregate == 'mean':
            self.aggregate_fn = lambda y: torch.mean(y, dim=-1, keepdim=True)
        else:
            raise Exception(f'Unsupported aggregation mode {aggregate}')

    def forward(self, feat, pos=None):
        # feat: [B, C, N]
        if len(feat.shape) == 4 and feat.shape[-1] == 1:
            feat = feat.squeeze(-1)
        feat = feat.permute(0, 2, 1).contiguous()   # [B, N, C]
        if pos is not None:
            knn_idx = self.dilated_knn_graph(pos)    # [B, N, k]
        else:
            knn_idx = self.dilated_knn_graph(feat)    # [B, N, k]
        feat = feat.permute(0, 2, 1).contiguous()  # [B, C, N]
        knn_idx = knn_idx.type(torch.int32).contiguous()
        center_feat = feat.unsqueeze(-1)
        feat = grouping_operation(feat, knn_idx)  # [B, C, N, k]

        edge_feat = feat - center_feat  # [B, C, N, k]
        feat = self.node_affine(feat) + self.edge_affine(edge_feat)
        feat = self.aggregate_fn(self.mlp(feat))  # [B, C, N, 1]
        return feat


class IDGCNLayer(torch.nn.Module):
    """Inception-DenseGCN layer from PU-GCN: https://arxiv.org/abs/1912.03264"""
    def __init__(self,
                 in_feats,
                 out_feats,
                 bn=True,     # use batch norm or not
                 insn=False,  # use instance norm or not
                 ln=False,    # use layer norm or not
                 sn=False,
                 residual=True):
        super(IDGCNLayer, self).__init__()

        if insn and bn:
            raise Exception('Cant use batch normalization and instance normalization at the same time')
        if bn:
            self.norm = 'batch'
        elif insn:
            self.norm = 'ins'
        else:
            self.norm = 'none'

        # bottleneck layer
        self.btn = conv_bn_layer(in_feats, in_feats//4, act=False, norm=self.norm, sn=sn)
        self.GCN1 = EdgeConv(in_feats//4, in_feats//4, k=20, dilation=1,
                             aggregate='max', mlp_layer=True, bn=bn, insn=insn, sn=sn)
        self.GCN2 = EdgeConv(in_feats//4, in_feats//4, k=20, dilation=2,
                             aggregate='max', mlp_layer=True, bn=bn, insn=insn, sn=sn)
        self.decoder = conv_bn_layer(in_feats//4*3, out_feats, act=True, norm=self.norm, sn=sn)

        self.use_layernorm = ln
        if ln:
            self.use_layernorm = ln
            self.layernorm = torch.nn.LayerNorm([out_feats])

        self.residual = residual
        if self.residual:
            self.skip_layer = conv_bn_layer(in_feats, out_feats, act=False, norm=self.norm, sn=sn)

    def forward(self, feature):
        # [B, C, N, 1]
        if self.residual:
            skip_connection = self.skip_layer(feature.clone())   # [B, C_out, N, 1]
        feature = self.btn(feature)  # [B, C//4, N, 1]
        local_knn_idx = knn_query(9, feature.squeeze(-1).permute(0, 2, 1).contiguous())[1]
        feature = feature.squeeze(-1).contiguous()
        local_knn_idx = local_knn_idx.type(torch.int32).contiguous()
        local_feature = grouping_operation(feature, local_knn_idx)  # [B, C//4, N, k]

        local_max = torch.max(local_feature, dim=-1, keepdim=True)[0]  # [B, C//4, N, 1]
        feat1 = self.GCN1(feature.squeeze(-1))  # [B, C//4, N, 1]
        feat2 = self.GCN2(feature.squeeze(-1))  # [B, C//4, N, 1]

        feature = torch.cat([local_max, feat1, feat2], dim=1)
        feature = self.decoder(feature)

        if self.use_layernorm:
            B, C, N, _ = feature.shape
            feature = feature.squeeze(-1).permute(0, 2, 1).reshape(-1, C)   # [B*N, C]
            feature = self.layernorm(feature)
            feature = feature.reshape(B, N, C).permute(0, 2, 1).unsqueeze(-1).contiguous()

        if self.residual:
            feature += skip_connection

        return feature
