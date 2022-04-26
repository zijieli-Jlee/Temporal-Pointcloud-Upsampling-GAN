import torch
import torch.nn
import dgl
from dgl.utils import expand_as_pair
import dgl.function as fn
import numpy as np
import frnn

from gcn_lib.graph_utils import DilatedKnnGraph, KNNGraph, FixedRadiusGraph
from gcn_lib.nn import MLP


class EdgeConv(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 k=9,
                 dilation=1,
                 mlp_layer=0,
                 aggregate='max',
                 **kwargs):
        super(EdgeConv, self).__init__()

        self.theta = torch.nn.Linear(in_feat, out_feat)
        self.phi = torch.nn.Linear(in_feat, out_feat)
        self.dilated_knn_graph = DilatedKnnGraph(k, dilation, **kwargs)

        self.mlp_layer = mlp_layer
        if self.mlp_layer > 1:
            self.mlp = MLP(out_feat, out_feat, hidden_dim=64, hidden_layer=self.mlp_layer,
                           activation='leaky_relu', activation_first=True)
        elif self.mlp_layer == 1:
            self.mlp = torch.nn.LeakyReLU()
        if aggregate == 'sum':
            self.aggregate_fn = fn.sum
        elif aggregate == 'max':
            self.aggregate_fn = fn.max
        elif aggregate == 'min':
            self.aggregate_fn = fn.min
        elif aggregate == 'mean':
            self.aggregate_fn = fn.mean
        else:
            raise Exception(f'Unsupported aggregation mode {aggregate}')

    def forward(self, feat):
        g = self.dilated_knn_graph(feat)
        g = g.to(feat.device)
        with g.local_scope():
            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))
            g.edata['theta'] = self.theta(g.edata['theta'])
            g.dstdata['phi'] = self.phi(g.dstdata['x'])
            if self.mlp_layer > 0:
                g.apply_edges(lambda edge: {'e': self.mlp(edge.dst['phi'] + edge.data['theta'])})
                g.update_all(fn.copy_e('e', 'm'), self.aggregate_fn('m', 'x'))
            else:
                g.update_all(fn.e_add_v('theta', 'phi', 'e'), self.aggregate_fn('e', 'x'))
            return g.dstdata['x']


class FixedRadiusEdgeConv(torch.nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 mlp_layer=0,
                 aggregate='max',
                 ):
        super(FixedRadiusEdgeConv, self).__init__()

        self.theta = torch.nn.Linear(in_feat, out_feat)
        self.frg = FixedRadiusGraph()
        self.mlp_layer = mlp_layer
        if self.mlp_layer > 1:
            self.mlp = MLP(out_feat, out_feat, hidden_dim=64, hidden_layer=self.mlp_layer,
                           activation='leaky_relu', activation_first=True)
        elif self.mlp_layer == 1:
            self.mlp = torch.nn.LeakyReLU()

        if aggregate == 'sum':
            self.aggregate_fn = fn.sum
        elif aggregate == 'max':
            self.aggregate_fn = fn.max
        elif aggregate == 'min':
            self.aggregate_fn = fn.min
        elif aggregate == 'mean':
            self.aggregate_fn = fn.mean
        else:
            raise Exception(f'Unsupported aggregation mode {aggregate}')

    def forward(self, feat, cutoff, return_g=False, precomputed_g=None):
        if precomputed_g is not None:
            g = precomputed_g
        else:
            g = self.frg(feat, cutoff)

        with g.local_scope():
            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst
            g.apply_edges(fn.v_sub_u('x', 'x', 'theta'))
            if self.mlp_layer > 0:
                g.edata['theta'] = self.theta(g.edata['theta'])
                g.apply_edges(lambda edge: {'e': self.mlp(edge.data['theta'])})
                g.update_all(fn.copy_e('e', 'm'), self.aggregate_fn('m', 'x'))
            else:
                g.edata['theta'] = self.theta(g.edata['theta'])
                g.update_all(fn.copy_e('theta', 'e'), self.aggregate_fn('e', 'x'))
            if return_g:
                return g.dstdata['x'], g
            return g.dstdata['x']


class GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats, act=True, use_layer_norm=False, aggregate='sum'):
        super(GCNLayer, self).__init__()
        if act:
            self.linear = torch.nn.Sequential(torch.nn.Linear(in_feats, out_feats), torch.nn.LeakyReLU())
        else:
            self.linear = torch.nn.Linear(in_feats, out_feats)
        if aggregate == 'sum':
            self.aggregate_fn = fn.sum
        elif aggregate == 'max':
            self.aggregate_fn = fn.max
        elif aggregate == 'min':
            self.aggregate_fn = fn.min
        elif aggregate == 'mean':
            self.aggregate_fn = fn.mean
        else:
            raise Exception(f'Unsupported aggregation mode {aggregate}')

        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(fn.copy_u(u='h', out='m'), self.aggregate_fn(msg='m', out='h'))
            h = g.ndata['h']
        if self.use_layer_norm:
            return self.layer_norm(self.linear(h))
        return self.linear(h)


class GCNInceptionLayer(torch.nn.Module):
    def __init__(self, in_feats,
                 out_feats,
                 act=True,
                 use_layer_norm=False,
                 residual=True):
        super(GCNInceptionLayer, self).__init__()
        self.btn = torch.nn.Linear(in_feats, in_feats//4)
        self.GCN1 = EdgeConv(in_feats//4, in_feats//4, k=20, dilation=1, aggregate='max', mlp_layer=2)
        self.GCN2 = EdgeConv(in_feats//4, in_feats//4, k=20, dilation=2, aggregate='max', mlp_layer=2)
        self.linear = torch.nn.Linear(in_feats//4 * 3, out_feats)
        self.kg = KNNGraph()
        self.act = act
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(out_feats)
        self.residual = residual

    def forward(self, feature):
        skip_connection = feature.clone()
        feature = self.btn(feature)
        g = self.kg(feature, 9)
        g = g.to(feature.device)
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(fn.copy_u(u='h', out='m'), fn.max(msg='m', out='h_max'))
            h_max = g.ndata['h_max']
        h_1 = self.GCN1(feature)
        h_2 = self.GCN2(feature)

        h = torch.cat((h_1, h_2, h_max), dim=1)
        h = self.linear(h)

        if self.use_layer_norm:
            h = self.layer_norm(h)

        if self.residual:
            h += skip_connection
        return h


class MessagePassingLayer(torch.nn.Module):
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 bn_dim,                    # bottleneck dimension
                 hidden_dim=64):

        super(MessagePassingLayer, self).__init__()

        self.edge_affine = torch.nn.Linear(in_edge_feats, bn_dim)
        self.src_affine = torch.nn.Linear(in_node_feats, bn_dim)
        self.dst_affine = torch.nn.Linear(in_node_feats, bn_dim)
        self.theta_edge = MLP(bn_dim, bn_dim, hidden_dim=bn_dim, activation='leaky_relu')

        self.phi_dst = torch.nn.Linear(in_node_feats, hidden_dim)
        self.phi_edge = torch.nn.Linear(bn_dim, hidden_dim)
        self.phi = MLP(hidden_dim, out_node_feats,
                       activation_first=True, hidden_layer=2, hidden_dim=hidden_dim, activation='leaky_relu')

    def forward(self, g, node_feat):
        h_src, h_dst = expand_as_pair(node_feat, g)
        g.srcdata['h'] = h_src
        g.dstdata['h'] = h_dst
        g.apply_edges(lambda edge: {'e_emb': self.theta_edge(self.edge_affine(edge.data['e']) +
                                                             self.src_affine(edge.src['h']) +
                                                             self.dst_affine(edge.dst['h']))})

        g.update_all(fn.copy_e('e_emb', 'm'), fn.sum('m', 'h'))
        message = g.ndata['h']

        node_emb = self.phi(self.phi_dst(node_feat) + self.phi_edge(message))
        return node_emb