import torch
import dgl
from pytorch3d.ops import knn_points
import frnn


def l2dist(pos_src, pos_dst):
    dist = torch.sum(pos_src ** 2 + pos_dst ** 2 - 2 * pos_dst * pos_src, dim=1, keepdim=True)
    dist[dist < 1e-8] = 0.  # to avoid numerical instability
    return torch.sqrt(dist)


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

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index.view(2, -1, num)
                edge_index = edge_index[:, :, randnum]
                return edge_index.view(2, -1)
            else:
                edge_index = edge_index[:, ::self.dilation]
        else:
            edge_index = edge_index[:, ::self.dilation]
        return edge_index


class FixedRadiusGraph(torch.nn.Module):
    def __init__(self):
        super(FixedRadiusGraph, self).__init__()

    @staticmethod
    @torch.no_grad()
    def build_fixed_radius_graph(pos_tsr: torch.Tensor, cutoff_radius):
        _, nbr_idxs, _, _ = frnn.frnn_grid_points(
            pos_tsr[None, ...], pos_tsr[None, ...],
            K=16,
            r=cutoff_radius,
            grid=None, return_nn=False, return_sorted=True
        )
        nbr_idx = nbr_idxs.squeeze(0)
        center_idx = nbr_idx.clone()
        center_idx[:] = torch.arange(pos_tsr.shape[0]).to(pos_tsr.device).reshape(-1, 1)
        mask = nbr_idx != -1
        nbr_idx = nbr_idx[mask]
        center_idx = center_idx[mask]
        graph = dgl.graph((nbr_idx, center_idx))
        return graph

    def forward(self, pos: torch.Tensor, cutoff) -> dgl.graph:
        return self.build_fixed_radius_graph(pos, cutoff)


class KNNGraph(torch.nn.Module):
    def __init__(self):
        super(KNNGraph, self).__init__()

    @torch.no_grad()
    def build_knn_graph(self, pos_tsr: torch.Tensor, k):
        _, nbr_idxs, _ = knn_points(
            pos_tsr[None, ...], pos_tsr[None, ...],
            K=k,
            return_nn=False,
            return_sorted=True
        )
        nbr_idx = nbr_idxs.squeeze(0)
        center_idx = nbr_idx.clone()
        center_idx[:] = torch.arange(pos_tsr.shape[0]).to(pos_tsr.device).reshape(-1, 1)
        mask = nbr_idx != -1
        nbr_idx = nbr_idx[mask]
        center_idx = center_idx[mask]
        graph = dgl.graph((nbr_idx, center_idx))
        return graph

    def forward(self, pos: torch.Tensor, k) -> dgl.graph:
        return self.build_knn_graph(pos, k)


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
        self.knn = KNNGraph()

    def forward(self, x):
        g = self.knn(x, self.k * self.dilation)
        edge_index = torch.cat([idx.view(1, -1) for idx in g.edges()], dim=0)
        edge_index = self._dilated(edge_index)
        g = dgl.graph((edge_index[0, :], edge_index[1, :]))
        return g
