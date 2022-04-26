import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.utils import expand_as_pair
import dgl.function as fn
import frnn
from pytorch3d.ops import knn_points


def l2dist(pos_src, pos_dst):
    dist = torch.sum(pos_src ** 2 + pos_dst ** 2 - 2 * pos_dst * pos_src, dim=1, keepdim=True)
    dist[dist < 1e-8] = 0.  # to avoid numerical instability
    return torch.sqrt(dist)

def get_local_neighbor_graph(query_pos, candidate_pos, cutoff,
                             filter_out_of_range=False,
                             knn_padding=False):
    if filter_out_of_range:
        _, nbr_idxs, _, _ = frnn.frnn_grid_points(
            query_pos[None, ...], candidate_pos[None, ...],
            K=32,
            r=cutoff,
            grid=None, return_nn=False, return_sorted=True
        )
        nbr_idx = nbr_idxs.squeeze(0)
        mask = nbr_idx != -1
        nbr_idx = nbr_idx[mask]
        in_range_idx = torch.unique(nbr_idx)

        candidate_pos = candidate_pos[in_range_idx]

    _, nbr_idxs, _, _ = frnn.frnn_grid_points(
        query_pos[None, ...], candidate_pos[None, ...],
        K=32,
        r=cutoff,
        grid=None, return_nn=False, return_sorted=True
    )
    nbr_idx = nbr_idxs.squeeze(0)
    center_idx = nbr_idx.clone()
    center_idx[:] = torch.arange(query_pos.shape[0]).to(query_pos.device).reshape(-1, 1)
    mask = nbr_idx != -1

    if knn_padding and torch.unique(center_idx[mask]).shape[0] < query_pos.shape[0]:

        to_be_pad_idx = torch.unique(center_idx[~mask])
        _, pad_nbr_idxs, _ = knn_points(
            query_pos[to_be_pad_idx][None, ...], candidate_pos[None, ...],
            K=4,
            return_nn=False,
            return_sorted=True
        )
        pad_nbr_idx = pad_nbr_idxs.squeeze(0)
        pad_center_idx = pad_nbr_idx.clone()
        pad_center_idx[:] = to_be_pad_idx.reshape(-1, 1)

        nbr_idx = nbr_idx[mask]
        center_idx = center_idx[mask]
        graph = dgl.graph((nbr_idx, center_idx + candidate_pos.shape[0]))
        graph.add_edges(pad_nbr_idx.view(-1), pad_center_idx.view(-1) + candidate_pos.shape[0])
    else:
        nbr_idx = nbr_idx[mask]
        center_idx = center_idx[mask]
        #
        # if torch.unique(center_idx).shape[0] < query_pos.shape[0]:
        #     dump_pointcloud_visualization(query_pos.detach().cpu().numpy(), './err_input.png')
        #     dump_pointcloud_visualization(candidate_pos.detach().cpu().numpy(), './err_output.png')
        #     raise Exception
        graph = dgl.graph((nbr_idx, center_idx + candidate_pos.shape[0]))

    # note that:
    # nbr_idx is used to index candidate pos
    # center idx is used to index query pos
    # edge direction (message flow direction) : nbr_idx --> center_idx
    # there is an offset of center idx
    # because we will input the node feature as torch.cat([encoded_feature, query_feature])
    if filter_out_of_range:
        return graph, in_range_idx
    else:
        return graph


def exponential_kernel(r, cutoff):
    coeff = 1./np.sqrt(np.pi**3) * cutoff**3
    return coeff * torch.exp(-(r/cutoff)**2)


def linear_kernel(r, cutoff):
    return torch.nn.ReLU()(1. - r/cutoff)


def bicubic_kernel(r, cutoff):
    coeff = 8. / (np.pi * cutoff**3)
    ker = torch.zeros_like(r)
    q = r / cutoff
    mask1 = torch.logical_and(q >= 0, q <= 0.5)
    ker[mask1] = (6. * (q**3 - q**2) + 1.)[mask1]
    mask2 = torch.logical_and(q > 0.5, q <= 1)
    ker[mask2] = (2. * (1. - q)**3)[mask2]
    return ker * coeff


def cubic_interpolation(query_pos, field, pos, cutoff):
    query_graph, in_range_idx = get_local_neighbor_graph(query_pos, pos, cutoff, filter_out_of_range=True, knn_padding=True)
    pos = pos[in_range_idx]
    with query_graph.local_scope():
        edge_idxs = query_graph.edges()
        src_idx = edge_idxs[0]                           # encoded pos index
        dst_idx = edge_idxs[1] - pos.shape[0]    # query pos index
        query_latent_repr = torch.zeros((query_pos.shape[0], field.shape[1]),
                                         dtype=torch.float32).to(query_pos.device)
        all_latent_repr = torch.cat((field[in_range_idx], query_latent_repr), dim=0)
        h_src, h_dst = expand_as_pair(all_latent_repr, query_graph)
        query_graph.srcdata['h'] = h_src
        query_graph.dstdata['h'] = h_dst
        E = l2dist(pos[src_idx], query_pos[dst_idx])

        query_graph.edata['e'] = E
        query_graph.apply_edges(lambda edge: {'w': bicubic_kernel(edge.data['e'], cutoff)})
        query_graph.update_all(fn.src_mul_edge('h', 'w', 'm'), fn.sum('m', 'h'))
        query_graph.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'k'))
        all_latent_repr = query_graph.ndata['h'] / (query_graph.ndata['k'] + 1e-6)
    return all_latent_repr[pos.shape[0]:]
