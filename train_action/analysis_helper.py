import os
import sys
import numpy as np
import torch
from pytorch3d.ops import knn_points

from chamferdist import ChamferDistance
import emd


class emdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert(n == m)
        assert(xyz1.size()[0] == xyz2.size()[0])
        assert(n % 1024 == 0)
        assert(batchsize <= 512)

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device='cuda').contiguous()
        assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
        assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
        price = torch.zeros(batchsize, m, device='cuda').contiguous()
        bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
        max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
        unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
        max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()

        emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment


class emdModule(torch.nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def position_loss(pos_pred,
                  pos_gt):

    CD = ChamferDistance()
    cd = \
        CD(pos_pred, pos_gt, bidirectional=True) / 2048

    emd_dist, _ = emdModule()(pos_pred/2., pos_gt/2., eps=0.002, iters=3000)   # emd requires distance < 3.
    return cd, torch.mean(torch.sqrt(emd_dist))*2.


def pad_with_appropriate_size(pos_lst, num_points=2048):
    clip = []
    for i in range(len(pos_lst)):
        p = pos_lst[i].copy()
        if p.shape[0] > num_points:
            r = np.random.choice(p.shape[0], size=num_points, replace=False)
        else:
            repeat, residue = num_points // p.shape[0], num_points % p.shape[0]
            r = np.random.choice(p.shape[0], size=residue, replace=False)
            r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
        p[:, 1] = -p[:, 1]
        clip.append(pc_normalize(p[r, :].astype(np.float32))[None, ...])
    return np.concatenate(clip, axis=0)


def knn(k, xyz1, xyz2):
    dist, nbr_idxs, _ = knn_points(
        xyz1, xyz2,
        K=k,
        return_nn=False,
        return_sorted=True
    )
    return dist, nbr_idxs