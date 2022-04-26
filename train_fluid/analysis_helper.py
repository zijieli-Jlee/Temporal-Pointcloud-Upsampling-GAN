import numpy as np
import torch.nn
from scipy.spatial import KDTree
from chamferdist import ChamferDistance
from pytorch3d.ops import knn_points
from gcn_lib import cubic_interpolation
import numba as nb
from geomloss import SamplesLoss
from torch import nn
from torch.autograd import Function
import emd


class emdFunction(Function):
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

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None



class emdModule(nn.Module):
    def __init__(self):
        super(emdModule, self).__init__()

    def forward(self, input1, input2, eps, iters):
        return emdFunction.apply(input1, input2, eps, iters)


def load_pos(filename):
    with np.load(filename) as dat:
        pos = dat['pos']
    return pos


def write_bgeo_from_numpy(outpath, pos_arr):
    import partio
    n = pos_arr.shape[0]

    p = partio.create()
    position_attr = p.addAttribute("position", partio.VECTOR, 3)

    for i in range(n):
        idx = p.addParticle()
        p.set(position_attr, idx, pos_arr[i].astype(float))

    partio.write(outpath, p)


def numpy_from_bgeo(path):
    import partio
    p = partio.read(path)
    pos = p.attributeInfo('position')
    ida = p.attributeInfo('trackid')  # old format
    if ida is None:
        ida = p.attributeInfo('id')  # new format after splishsplash update
    n = p.numParticles()
    pos_arr = np.empty((n, pos.count))
    for i in range(n):
        pos_arr[i] = p.get(pos, i)

    return pos_arr


@nb.njit
def bicubic_kernel(r, re):
    # coeff = 8. / (np.pi * re**3)
    coeff = 1.
    q = r / re
    if 0 <= q <= 0.5:
        ker = 6. * (q ** 3 - q ** 2) + 1.
    elif 0.5 < q <= 1:
        ker = 2. * (1. - q) ** 3
    else:
        ker = 0.
    return ker * coeff


@nb.njit(parallel=True)
def calc_dns(dns, pos, nbr_lst, cutoff, offset):
    for i in nb.prange(len(nbr_lst)):
        i_nbrs = nbr_lst[i]
        pos_i = pos[i]
        for j in i_nbrs:
            if j == -1:
                break
            j = j + offset
            pos_j = pos[j]
            r = np.sqrt(np.sum((pos_i-pos_j)**2))
            dns[i] += bicubic_kernel(r, cutoff)
    return


def make_2D_array(lis):
    """Funciton to get 2D array from a list of lists
    """
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = -np.ones((n, max_len), dtype=np.int64)
    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr


def get_particle_density(pos, cutoff):
    tree1 = KDTree(pos)
    tree2 = KDTree(pos)
    dns = np.zeros((pos.shape[0], 1))
    indexes = tree1.query_ball_tree(tree2, r=cutoff)
    indexes = make_2D_array(indexes)
    calc_dns(dns, pos, indexes, cutoff, offset=0)
    return dns


def get_particle_density_of_two_pcd(pos_src, pos_dst, cutoff):
    tree1 = KDTree(pos_src)
    tree2 = KDTree(pos_dst)
    dns = np.zeros((pos_src.shape[0], 1))
    indexes = tree1.query_ball_tree(tree2, r=cutoff)
    indexes = make_2D_array(indexes)
    pos = np.concatenate((pos_src, pos_dst), axis=0)
    calc_dns(dns, pos, indexes, cutoff, offset=pos_src.shape[0])
    return dns


def get_1st_derivative(y, dt):
    dy_dt = np.gradient(y, edge_order=dt)
    return dy_dt


def get_2nd_derivative(y, dt):
    dy_dt = get_1st_derivative(y, dt)
    ddy_dt = np.gradient(dy_dt, edge_order=dt)
    return ddy_dt


def cycle_consistency(lowres_pos_left,
                      lowres_pos_right,
                      highres_advection,
                      highres_pos_left,
                      cutoff,
                      sr_net: torch.nn.Module,
                      use_vel=False,
                      lowres_vel_left=None,
                      lowres_vel_right=None):
    """"
    two path:
    1. lowres_pos_left ---sr_net---> highres_pos_pred_left ---advection---> highres_pos_pred_right_1
    2. lowres_pos_right ---sr_net---> highres_pos_pred_right ---advection---> highres_pos_pred_right_2
    """
    # 1st path
    if use_vel:
        feature = torch.cat([lowres_pos_left, lowres_vel_left * 0.025], dim=2)
    else:
        feature = lowres_pos_left
    pred_pos_left, _, _ = sr_net(feature, lowres_pos_left)
    pred_advection = cubic_interpolation(pred_pos_left[0],
                                        highres_advection[0],
                                        highres_pos_left[0],
                                        1.6 * cutoff)
    pred_pos_right_advect = pred_pos_left + pred_advection.unsqueeze(0)

    # 2nd path
    if use_vel:
        feature = torch.cat([lowres_pos_right, lowres_vel_right * 0.025], dim=2)
    else:
        feature = lowres_pos_right
    pred_pos_right, _, _ = sr_net(feature, lowres_pos_right)
    CD = ChamferDistance()
    cd = \
        CD(pred_pos_right, pred_pos_right_advect, bidirectional=True) / pred_pos_right.shape[1]

    m1 = torch.min(pred_pos_right, dim=1, keepdim=True)[0]
    m2 = torch.min(pred_pos_right_advect, dim=1, keepdim=True)[0]
    m_mask = torch.gt(m1, m2)
    m1[m_mask] = m2[m_mask]
    pred_pos_right -= m1
    pred_pos_right_advect -= m1

    h1 = torch.amax(
             torch.sqrt(torch.sum(pred_pos_right ** 2, dim=-1)))
    h2 = torch.amax(
        torch.sqrt(torch.sum(pred_pos_right_advect ** 2, dim=-1)))
    h = max(h1, h2).item()

    emd_dist, _ = emdModule()(pred_pos_right / h, pred_pos_right_advect / h, eps=0.03,
                              iters=3000)
    mmd_module = SamplesLoss(loss='gaussian', scaling=0.8, blur=0.01)
    mmd = mmd_module(pred_pos_right/h, pred_pos_right_advect/h)

    return cd, torch.mean(torch.sqrt(emd_dist)), mmd


def position_loss(
                  masked_pos,
                  pos_pred,
                  pos_gt):

    CD = ChamferDistance()

    cd = \
        CD(pos_pred, pos_gt, bidirectional=True) / (pos_gt.shape[1])
    m1 = torch.min(pos_pred, dim=1, keepdim=True)[0]
    m2 = torch.min(pos_gt, dim=1, keepdim=True)[0]
    m_mask = torch.gt(m1, m2)
    m1[m_mask] = m2[m_mask]
    pos_pred -= m1
    pos_gt -= m1
    h1 = torch.amax(
        torch.sqrt(torch.sum(pos_pred ** 2, dim=-1)), dim=1)
    h2 = torch.amax(
        torch.sqrt(torch.sum(pos_gt ** 2, dim=-1)), dim=1)
    h_mask = torch.gt(h1, h2)
    h1[~h_mask] = h2[~h_mask]
    h = h1.view((-1, 1, 1)).repeat((1, pos_gt.shape[1], 3))

    emd_dist, _ = emdModule()(pos_pred / h, pos_gt / h, eps=0.03, iters=3000)
    mmd_module = SamplesLoss(loss='gaussian', scaling=0.8, blur=0.01)

    # currently this mmd only supports single batch mode
    masked_pos -= m1
    mmd = mmd_module(masked_pos / h1[0], pos_gt / h)

    return cd, torch.mean(torch.sqrt(emd_dist)), torch.mean(mmd)


def knn(k, xyz1, xyz2):
    dist, nbr_idxs, _ = knn_points(
        xyz1, xyz2,
        K=k,
        return_nn=False,
        return_sorted=True
    )
    return dist, nbr_idxs


def free_surface_particle_loss(pos_pred, pos_gt):
    if not isinstance(pos_pred, np.ndarray):
        pos_pred = pos_pred.detach().cpu().numpy()
    free_pos_pred = get_free_surface_particles(pos_pred, 0.025)
    free_pos_gt = get_free_surface_particles(pos_gt, 0.025)
    number_difference = abs(free_pos_pred.shape[0] - free_pos_gt.shape[0])
    return number_difference


def nearest_set(pcd, reference_pcd):
    reference_tree = KDTree(reference_pcd)
    _, set_idx = reference_tree.query(pcd, k=1)
    set_idx, set_size = np.unique(set_idx, return_counts=True)
    return set_idx, set_size


def particle_dns2grid_dns(grid_pos, pcd_pos, cutoff):
    """this calculates the density on each grid point"""
    grid_dns = get_particle_density_of_two_pcd(grid_pos, pcd_pos, cutoff)
    return grid_dns


def eval_spatial_grid_gradient(field, grid):
    if field.shape != grid.shape:
        field.reshape(grid.shape)
    dfield_dx = np.gradient(field, axis=0)
    dfield_dy = np.gradient(field, axis=1)
    dfield_dz = np.gradient(field, axis=2)
    return dfield_dx, dfield_dy, dfield_dz


if __name__ == '__main__':
    import time
    cd_time = []
    emd_time = []
    CD = ChamferDistance()
    EMD_ = emdModule()
    for _ in range(50):
        pos1 = torch.randn((8, 79872, 3)).cuda()
        pos2 = torch.randn((8, 79872, 3)).cuda()

        cd_start = time.time()
        _ = \
            CD(pos1, pos2, bidirectional=True)
        cd_end = time.time()
        cd_used = cd_end - cd_start
        cd_time += [cd_used]

        emd_start = time.time()
        m1 = torch.min(pos1, dim=1)[0]
        m2 = torch.min(pos2, dim=1)[0]
        m_mask = torch.gt(m1, m2)
        m1[m_mask] = m2[m_mask]
        pos1 -= m1.view(-1, 1, 3)
        pos2 -= m1.view(-1, 1, 3)

        h1 = torch.amax(
            torch.sqrt(torch.sum(pos1 ** 2, dim=-1, keepdim=True)), dim=1)
        h2 = torch.amax(
            torch.sqrt(torch.sum(pos2 ** 2, dim=-1, keepdim=True)), dim=1)
        h_mask = torch.gt(h1, h2)
        h2[h_mask] = h1[h_mask]

        _ = \
            EMD_(pos1 / h1.view(-1, 1, 1), pos2 / h2.view(-1, 1, 1), eps=0.05, iters=100)
        emd_end = time.time()
        emd_used = emd_end - emd_start
        emd_time += [emd_used]
    print(np.mean(cd_time))
    print(np.mean(emd_time))


