import torch
import numpy as np
from chamferdist import ChamferDistance
import frnn
import emd
from torch import nn
from torch.autograd import Function


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

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


def differentiable_nbr_distance(nbr_idx, pos: torch.Tensor):
    center_idx = nbr_idx.clone()
    center_idx[:] = torch.arange(pos.shape[0]).reshape(-1, 1)
    distance2 = pos[nbr_idx]**2 + pos[center_idx]**2 - 2*pos[center_idx]*pos[nbr_idx]
    distance2[distance2 < 1e-9] = 0.0
    distance2 = torch.sum(distance2, dim=-1)
    distance = torch.sqrt(distance2)

    return distance


def differentiable_distance(pos1, pos2):
    distance2 = pos1**2 + pos2**2 - 2*pos1*pos2
    distance2[distance2 < 1e-8] = 0.0
    distance2 = torch.sum(distance2, dim=-1)
    distance = torch.sqrt(distance2)
    return distance


def density(pcd_pos: torch.Tensor, h):
    """
    calculate the density of point cloud give window h
    """
    cutoff_radius = 2.1*h
    _, nbr_idx, _, _ = frnn.frnn_grid_points(
                                    pcd_pos[None, ...], pcd_pos[None, ...],
                                    K=32,
                                    r=cutoff_radius,
                                    grid=None, return_nn=False, return_sorted=True
                                )

    nbr_idx = nbr_idx.squeeze(0)
    distance = differentiable_nbr_distance(nbr_idx, pcd_pos)
    distance_ = distance.clone()
    mask = torch.logical_or(nbr_idx == -1, distance < 1e-8)
    distance_[mask] = cutoff_radius
    density = torch.sum(torch.nn.ReLU()(cutoff_radius / distance_ - 1.), dim=1, keepdim=True)
    return density


def chamfer_distance_loss(pcd1_pos: torch.Tensor, pcd2_pos: torch.Tensor):
    """
    calculate the chamfer distance from pcd1 to pcd2
    """
    cd = ChamferDistance()

    dist = cd(pcd1_pos[None,...], pcd2_pos[None, ...], bidirectional=True)
    return dist


def dense_loss(pred_prob, h, furthest_distance):
    """
    calculate how dense
    """
    h = h/furthest_distance
    return torch.mean(torch.sum(torch.abs(pred_prob), dim=1)) / h


def repulsion_loss(pred_pos, h, furthest_distance):

    h = h / furthest_distance
    _, nbr_idx, _, _ = frnn.frnn_grid_points(
                                        pred_pos[None, ...], pred_pos[None, ...],
                                        K=8,
                                        r=1.1*h,
                                        grid=None, return_nn=False, return_sorted=True
                                                     )
    nbr_idx = nbr_idx.squeeze(0)
    distance = differentiable_nbr_distance(nbr_idx, pred_pos)
    mask = torch.logical_or(nbr_idx == -1, distance < 1e-9)
    smeared_distance = (torch.clamp(distance, max=3.1*h) - h)**2 / (h**2)
    smeared_distance_ = smeared_distance.clone()
    smeared_distance_[mask] = 0.
    rep_loss = torch.mean(torch.sum(smeared_distance_, dim=1))
    return rep_loss

def edge_uniform_loss(edge, cutoff):
    edge_norm2 = torch.sum(edge**2, dim=-1)
    edge_target = 4*cutoff + 1e-6
    mask = edge_norm2 > (edge_target**2)
    if torch.sum(mask) == 0:
        edge_loss = torch.zeros((1,), dtype=torch.float32).to(edge.device)
    else:
        edge_loss = torch.mean((edge_norm2[mask] - edge_target**2) / edge_target**2)
    return edge_loss


def tpugan_sr_loss(w1, gt_pcd_pos, pred_pcd_pos, input_pcd_pos, mask,
                particle_radius, n_iter):

    if n_iter > 10 and w1 != 0:
        m_loss = masking_loss(gt_pcd_pos, input_pcd_pos, mask, particle_radius)
    else:    #   w1 = 0 means masking loss is not applied
        m_loss = torch.tensor([1.0]).cuda()

    cd = ChamferDistance()
    if len(gt_pcd_pos.shape) == 2:
        gt_pcd_pos = gt_pcd_pos.unsqueeze(0)
    if len(pred_pcd_pos.shape) == 2:
        pred_pcd_pos = pred_pcd_pos.unsqueeze(0)
    cd_dist = cd(gt_pcd_pos, pred_pcd_pos, bidirectional=True)
    loss = cd_dist + w1 * m_loss
    return loss, cd_dist, m_loss


def sr_loss(gt_pcd_pos, pred_pcd_pos):
    cd = chamfer_distance_loss(gt_pcd_pos, pred_pcd_pos)
    loss = cd
    return loss, cd


# def free_particle_loss(pos_gt, pos_pred, particle_radius):
#     with torch.no_grad():
#         dns_gt = density(pos_gt, particle_radius)
#         dns_gt_threshold = 0.90 * \
#                            torch.mean(
#                                torch.sort(dns_gt, dim=0)[0][-int(0.10*pos_gt.shape[0]):-int(0.01*pos_gt.shape[0])])
#         dns_gt_mask = dns_gt < dns_gt_threshold
#         free_pcd_gt = pos_gt[dns_gt_mask.view(-1)]
#     # find gt's free particles' nearest point in pred
#     _, nbr_idx, _, _ = frnn.frnn_grid_points( free_pcd_gt[None, ...], pos_pred[None, ...],
#                                               K=16,
#                                               r=particle_radius*3.1,
#                                               grid=None, return_nn=False, return_sorted=True
#                                              )
#     nbr_idx = nbr_idx[nbr_idx != -1].view(-1)
#     free_pcd_pred = pos_pred[nbr_idx]
#     # visualize_pointcloud(pos_gt.detach().cpu().numpy())
#     # visualize_pointcloud(free_pcd_gt.detach().cpu().numpy())
#     # visualize_pointcloud(pos_pred.detach().cpu().numpy())
#     # visualize_pointcloud(free_pcd_pred.detach().cpu().numpy())
#     cd = ChamferDistance()
#     return cd(free_pcd_gt.unsqueeze(0), free_pcd_pred.unsqueeze(0),
#               bidirectional=True)


def free_particle_loss(free_pcd_gt, pos_pred, particle_radius):
    # find gt's free particles' nearest point in pred

    # visualize_pointcloud(pos_gt.detach().cpu().numpy())
    # visualize_pointcloud(free_pcd_gt.detach().cpu().numpy())
    # visualize_pointcloud(pos_pred.detach().cpu().numpy())
    # visualize_pointcloud(free_pcd_pred.detach().cpu().numpy())
    cd = ChamferDistance()
    return cd(free_pcd_gt.unsqueeze(0), pos_pred.unsqueeze(0), bidirectional=True)


def density_loss(pred_pos, particle_radius):
    _, nbr_idx, _, _ = frnn.frnn_grid_points(
                                        pred_pos[None, ...], pred_pos[None, ...],
                                        K=8,
                                        r=1.5*particle_radius,
                                        grid=None, return_nn=False, return_sorted=True
                                                     )
    nbr_idx = nbr_idx.squeeze(0)
    distance = differentiable_nbr_distance(nbr_idx, pred_pos)
    mask = torch.logical_or(nbr_idx == -1, distance < 1e-4)
    smeared_distance = (distance - particle_radius)**2 / (particle_radius**2)
    smeared_distance_ = smeared_distance.clone()
    del smeared_distance
    smeared_distance_[mask] = 0.
    rep_loss = torch.mean(torch.sum(smeared_distance_, dim=1))
    return rep_loss


def refinement_loss(w, pos_gt, pos_pred, particle_radius):
    free_loss = free_particle_loss(pos_gt, pos_pred, particle_radius)
    dns_loss = density_loss(pos_pred, particle_radius)
    loss = free_loss + w*dns_loss
    return loss, free_loss, dns_loss


def masking_loss(pos_gt, pos_input, binary_mask, particle_radius):
    # binary mask [b, N]
    # find pred's nearest point in gt
    _, nbr_idx, _, _ = frnn.frnn_grid_points(pos_input, pos_gt,
                                             K=1,
                                             r=particle_radius*1.9,
                                             grid=None, return_nn=False, return_sorted=True
                                             )
    _, self_nbr_idx, _, _ = frnn.frnn_grid_points(pos_gt, pos_gt,
                                                 K=16,
                                                 r=particle_radius * 1.4,
                                                 grid=None, return_nn=False, return_sorted=True
                                                 )
    # nbr_idx: [batch_size, input_size, 1], self_nbr_idx: [batch_size, gt_size, 16]
    nbr_count = torch.sum((self_nbr_idx != -1), dim=-1)  # [b, gt.shape[1]   (gt_size)]
    mask = nbr_count > 3
    nbr_count[mask] = 1.
    nbr_count[~mask] = 0.
    # since some input point may not find neighbor in ground truth
    # their nbr_idx will be -1, we need to pad for the -1 index
    nbr_count = torch.cat((nbr_count, torch.zeros((nbr_count.shape[0], 1), device=pos_input.device)), dim=1)
    return torch.nn.L1Loss()(binary_mask,
                             index_points(nbr_count.unsqueeze(-1), nbr_idx.squeeze(-1)))


def temporal_loss(advect_particle_right, advect_particle_left,
                  upsample_particle_right, upsample_particle_left):
    cd = ChamferDistance()
    dist1 = cd(advect_particle_left.unsqueeze(0), upsample_particle_left.unsqueeze(0), bidirectional=True)
    dist2 = cd(advect_particle_right.unsqueeze(0), upsample_particle_right.unsqueeze(0), bidirectional=True)
    return 0.5 * dist1 + 0.5 * dist2


def tempo_discriminator_loss(pred_label_true, pred_label_fake):
    return (pred_label_true - 1.)**2 + pred_label_fake**2


def tempo_generator_loss(pred_label_fake):
    return (pred_label_fake - 1.)**2


def earth_mover_distance_loss(pred, target):
    m1 = torch.min(pred, dim=0)[0]
    m2 = torch.min(target, dim=0)[0]
    m_mask = torch.gt(m1, m2)
    m1[m_mask] = m2[m_mask]
    pred -= m1
    target -= m1

    h1 = torch.amax(
        torch.sqrt(torch.sum(pred ** 2, dim=-1, keepdim=True)), dim=0)
    h2 = torch.amax(
        torch.sqrt(torch.sum(target ** 2, dim=-1, keepdim=True)), dim=0)
    h = max(h1, h2)
    N = min(pred.shape[0], target.shape[0])
    idx = np.random.choice(N,
                           N // 1024 * 1024, replace=False)
    with torch.no_grad():
        _, assignment = emdModule()(pred[None, idx] / h, target[None, idx] / h, eps=0.05, iters=2000)
    emd_dist = torch.sum((pred[idx] - target[idx][assignment[0].long()])**2, dim=-1)
    emd_loss = torch.sum(torch.sqrt(emd_dist))
    if emd_loss.item() is np.nan:
        emd_loss = torch.tensor([0.]).to(pred.device)
    return emd_loss






