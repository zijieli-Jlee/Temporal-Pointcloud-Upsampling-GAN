import torch
import numpy as np
from loss import tpugan_sr_loss, chamfer_distance_loss
from train_utils import visualize_pointcloud, dump_pointcloud_visualization
from gcn_lib import cubic_interpolation

DT = 0.025


def get_rotation_matrix():
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.random.uniform(size=3) * 2 * np.pi
    Rx = torch.tensor([[1., 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=torch.float32)
    Ry = torch.tensor([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=torch.float32)
    Rz = torch.tensor([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]], dtype=torch.float32)
    rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))

    return rotation_matrix.cuda()


def advect_particle(pos, vel, sign):
    # can be np.ndarray or torch tensor
    return pos + sign * vel * DT


def rotate_lst(pos_lst, vel_lst=None):
    for i in range(len(pos_lst)):
        r0 = get_rotation_matrix()
        r0 = r0.unsqueeze(0)
        pos_lst[i] = torch.bmm(pos_lst[i], r0.repeat(pos_lst[i].shape[0], 1, 1))
        if vel_lst is not None:
            vel_lst[i] = torch.bmm(vel_lst[i], r0.repeat(pos_lst[i].shape[0], 1, 1))

    if vel_lst is not None:
        return pos_lst, vel_lst
    return pos_lst


def interpolate_vel_lst(pred_pos_lst, gt_pos_lst, gt_vel_lst, opt, furthest_distance):
    gt_adv_lst = []
    pred_adv_lst = []
    for f in range(len(pred_pos_lst)):  # frame
        gt_adv_batch = []
        pred_adv_batch = []
        for b_idx in range(pred_pos_lst[f].shape[0]):   # per batch
            with torch.no_grad():
                highres_adv = gt_vel_lst[f][b_idx] * DT
                pred_adv = cubic_interpolation(pred_pos_lst[f][b_idx], highres_adv, gt_pos_lst[f][b_idx],
                                               1.6 * opt.R / furthest_distance)
            gt_adv_batch += [highres_adv.unsqueeze(0)]
            pred_adv_batch += [pred_adv.unsqueeze(0)]
        gt_adv_lst += [torch.cat(gt_adv_batch, dim=0)]
        pred_adv_lst += [torch.cat(pred_adv_batch, dim=0)]
    return gt_adv_lst, pred_adv_lst


def tempo_gan_step(
                   sr_net: torch.nn.Module,
                   spatial_dis: torch.nn.Module,
                   tempo_dis: torch.nn.Module,
                   lowres_pos_lst,
                   lowres_vel_lst,
                   highres_pos_lst,
                   highres_vel_lst,
                   furthest_distance,
                   opt,
                   n_iter,
                   sr_net_optim,
                   tempo_dis_optim,
                   spatial_dis_optim,
                   freeze_D=False
                    ):
    valid = np.random.uniform(0.8, 1.2)
    invalid = np.random.uniform(0.0, 0.2)
    flip_flag = np.random.uniform(0.0, 1.0)
    # randomly mix-up the label
    if flip_flag < 0.03:
        valid, invalid = invalid, valid

    lowres_pos_batch = lowres_pos_lst[1]

    if not opt.use_vel:
        pred_pos_batch, pred_mask_batch, padded_pred_pos_batch = \
            sr_net(lowres_pos_batch, lowres_pos_batch, hard_masking=True)  # [B, rN, 3]
    else:
        lowres_vel_batch = lowres_vel_lst[1]
        if opt.in_node_feats == 6:
            feature = torch.cat([lowres_pos_batch, lowres_vel_batch*DT], dim=2)
        else:
            feature = lowres_pos_batch
        pred_pos_batch, pred_mask_batch, padded_pred_pos_batch = \
            sr_net(feature, lowres_pos_batch, hard_masking=True)  # [B, rN, 3]

    highres_pos_batch = highres_pos_lst[1]

    position_loss, cd, ml = tpugan_sr_loss(100.,
                                          highres_pos_batch,
                                          pred_pos_batch,
                                          lowres_pos_batch,
                                          pred_mask_batch,
                                          opt.cutoff / furthest_distance, n_iter)
    del pred_pos_batch, pred_mask_batch

    # pred fake
    if ml < 0.1:
        # ====================
        # spatial gan
        fake_label = spatial_dis(padded_pred_pos_batch[:, torch.randperm(padded_pred_pos_batch.shape[1])])
        spatial_loss = 0.5 * (fake_label - np.random.uniform(0.8, 1.2)) ** 2
        spatial_loss = spatial_loss.mean()

        # sequentially upsample
        # upsample particles on left and right
        pred_pos_lst = [0] * len(highres_pos_lst)
        pred_pos_lst[1] = padded_pred_pos_batch
        for frame in [0] + list(range(2, len(highres_pos_lst))):
            lowres_pos_batch = lowres_pos_lst[frame]
            if not opt.use_vel:
                _, _, padded_pred_pos_batch = sr_net(lowres_pos_batch, lowres_pos_batch, hard_masking=True)
            else:
                lowres_vel_batch = lowres_vel_lst[frame]
                if opt.in_node_feats == 6:
                    feature = torch.cat([lowres_pos_batch, lowres_vel_batch * DT], dim=2)
                else:
                    feature = lowres_pos_batch
                _, _, padded_pred_pos_batch = sr_net(feature,
                                                     lowres_pos_batch, hard_masking=True)
            pred_pos_lst[frame] = padded_pred_pos_batch[:, torch.randperm(padded_pred_pos_batch.shape[1])]

        # ======================
        # tempo dis
        if not opt.use_vel:
            fake_label = tempo_dis(pred_pos_lst, opt.R)
        else:
            gt_adv_lst, pred_adv_lst = interpolate_vel_lst(pred_pos_lst, highres_pos_lst, highres_vel_lst,
                                                           opt, furthest_distance)

            fake_label = tempo_dis(pred_pos_lst, opt.R, feat_lst=pred_adv_lst)

        tempo_loss = 0.5 * (fake_label - np.random.uniform(0.8, 1.2)) ** 2
        tempo_loss = tempo_loss.mean()
    else:
        # placeholder
        spatial_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        tempo_loss = torch.tensor([0.], dtype=torch.float32).cuda()

    sr_loss = tempo_loss + spatial_loss + opt.w * position_loss

    sr_net_optim.zero_grad()
    sr_loss.backward()
    sr_net_optim.step()

    # pred true
    if n_iter % 2 == 0 and not freeze_D and ml < 0.1:
        # temporal discriminator update
        pred_pos_lst = [pred_pos.detach() for pred_pos in pred_pos_lst]

        prob = np.random.uniform()

        if not opt.use_vel:  # not use velocity
            if prob > 0.7:
                pred_pos_lst = rotate_lst(pred_pos_lst)
                highres_pos_lst = rotate_lst(highres_pos_lst)
            fake_label = tempo_dis(pred_pos_lst, opt.R)
            true_label = tempo_dis(highres_pos_lst, opt.R)
        else:  # use velocity
            if prob > 0.7:
                pred_pos_lst_detach, pred_adv_lst = rotate_lst(pred_pos_lst, pred_adv_lst)
                highres_pos_lst, gt_adv_lst = rotate_lst(highres_pos_lst, gt_adv_lst)
            fake_label = tempo_dis(pred_pos_lst, opt.R, feat_lst=pred_adv_lst)
            true_label = tempo_dis(highres_pos_lst, opt.R, feat_lst=gt_adv_lst)

        tempo_dis_loss = 0.5 * ((true_label - valid) ** 2 + (fake_label - invalid) ** 2)
        tempo_dis_loss = tempo_dis_loss.mean()

        tempo_dis_optim.zero_grad()
        tempo_dis_loss.backward()
        tempo_dis_optim.step()

        # spatial discriminator update
        prob = np.random.uniform()
        if prob > 0.7:
            R0 = []
            for b in range(highres_pos_batch.shape[0]):
                r0 = get_rotation_matrix()
                R0 += [r0.unsqueeze(0)]
            R0 = torch.cat(R0, dim=0)
            highres_pos_batch = torch.bmm(highres_pos_batch, R0)

            R1 = []
            for b in range(highres_pos_batch.shape[0]):
                r1 = get_rotation_matrix()
                R1 += [r1.unsqueeze(0)]
            R1 = torch.cat(R1, dim=0)
            padded_pred_pos_batch = torch.bmm(padded_pred_pos_batch, R1)

        fake_label = spatial_dis(padded_pred_pos_batch.detach())
        true_label = spatial_dis(highres_pos_batch)
        spatial_dis_loss = 0.5 * ((true_label - valid) ** 2 + (fake_label - invalid) ** 2)
        spatial_dis_loss = spatial_dis_loss.mean()

        spatial_dis_optim.zero_grad()
        spatial_dis_loss.backward()
        spatial_dis_optim.step()
    else:
        # just put some placeholder, do nothing
        tempo_dis_loss = torch.tensor([0.], dtype=torch.float32)
        spatial_dis_loss = torch.tensor([0.], dtype=torch.float32)

    return {
                'tempo_G_loss': tempo_loss.item(),
                'tempo_D_loss': tempo_dis_loss.item(),
                'Chamfer_distance_no_norm': cd.item(),   # not divided by the num of points
                'masking_loss': ml.item(),
                'spatial_G_loss': spatial_loss.item(),
                'spatial_D_loss': spatial_dis_loss.item(),

    }


def tempo_gan_step_no_mask(
                           sr_net: torch.nn.Module,
                           spatial_dis: torch.nn.Module,
                           tempo_dis: torch.nn.Module,
                           lowres_pos_lst,
                           highres_pos_lst,
                           opt,
                           n_iter,
                           sr_net_optim,
                           tempo_dis_optim,
                           spatial_dis_optim,
                           freeze_D=False
                        ):
    valid = np.random.uniform(0.8, 1.2)
    invalid = np.random.uniform(0.0, 0.2)
    flip_flag = np.random.uniform(0.0, 1.0)
    # randomly mix-up the label
    if flip_flag < 0.03:
        valid, invalid = invalid, valid

    lowres_pos_batch = lowres_pos_lst[1]
    pred_pos_batch, _ = sr_net(lowres_pos_batch, lowres_pos_batch)  # [B, rN, 3]
    highres_pos_batch = highres_pos_lst[1]

    fake_label = spatial_dis(pred_pos_batch[:, torch.randperm(pred_pos_batch.shape[1])])
    spatial_loss = 0.5*(fake_label - np.random.uniform(0.8, 1.2))**2
    spatial_loss = spatial_loss.mean()

    position_loss, cd, _ = tpugan_sr_loss(0,
                                         highres_pos_batch,
                                         pred_pos_batch,
                                         0., 0.,
                                         0., 0)

    # pred fake
    # upsample particles on left and right
    pred_pos_lst = [0] * len(highres_pos_lst)
    pred_pos_lst[1] = pred_pos_batch[:, torch.randperm(pred_pos_batch.shape[1])]
    for frame in [0] + list(range(2, len(highres_pos_lst))):
        lowres_pos_batch = lowres_pos_lst[frame]
        pred_pos_b, _ = sr_net(lowres_pos_batch, lowres_pos_batch)
        pred_pos_lst[frame] = pred_pos_b[:, torch.randperm(pred_pos_b.shape[1])]
    fake_label = tempo_dis(pred_pos_lst, opt.R)
    tempo_loss = 0.5 * (fake_label - np.random.uniform(0.8, 1.2))**2
    tempo_loss = tempo_loss.mean()

    sr_loss = tempo_loss + spatial_loss + opt.w * position_loss

    sr_net_optim.zero_grad()
    sr_loss.backward()
    sr_net_optim.step()

    # pred true
    if n_iter % 2 == 0 and not freeze_D:

        pred_pos_lst = [pred_pos.detach() for pred_pos in pred_pos_lst]
        fake_label = tempo_dis(pred_pos_lst, opt.R)
        true_label = tempo_dis(highres_pos_lst, opt.R)

        tempo_dis_loss = 0.5 * ((true_label - valid) ** 2 + (fake_label - invalid) ** 2)
        tempo_dis_loss = tempo_dis_loss.mean()

        tempo_dis_optim.zero_grad()
        tempo_dis_loss.backward()
        tempo_dis_optim.step()

        fake_label = spatial_dis(pred_pos_batch[:, torch.randperm(pred_pos_batch.shape[1])].detach())
        true_label = spatial_dis(highres_pos_batch)

        spatial_dis_loss = 0.5 * ((true_label - valid) ** 2 + (fake_label - invalid) ** 2)
        spatial_dis_loss = spatial_dis_loss.mean()

        spatial_dis_optim.zero_grad()
        spatial_dis_loss.backward()
        spatial_dis_optim.step()
    else:
        # just put some placeholder, do nothing
        tempo_dis_loss = torch.tensor([0.], dtype=torch.float32)
        spatial_dis_loss = torch.tensor([0.], dtype=torch.float32)

    return {
                'tempo_G_loss': tempo_loss.item(),
                'tempo_D_loss': tempo_dis_loss.item(),
                'Chamfer_distance_no_norm': cd.item(),
                'spatial_G_loss': spatial_loss.item(),
                'spatial_D_loss': spatial_dis_loss.item(),

    }
