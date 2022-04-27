import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.nn.utils import spectral_norm as sp_norm
import numpy as np
from pointnet2_ops.pointnet2_utils import grouping_operation
from pointnet2_ops import pointnet2_utils
from pytorch3d.ops import knn_points
import frnn


def knn(k, xyz1, xyz2):
    # wrapper for pytorch3d's knn api
    dist, nbr_idxs, _ = knn_points(
        xyz1, xyz2,
        K=k,
        return_nn=False,
        return_sorted=True
    )
    return dist, nbr_idxs


def ball_query_wrapper(radius, sample, xyz1, xyz2):
    # If the ball neighborhood points are less than nsample,
    # than use the knn neighborhood points
    _, nbr_idxs, _, _ = frnn.frnn_grid_points(
        xyz1, xyz2,
        K=sample,
        r=radius,
        grid=None, return_nn=False, return_sorted=True
    )
    _, knn_nbr_idxs, _ = knn_points(
        xyz1, xyz2,
        K=sample,
        return_nn=False,
        return_sorted=True
    )
    nbr_idxs[nbr_idxs == -1] = knn_nbr_idxs[nbr_idxs == -1]
    return nbr_idxs


def index_points(points, idx):
    """
    (Code copied from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
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


def build_shared_mlp(mlp_spec: List[int], bn: bool = True, sn: bool = True, act_fn=nn.ReLU(True)):
    layers = []
    for i in range(1, len(mlp_spec)):
        if sn:
            layers.append(
                sp_norm(nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn))
            )
        else:
            layers.append(
                nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
            )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        # layers.append(nn.ReLU(True))   # potential improvement may be using other activation
        layers.append(act_fn)
    return nn.Sequential(*layers)


# Set abstraction layer modified from: https://github.com/erikwijmans/Pointnet2_PyTorch
class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.mask_dummy = False


    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.npoint is not None:
            # drop the points containing dummy points [999, 999, 999], dummy mask
            fps_center = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            if self.mask_dummy:
                mask = torch.abs(index_points(xyz, fps_center.long())[:, :, 0] - 999) < 1e-4  # [b ,n_point]
                if torch.any(mask):
                    for b in range(xyz.shape[0]):
                        dummy_idx = mask[b].nonzero().view(-1)

                        # sample fps center randomly from those not in dummy index
                        if dummy_idx.shape[0] > 0:
                            combined = torch.cat((torch.arange(xyz.shape[1], device=xyz.device), dummy_idx))
                            uniques, counts = combined.unique(return_counts=True)
                            difference = uniques[counts == 1]
                            random_center = difference[np.random.choice(np.arange(difference.shape[0]),
                                                             dummy_idx.shape[0], replace=False)]
                            fps_center[b] = torch.cat((fps_center[b][~mask[b]],
                                                       random_center))
                del mask
            new_xyz = (
                pointnet2_utils.gather_operation(
                    xyz_flipped, fps_center
                )
                .transpose(1, 2)
                .contiguous()
            )
        else:
            new_xyz = None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class MSGSetConv(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    sn : bool
        Use SpectralNorm
    """

    def __init__(self, npoint, radii, nsamples, mlps,
                 act_fn=None,
                 mask_dummy=False, bn=True, use_xyz=True, sn=True):
        # type: (SetConv, int, List[float], List[int], List[List[int]], nn.Module, bool, bool, bool, bool) -> None
        super(MSGSetConv, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.mask_dummy = mask_dummy
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if act_fn is not None:
                self.mlps.append(build_shared_mlp(mlp_spec, bn, sn, act_fn=act_fn))
            else:
                self.mlps.append(build_shared_mlp(mlp_spec, bn, sn))


class SSGSetConv(MSGSetConv):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, mask_dummy=None, radius=None, nsample=None, bn=True, use_xyz=True, sn=True, act_fn=None,
    ):
        super(SSGSetConv, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            mask_dummy=mask_dummy,
            bn=bn,
            use_xyz=use_xyz,
            sn=sn,
            act_fn=act_fn,
        )


class FlowEmbedding(nn.Module):
    def __init__(self, in_channel, mlp, pooling='max', corr_func='concat', sn=False):
        super(FlowEmbedding, self).__init__()
        self.pooling = pooling
        self.corr_func = corr_func
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if corr_func is 'concat':
            last_channel = in_channel * 2 + 3
        for out_channel in mlp:
            if sn:
                self.mlp_convs.append(sp_norm(nn.Conv2d(last_channel, out_channel, 1, bias=False)))
            else:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2, radius):
        """
        Input:
            xyz1: (batch_size, 3, npoint)
            xyz2: (batch_size, 3, npoint)
            feat1: (batch_size, channel, npoint)
            feat2: (batch_size, channel, npoint)
        Output:
            xyz1: (batch_size, 3, npoint)
            feat1_new: (batch_size, mlp[-1], npoint)
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, N, C = pos1_t.shape

        idx = ball_query_wrapper(radius, 32, pos1_t, pos2_t)  # idx is use to index pos2
        idx = idx.type(torch.int32).contiguous()

        pos2_grouped = grouping_operation(pos2, idx)  # [B, 3, N, S]
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B, 3, N, S]

        feat2_grouped = grouping_operation(feature2, idx)  # [B, C, N, S]
        if self.corr_func == 'concat':
            feat_diff = torch.cat([feat2_grouped, feature1.view(B, -1, N, 1).repeat(1, 1, 1, 32)], dim=1)

        feat1_new = torch.cat([pos_diff, feat_diff], dim=1)  # [B, 2*C+3,N,S]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat1_new = F.leaky_relu(bn(conv(feat1_new)))

        feat1_new = torch.max(feat1_new, -1)[0]  # [B, mlp[-1], npoint]
        return pos1, feat1_new


class FlowModule(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, sequence_length, sn=False):
        super(FlowModule, self).__init__()
        self.flow_emb_layers = nn.ModuleList()
        if sequence_length < 1:
            raise Exception('Flow module only accepts sequence with length greater than 1')

        self.depth = sequence_length - 1
        if self.depth == 1:
            hidden_feat = out_feat
        for depth in range(sequence_length-1):
            if depth == 0:
                self.flow_emb_layers.append(FlowEmbedding(in_feat, [in_feat, hidden_feat // 2, hidden_feat], sn=sn))
            elif depth == sequence_length - 2:
                self.flow_emb_layers.append(FlowEmbedding(hidden_feat, [hidden_feat, out_feat, out_feat], sn=sn))
            else:
                self.flow_emb_layers.append(FlowEmbedding(hidden_feat, [hidden_feat, hidden_feat // 2, hidden_feat], sn=sn))

    def forward(self,
                feature_lst,  # list of length with with each element has size [B, C, N]
                pos_lst,      # list of length with with each element has size [B, 3, N]
                cutoff):
        assert len(feature_lst) == (self.depth + 1)
        for depth in range(self.depth):
            flow_emb_layer = self.flow_emb_layers[depth]
            mix_num = len(feature_lst) - 1
            for l in range(mix_num):
                feature_0 = feature_lst[l].contiguous()
                feature_1 = feature_lst[l+1].contiguous()
                pos_0 = pos_lst[l].contiguous()
                pos_1 = pos_lst[l+1].contiguous()

                _, feature_01 = flow_emb_layer(pos_0, pos_1, feature_0, feature_1, cutoff)
                feature_lst.append(feature_01)
            feature_lst = feature_lst[mix_num + 1:]
        assert len(feature_lst) == 1
        return feature_lst[-1]


class ActionTempoDis(nn.Module):
    def __init__(self, sequence_length, sn=True):
        super(ActionTempoDis, self).__init__()
        self.coarse_graining_module = nn.ModuleList()
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=512,
                radius=0.8,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
                sn=sn
            )
        )
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=256,
                radius=1.2,
                nsample=32,
                mlp=[128, 128, 256],
                use_xyz=True,
                sn=sn
            )
        )

        self.flow_module = FlowModule(in_feat=256, hidden_feat=256, out_feat=256, sequence_length=sequence_length, sn=sn)

        self.SA_pooling = SSGSetConv(
                mlp=[256, 256, 512], use_xyz=True, sn=sn
            )

        self.fc_layers = nn.Sequential(sp_norm(nn.Linear(512, 256)),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.3),
                                       sp_norm(nn.Linear(256, 64)),
                                       nn.BatchNorm1d(64),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1),
                                       sp_norm(nn.Linear(64, 1)))

    def forward(self,
                pos_lst,  # point cloud sequence
                cutoff,
                ):
        # first round of coarse sampling
        feature_lst = []
        pos_lst_new = []
        for pos in pos_lst:
            # feature in shape [B, C, N]
            # points in shape [B, N, 3]
            pos_ds_1, feature = self.coarse_graining_module[0](pos, pos.transpose(1, 2).contiguous() )
            feature_lst.append(feature)
            pos_lst_new.append(pos_ds_1)
        del pos_lst
        pos_lst = pos_lst_new
        # second round of coarse sampling
        feature_lst_new = []
        pos_lst_new = []
        for feature, pos in zip(feature_lst, pos_lst):
            pos_ds_2, feature = self.coarse_graining_module[1](pos, feature)
            feature_lst_new.append(feature)
            # need to make pos in shape [B, 3, N] to fit flowembedding layer
            pos_lst_new.append(pos_ds_2.permute(0, 2, 1))

        del pos_lst, feature_lst
        pos_lst = pos_lst_new
        feature_lst = feature_lst_new

        feature = self.flow_module(feature_lst, pos_lst, cutoff)

        pos = pos_lst[0].permute(0, 2, 1)   # back to [B, N, 3]
        _, feature = self.SA_pooling(pos, feature)
        # feature = feature.mean(dim=-1)
        # global max pooling
        feature = feature.view(-1, 512)
        feature = self.fc_layers(feature)
        return feature


class ActionSpatialDis(nn.Module):
    def __init__(self, sn=True):
        super(ActionSpatialDis, self).__init__()
        self.coarse_graining_module = nn.ModuleList()
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=512,
                radius=0.3,
                nsample=32,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
                sn=sn
            )
        )
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=256,
                radius=0.6,
                nsample=32,
                mlp=[128, 128, 128],
                use_xyz=True,
                sn=sn
            )
        )
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=128,
                radius=1.0,
                nsample=32,
                mlp=[128, 128, 256],
                use_xyz=True,
                sn=sn
            )
        )

        self.SA_pooling = SSGSetConv(
            mlp=[256, 256, 512], use_xyz=True, sn=sn,
        )

        self.fc_layers = nn.Sequential(sp_norm(nn.Linear(512, 256)),
                                           nn.BatchNorm1d(256),
                                           nn.LeakyReLU(),
                                           nn.Dropout(0.3),
                                           sp_norm(nn.Linear(256, 64)),
                                           nn.BatchNorm1d(64),
                                           nn.LeakyReLU(),
                                           nn.Dropout(0.1),
                                           sp_norm(nn.Linear(64, 1)))

    def forward(self,
                pos,  # point clouds [B, N, 3]
                ):
        feature = None
        for sa_module in self.coarse_graining_module:
            # feature in shape [B, C, N]
            # points in shape [B, N, 3]
            if feature is None:
                pos, feature = sa_module(pos, pos.transpose(1, 2).contiguous())
            else:
                pos, feature = sa_module(pos, feature)
        _, feature = self.SA_pooling(pos, feature)
        # feature = feature.mean(dim=-1)
        # global max pooling
        feature = feature.view(-1, 512)
        feature = self.fc_layers(feature)
        return feature


class FluidTempoDis(nn.Module):
    def __init__(self, sequence_length, sn=True):
        super(FluidTempoDis, self).__init__()
        self.coarse_graining_module = nn.ModuleList()
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=1024,
                radius=0.10,
                nsample=32,
                mlp=[3, 64, 128],
                use_xyz=True,
                sn=sn,
                mask_dummy=True,
                act_fn=nn.LeakyReLU()
            )
        )
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=256,
                radius=0.20,
                nsample=32,
                mlp=[128, 128, 256],
                use_xyz=True,
                sn=sn,
                act_fn=nn.LeakyReLU()
        )
        )

        self.flow_module = FlowModule(in_feat=256, hidden_feat=256, out_feat=256, sequence_length=sequence_length, sn=sn)

        self.SA_pooling = SSGSetConv(
                mlp=[256, 256, 256], use_xyz=True, sn=sn,
                act_fn=nn.LeakyReLU()

        )

        self.fc_layers = nn.Sequential(sp_norm(nn.Linear(256, 256)),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.2),
                                       sp_norm(nn.Linear(256, 64)),
                                       nn.BatchNorm1d(64),
                                       nn.LeakyReLU(),
                                       sp_norm(nn.Linear(64, 1)))

    def forward(self,
                pos_lst,  # point cloud sequence
                cutoff,
                feat_lst=None
                ):
        if feat_lst is not None:
            assert len(feat_lst) == len(pos_lst)
        # first round of coarse sampling
        feature_lst = []
        pos_lst_new = []
        for i, pos in enumerate(pos_lst):
            # feature in shape [B, C, N]
            # points in shape [B, N, 3]
            if feat_lst is not None:
                pos_ds_1, feature = self.coarse_graining_module[0](pos, feat_lst[i].transpose(1, 2).contiguous())
            else:
                pos_ds_1, feature = self.coarse_graining_module[0](pos, pos.transpose(1, 2).contiguous() )
            feature_lst.append(feature)
            pos_lst_new.append(pos_ds_1)
        del pos_lst
        pos_lst = pos_lst_new
        # second round of coarse sampling
        feature_lst_new = []
        pos_lst_new = []
        for feature, pos in zip(feature_lst, pos_lst):
            pos_ds_2, feature = self.coarse_graining_module[1](pos, feature)
            feature_lst_new.append(feature)
            # need to make pos in shape [B, 3, N] to fit flowembedding layer
            pos_lst_new.append(pos_ds_2.permute(0, 2, 1))

        del pos_lst, feature_lst
        pos_lst = pos_lst_new
        feature_lst = feature_lst_new

        feature = self.flow_module(feature_lst, pos_lst, 20*cutoff)

        pos = pos_lst[0].permute(0, 2, 1)   # back to [B, N, 3]
        _, feature = self.SA_pooling(pos, feature)
        # global max pooling
        feature = feature.view(-1, 256)
        feature = self.fc_layers(feature)
        return feature


class FluidSpatialDis(nn.Module):
    def __init__(self, sn=True):
        super(FluidSpatialDis, self).__init__()
        self.coarse_graining_module = nn.ModuleList()
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=1024,
                radius=0.15,
                nsample=32,
                mlp=[3, 64, 128],
                use_xyz=True,
                sn=True,
                mask_dummy=True,
                act_fn=nn.LeakyReLU()
            )
        )
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=512,
                radius=0.30,
                nsample=32,
                mlp=[128, 128, 128],
                use_xyz=True,
                sn=True,
                act_fn=nn.LeakyReLU()
            )
        )
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=128,
                radius=0.60,
                nsample=16,
                mlp=[128, 128, 256],
                use_xyz=True,
                sn=True,
                act_fn=nn.LeakyReLU()
            )
        )

        self.SA_pooling = SSGSetConv(
            mlp=[256, 256, 256], use_xyz=True, sn=sn,
        )

        self.fc_layers = nn.Sequential(sp_norm(nn.Linear(256, 256)),
                                           nn.BatchNorm1d(256),
                                           nn.LeakyReLU(),
                                           nn.Dropout(0.2),
                                           sp_norm(nn.Linear(256, 64)),
                                           nn.BatchNorm1d(64),
                                           nn.LeakyReLU(),
                                           sp_norm(nn.Linear(64, 1)))

    def forward(self,
                pos,  # point clouds [B, N, 3]
                ):
        feature = None
        for sa_module in self.coarse_graining_module:
            # feature in shape [B, C, N]
            # points in shape [B, N, 3]
            if feature is None:
                pos, feature = sa_module(pos, pos.transpose(1, 2).contiguous())
            else:
                pos, feature = sa_module(pos, feature)
        _, feature = self.SA_pooling(pos, feature)
        # global max pooling
        feature = feature.view(-1, 256)
        feature = self.fc_layers(feature)
        return feature


class ActionCls(nn.Module):
    # this is for testing extracted features
    def __init__(self, sequence_length):
        super(ActionCls, self).__init__()
        self.coarse_graining_module = nn.ModuleList()
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=512,
                radius=0.8,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
                sn=False
            )
        )
        self.coarse_graining_module.append(
            SSGSetConv(
                npoint=256,
                radius=1.2,
                nsample=32,
                mlp=[128, 128, 256],
                use_xyz=True,
                sn=False
            )
        )

        self.flow_module = FlowModule(in_feat=256, hidden_feat=256, out_feat=256, sequence_length=sequence_length,
                                      sn=False)
        self.SA_pooling = SSGSetConv(
                mlp=[256, 512, 512], use_xyz=True, sn=False
            )

        self.fc_layers = nn.Sequential(nn.Linear(512, 256),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.3),
                                       nn.Linear(256, 64),
                                       nn.BatchNorm1d(64),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(64, 20))

    def copy_params(self, module1, module2, trainable=False):
        dict_params1 = dict(module1.named_parameters())
        for name2, param2 in module2.named_parameters():
            if name2.find('orig') != -1:
                name2 = name2[:-5]
            if name2 in dict_params1.keys():
                dict_params1[name2].data.copy_(param2.data)
                dict_params1[name2].requires_grad = trainable

    def init_feature_extractor(self, trained_model):
        self.copy_params(self.coarse_graining_module, trained_model.coarse_graining_module, False)
        self.copy_params(self.flow_module, trained_model.flow_module, False)

    def forward(self,
                pos_lst,  # point cloud sequence
                cutoff,
                ):
        # first round of coarse sampling
        feature_lst = []
        pos_lst_new = []
        for pos in pos_lst:
            # feature in shape [B, C, N]
            # points in shape [B, N, 3]
            pos_ds_1, feature = self.coarse_graining_module[0](pos, pos.transpose(1, 2).contiguous() )
            feature_lst.append(feature)
            pos_lst_new.append(pos_ds_1)
        del pos_lst
        pos_lst = pos_lst_new
        # second round of coarse sampling
        feature_lst_new = []
        pos_lst_new = []
        for feature, pos in zip(feature_lst, pos_lst):
            pos_ds_2, feature = self.coarse_graining_module[1](pos, feature)
            feature_lst_new.append(feature)
            # need to make pos in shape [B, 3, N] to fit flowembedding layer
            pos_lst_new.append(pos_ds_2.permute(0, 2, 1))

        del pos_lst, feature_lst
        pos_lst = pos_lst_new
        feature_lst = feature_lst_new

        feature = self.flow_module(feature_lst, pos_lst, cutoff)

        pos = pos_lst[0].permute(0, 2, 1)   # back to [B, N, 3]
        _, feature = self.SA_pooling(pos, feature)
        # global max pooling
        feature = feature.view(-1, 512)
        feature = self.fc_layers(feature)
        return feature

