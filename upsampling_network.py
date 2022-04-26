import torch
import torch.nn as nn

from gcn_lib.pointnet import EdgeConv, IDGCNLayer, conv_bn_layer, build_shared_mlp


class GCNFeatureExtractor(nn.Module):
    def __init__(self,
                 layer_num,
                 in_node_feat_dim,
                 out_node_feat_dim,
                 node_emb_dim=128,
                 ):
        super(GCNFeatureExtractor, self).__init__()
        self.conv_layers = nn.ModuleList()
        for l in range(layer_num):
            if l == 0:
                self.conv_layers.append(EdgeConv(in_node_feat_dim, node_emb_dim,
                                                 bn=False, insn=False, k=20, mlp_layer=True))
            elif l == (layer_num - 1):
                self.conv_layers.append(IDGCNLayer(node_emb_dim, out_node_feat_dim,
                                                   bn=False, insn=False, residual=True))
            else:
                self.conv_layers.append(IDGCNLayer(node_emb_dim, node_emb_dim,
                                                   bn=False, insn=False, ln=False, residual=True))

    def forward(self, feature, pos=None):
        # feature [B, N, C]
        feature = feature.permute(0, 2, 1).contiguous()
        feature_lst = []
        for l, conv_layer in enumerate(self.conv_layers):
            if l == 0:
                if pos is not None:
                    feature = conv_layer.forward(feature, pos)
                else:
                    feature = conv_layer.forward(feature)
            else:
                feature = conv_layer.forward(feature)
                feature_lst += [feature]
        feature = torch.cat(feature_lst, dim=1)   # [B, C, N, 1]
        return feature


class UpsamplingModule(nn.Module):
    def __init__(self,
                 in_node_feat_dim,
                 upsample_ratio,
                 gcn_layer=2,
                 ):
        super(UpsamplingModule, self).__init__()
        out_node_feat_dim = 3*upsample_ratio  # 3 dimension problem
        self.upsample_ratio = upsample_ratio
        self.upsample_layers = nn.ModuleList()
        for l in range(gcn_layer):
            if l != gcn_layer - 1:
                self.upsample_layers.append(conv_bn_layer(in_node_feat_dim, in_node_feat_dim//4, norm='none'))
                self.upsample_layers.append(EdgeConv(in_node_feat_dim//4, in_node_feat_dim,
                                                     aggregate='max', mlp_layer=True, k=12, bn=False, insn=False))
            else:
                self.upsample_layers.append(conv_bn_layer(in_node_feat_dim, in_node_feat_dim//4, norm='none'))
                self.upsample_layers.append(EdgeConv(in_node_feat_dim//4,
                                                     in_node_feat_dim,
                                                     aggregate='max', mlp_layer=True, k=4, bn=False, insn=False))
        self.decoder = nn.Sequential(
            build_shared_mlp([in_node_feat_dim, out_node_feat_dim//2, out_node_feat_dim], norm='none'),
            nn.Conv2d(out_node_feat_dim, out_node_feat_dim, 1, 1, 0, bias=True))

    def forward(self, feature):
        # feature [B, C, N, 1]
        for l, layer in enumerate(self.upsample_layers):
            feature = layer.forward(feature)
        feature = self.decoder(feature)   # [B, out_C, N, 1]
        feature = feature.squeeze(-1).permute(0, 2, 1).contiguous()
        return feature


class BinaryMaskingModule(nn.Module):
    def __init__(self,
                 in_node_feat_dim,
                 gcn_layer=2,
                 ):
        super(BinaryMaskingModule, self).__init__()

        self.upsample_layers = nn.ModuleList()
        for l in range(gcn_layer):
            if l != gcn_layer - 1:
                self.upsample_layers.append(conv_bn_layer(in_node_feat_dim, in_node_feat_dim // 4, norm='none'))
                self.upsample_layers.append(EdgeConv(in_node_feat_dim // 4, in_node_feat_dim,
                                                     aggregate='max', mlp_layer=True, k=12, bn=False, insn=False))
            else:
                self.upsample_layers.append(conv_bn_layer(in_node_feat_dim, in_node_feat_dim // 4, norm='none'))
                self.upsample_layers.append(EdgeConv(in_node_feat_dim // 4,
                                                     in_node_feat_dim,
                                                     aggregate='sum', mlp_layer=False, k=8, bn=False, insn=False))
        self.decoder = nn.Sequential(
            build_shared_mlp([in_node_feat_dim, in_node_feat_dim//2, in_node_feat_dim//4], norm='none'),
            nn.Conv2d(in_node_feat_dim//4, 1, 1, 1, 0, bias=True))

    def forward(self, feature):
        for l, layer in enumerate(self.upsample_layers):
            feature = layer.forward(feature)
        feature = torch.nn.ReLU()(self.decoder(feature))
        feature = feature.squeeze(-1).permute(0, 2, 1).contiguous()
        return feature


# standard upsampling with masking module
class SRNet(nn.Module):
    def __init__(self,
                 in_feats,
                 node_emb_dim,
                 upsample_ratio=8,
                 feature_extractor_depth=3,
                 ):
        super(SRNet, self).__init__()
        self.in_feats = in_feats
        self.feature_extractor = GCNFeatureExtractor(
                                                  layer_num=feature_extractor_depth,
                                                  in_node_feat_dim=in_feats,
                                                  out_node_feat_dim=node_emb_dim,
                                                  )

        self.upsampling_block = UpsamplingModule(node_emb_dim*(feature_extractor_depth-1), upsample_ratio)

        self.filter_block = BinaryMaskingModule(node_emb_dim*(feature_extractor_depth-1))
        self.upsample_ratio = upsample_ratio

        # some hyperparameter
        self.epsilon = 0.01

    def expand_pos_with_masking(self, pos, upsample_edge, binary_mask, hard_masking=False):
        batch_size, _, _ = pos.shape
        binary_mask = binary_mask.detach()  # stop gradient
        binary_mask = binary_mask.view(batch_size, -1, 1) > self.epsilon

        pos_duplicate = torch.cat([pos] * self.upsample_ratio, dim=2)    # [b, N, r*3]
        upsample_edge = upsample_edge * binary_mask.float()
        expanded_pos = pos_duplicate.view((batch_size, -1, 3)) + upsample_edge.view((batch_size, -1, 3))

        if hard_masking:
            hard_mask = torch.cat([binary_mask] * self.upsample_ratio, dim=2)  # [b, N, r]
            hard_mask[:, :, 0] = True
            num_of_points = torch.sum(torch.sum(hard_mask, dim=-1), dim=-1)  # [b]
            max_num = torch.max(num_of_points)
            hard_mask = hard_mask.view(batch_size, -1)  # [b, N*r]
            # pad each batch
            if expanded_pos.shape[0] > 1 and torch.any(num_of_points != max_num):
                unpadded_pos = expanded_pos.clone()
                expanded_pos[~hard_mask] = 999
                return unpadded_pos, expanded_pos
            else:
                unpadded_pos = expanded_pos.clone()
                expanded_pos = expanded_pos[hard_mask]  # [b*N*r0, 3] r0 < r, r0 may vary in each batch
                expanded_pos = expanded_pos.view((batch_size, -1, 3))
                return unpadded_pos, expanded_pos
        else:
            return expanded_pos, None

    def forward_with_context(self, feature, pos, previous_mask):
        # this stores running history when used to rollout the whole sequence
        encoding = self.feature_extractor(feature)
        edge = self.upsampling_block(encoding)
        mask = self.filter_block(encoding)
        mask[mask < 0.6] = 0.
        mask[mask > 0.6] = 0.6

        if len(previous_mask) >= 25:
            previous_mask = previous_mask[-24:]
            previous_mask.append(mask)
        else:
            previous_mask.append(mask)
        mask = torch.mean(torch.cat(previous_mask, dim=0), dim=0)
        _, pos = self.expand_pos_with_masking(pos, edge, mask, hard_masking=True)
        return pos, previous_mask

    def forward(self, feature, pos, hard_masking=False):
        if self.in_feats > 3:
            encoding = self.feature_extractor(feature, pos)
        else:
            encoding = self.feature_extractor(feature)
        edge = self.upsampling_block(encoding)
        mask = self.filter_block(encoding)

        pos, padded_pos = self.expand_pos_with_masking(pos, edge, mask, hard_masking=hard_masking)
        return pos, mask, padded_pos   # padded or being hard masked


# no masking applied, decrease the num of parameter
class NoMaskSRNet(nn.Module):
    def __init__(self,
                 in_feats,
                 node_emb_dim,
                 upsample_ratio=8,
                 feature_extractor_depth=3,
                 ):
        super(NoMaskSRNet, self).__init__()

        self.feature_extractor = GCNFeatureExtractor(
                                                  layer_num=feature_extractor_depth,
                                                  in_node_feat_dim=in_feats,
                                                  out_node_feat_dim=node_emb_dim,
                                                  )

        self.upsampling_block = UpsamplingModule(node_emb_dim*(feature_extractor_depth-1), upsample_ratio)

        self.upsample_ratio = upsample_ratio

    def expand_pos(self, pos, upsample_edge):
        B, _, _ = pos.shape
        upsample_edge = upsample_edge
        pos_duplicate = torch.cat([pos] * self.upsample_ratio, dim=-1)
        expanded_pos = pos_duplicate.view((B, -1, 3)) + upsample_edge.view((B, -1, 3))
        return expanded_pos

    def forward(self, feature, pos):
        if len(feature.shape) == 2:
            feature = feature.unsqueeze(0)
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)
        encoding = self.feature_extractor(feature)
        edge = self.upsampling_block(encoding)
        pos = self.expand_pos(pos, edge)
        return pos, edge.view(pos.shape[0], -1, 3)
