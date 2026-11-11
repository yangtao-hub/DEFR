import torch
import torch.nn as nn
import torch.nn.functional as F

def get_functional_act(act_name):
    if act_name == 'relu':
        return nn.ReLU(inplace=True)
    if act_name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if act_name == 'tanh':
        return nn.Tanh()
    if act_name == 'sigmoid':
        return nn.Sigmoid()
    return nn.Identity()

class conv3d_norm_act(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), pad_mode='zeros', norm_mode='in', act_mode='lrelu'):
        super().__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, padding_mode=pad_mode, bias=True)
        if norm_mode == 'in':
            self.norm = nn.InstanceNorm3d(out_planes, affine=True)
        elif norm_mode == 'gn':
            channels_per_group = 8
            num_groups = max(1, out_planes // channels_per_group)
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_planes, affine=True)
        elif norm_mode == 'bn':
            self.norm = nn.BatchNorm3d(out_planes, affine=True)
        else:
            self.norm = nn.Identity()
        self.act = get_functional_act(act_mode)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class residual_block_3d(nn.Module):

    def __init__(self, in_planes, out_planes, projection=False, pad_mode='zeros', norm_mode='in', act_mode='lrelu'):
        super(residual_block_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(conv3d_norm_act(in_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode), conv3d_norm_act(out_planes, out_planes, kernel_size=(3, 3, 3), padding=(1, 1, 1), pad_mode=pad_mode, norm_mode=norm_mode))
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1, 1, 1), padding=(0, 0, 0), norm_mode=norm_mode)
        self.act = get_functional_act(act_mode)

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y

class DistanceFusionBlock(nn.Module):

    def __init__(self, structure_channels, guide_channels, temperature=0.1):
        super().__init__()
        if structure_channels != guide_channels:
            self.align_net = nn.Conv3d(guide_channels, structure_channels, kernel_size=1, bias=False)
        else:
            self.align_net = nn.Identity()
        self.temperature = temperature

    def forward(self, structure_feat, guide_feat):
        if structure_feat.shape != guide_feat.shape:
            guide_aligned = self.align_net(guide_feat)
        else:
            guide_aligned = guide_feat
        diff = torch.mean((structure_feat - guide_aligned) ** 2, dim=1, keepdim=True)
        gate = torch.exp(-diff / self.temperature)
        self.last_gate = gate
        return (gate * guide_aligned + (1.0 - gate) * structure_feat, 1.0 - gate)

class UNetEncoder(nn.Module):

    def __init__(self, in_channels=1, filters=[16, 32, 64, 128, 256], pad_mode='zeros', norm_mode='in', act_mode='lrelu', **kwargs):
        super().__init__()
        self.depth = len(filters) - 1
        filters_in = [in_channels] + filters
        self.downC = nn.ModuleList([nn.Sequential(conv3d_norm_act(in_planes=filters_in[x], out_planes=filters_in[x + 1], kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode), residual_block_3d(filters_in[x + 1], filters_in[x + 1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)) for x in range(self.depth)])
        self.center1 = nn.Sequential(conv3d_norm_act(in_planes=filters_in[-2], out_planes=filters_in[-1], kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode), residual_block_3d(filters_in[-1], filters_in[-1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode))
        self.center2 = conv3d_norm_act(in_planes=filters_in[-1], out_planes=filters_in[-1], kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), pad_mode=pad_mode, norm_mode='', act_mode=act_mode)
        self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) for _ in range(self.depth)])

    def forward(self, x):
        down_u = []
        curr_x = x
        for i in range(self.depth):
            feat = self.downC[i](curr_x)
            down_u.append(feat)
            curr_x = self.downS[i](feat)
        x = self.center1(curr_x)
        x = self.center2(x)
        down_u.append(x)
        return down_u

class UNetDecoder_DF(nn.Module):

    def __init__(self, out_channels=1, in_channels=1, filters=[16, 32, 64, 128, 256], pad_mode='zeros', norm_mode='in', act_mode='lrelu', output_act='none', head_depth=1, adain=False, style_size=0, use_attention=False, fusion_type='distance', **kwargs):
        super().__init__()
        self.depth = len(filters) - 1
        filters_in = [in_channels] + filters
        self.adain = adain
        self.use_attention = use_attention
        self.fusion_type = fusion_type
        self.upC = nn.ModuleList([nn.Sequential(conv3d_norm_act(in_planes=filters_in[x + 1] + filters_in[x], out_planes=filters_in[x], kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode), residual_block_3d(filters_in[x], filters_in[x], projection=True, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)) for x in range(1, self.depth + 1)])
        self.upS = nn.ModuleList([nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False) for x in range(self.depth)])
        if self.use_attention:
            self.fusions = nn.ModuleList()
            for x in range(1, self.depth + 2):
                struct_ch = filters_in[x]
                guide_ch = filters_in[x]
                self.fusions.append(DistanceFusionBlock(structure_channels=struct_ch, guide_channels=guide_ch, temperature=0.1))
        head_pred = [residual_block_3d(filters_in[1], filters_in[1], projection=True, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode) for _ in range(head_depth - 1)] + [conv3d_norm_act(filters_in[1], out_channels, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0), pad_mode=pad_mode, norm_mode='', act_mode='')]
        self.head_pred = nn.Sequential(*head_pred)
        self.out_act = get_functional_act(output_act)

    def forward(self, enc_features, s=None, guide_features=None):
        fus_features = []
        alphas = []
        if self.use_attention and guide_features is not None:
            i = 0
            for enc_feature, guide_feature in zip(enc_features, guide_features):
                if enc_feature.shape == guide_feature.shape:
                    fus_feature, alpha = self.fusions[i](enc_feature, guide_feature)
                    fus_features.append(fus_feature)
                    alphas.append(alpha)
                    i += 1
                else:
                    print(f'Skipping fusion due to size mismatch: enc_feature {enc_feature.shape}, guide_feature {guide_feature.shape}')
        x = fus_features[-1]
        if self.adain and s is not None:
            x = self.center_adain(x, s)
        for idx, i in enumerate(range(self.depth - 1, -1, -1)):
            up_x = self.upS[i](x)
            skip_feat = fus_features[i]
            x = torch.cat([skip_feat, up_x], dim=1)
            x = self.upC[i](x)
            if self.adain and s is not None:
                x = self.upAdain[i](x, s)
        x = self.head_pred(x)
        x = self.out_act(x)
        return (x, alphas)
