import kornia as K
import torch
from torch import nn
from loss import SSIM_Loss
from .Edge_Enhancement_Rec import UNetEncoder, UNetDecoder_DF

class SingleEdgeImageUNet(nn.Module):
    single_reconstruction = True

    def __init__(self, input_size=None, filters=None, adain=False, fusion_type='distance', fusion_temperature=0.1):
        super().__init__()
        if adain:
            raise ValueError('SingleEdgeImageUNet does not use AdaIN')
        if fusion_type != 'distance':
            raise ValueError('unsupported fusion_type: ' + str(fusion_type))
        if fusion_temperature <= 0:
            raise ValueError('fusion_temperature must be positive')
        self.input_size = list(input_size or [112, 160, 160])
        self.filters = list(filters or [16, 32, 64, 128, 256])
        if len(self.filters) < 2 or any((c <= 0 for c in self.filters)):
            raise ValueError('filters must contain at least two positive channel counts')
        self.adain = False
        self.fusion_type = fusion_type
        self.fusion_temperature = float(fusion_temperature)
        encoder_args = dict(in_channels=1, filters=self.filters, pad_mode='zeros', norm_mode='in', act_mode='lrelu')
        self.edge_encoder = UNetEncoder(**encoder_args)
        self.image_encoder = UNetEncoder(**encoder_args)
        self.decoder = UNetDecoder_DF(in_channels=1, out_channels=1, filters=self.filters, pad_mode='zeros', norm_mode='', act_mode='lrelu', output_act='none', head_depth=1, adain=False, style_size=0, use_attention=True, fusion_type=fusion_type)
        if fusion_type == 'distance':
            for fusion in self.decoder.fusions:
                fusion.temperature = self.fusion_temperature
        self.l1_loss = nn.L1Loss()
        self.SSIM_loss = SSIM_Loss(window_size=11, size_average=True, do_3d=True)

    @staticmethod
    @torch.no_grad()
    def canny_edges(x):
        slices = torch.split(x.detach().squeeze(1), 1, dim=1)
        return torch.cat([K.filters.canny(s.float(), low_threshold=0.1, high_threshold=0.2, kernel_size=(5, 5), sigma=(1.0, 1.0), hysteresis=True)[1] for s in slices], dim=1).unsqueeze(1).to(x.dtype)

    def forward(self, x_n, mask=None, return_details=False):
        if x_n.ndim != 5 or x_n.shape[1] != 1:
            raise ValueError('expected [B,1,D,H,W] input')
        factor = 2 ** (len(self.filters) - 1)
        if any((size % factor for size in x_n.shape[-3:])):
            raise ValueError('spatial dimensions must be divisible by ' + str(factor))
        if mask is None:
            mask = x_n.detach() > 0
        edges = self.canny_edges(x_n)
        edge_features = self.edge_encoder(edges)
        image_features = self.image_encoder(x_n)
        reconstruction, edge_weights = self.decoder(edge_features, guide_features=image_features)
        reconstruction = reconstruction * mask
        if return_details:
            return {'reconstruction': reconstruction, 'edges': edges, 'reference_gates': tuple((1.0 - weight for weight in edge_weights))}
        return reconstruction

    def loss(self, x_n, x, w=None):
        if x_n.shape != x.shape:
            raise ValueError('input and clean target must have the same shape')
        mask = x.detach() > 0
        if not mask.any():
            raise ValueError('reconstruction loss requires non-empty foreground')
        details = self.forward(x_n, mask=mask, return_details=True)
        reconstruction = details['reconstruction']
        l1 = self.l1_loss(reconstruction[mask], x[mask])
        total = l1
        logs = {'loss_rec_l1': l1.item()}
        ssim_weight = float((w or {}).get('l_ssim', 0.0))
        if ssim_weight > 0:
            ssim = ssim_weight * self.SSIM_loss(reconstruction, x, mask)
            total = total + ssim
            logs['loss_rec_ssim'] = ssim.item()
        panels = (details['edges'], *details['reference_gates'], reconstruction)
        return (total, panels, ('img',) * len(panels), logs)
