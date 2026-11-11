import torch
from .factory import create_model

def build_model(args, checkpoint=None, model='SingleEdgeImageUNet'):
    if checkpoint is not None or model != 'SingleEdgeImageUNet':
        raise ValueError('This entrypoint starts this method from scratch')
    if args.optimizer != 'Adam' or args.weight_decay != 0:
        raise ValueError('This configuration uses Adam with zero weight decay')
    net = create_model(vars(args)).cuda()
    return {'model': net, 'optimizer': torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0), 'start_epoch': 0, 'loss_dict': {k: [] for k in ['epoch', 'train_loss', 'valid_loss', 'ap_hybrid', 'ap_l1', 'ap_ssim', 'ap_pred']}}
