import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from scipy.ndimage import binary_erosion
from torchmetrics.classification import BinaryAveragePrecision
from torchvision.utils import save_image
import utils
from data.dataset import build_dataloader, get_dataset
from loss import SSIM_Loss
from model import build_model
import torchio as tio
import random
from copy import deepcopy
from data.dataset.local_dae import local_noise_torch

class Trainer:

    def __init__(self, args, logging, checkpoint=None):
        self.args = args
        self.seed = args.seed
        self.train_img_dir = args.train_img_dir
        self.test_img_dir = args.test_img_dir
        self.test_lab_dir = args.test_lab_dir
        self.save_dir = args.save_dir
        self.in_channels = args.in_channels
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.filters = args.filters
        self.batch_size = args.batch_size
        self.logging = logging
        self.model_type = args.model
        self.use_amp = args.use_amp
        self.input_noise = args.input_noise
        self.fig_dir = os.path.join(args.save_dir, 'figs')
        os.makedirs(self.fig_dir, exist_ok=True)
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        train_augmentor = None
        test_augmentor = None
        utils.seed_everything(args.seed)
        train_dataset = get_dataset(args=args, mode='train', augmentor=train_augmentor)
        self.train_loader = build_dataloader(train_dataset, args.batch_size, mode='train', num_workers=args.num_workers, seed=args.seed)
        test_dataset = get_dataset(args=args, mode='test', augmentor=test_augmentor, num_data=args.test_num)
        self.test_loader = build_dataloader(test_dataset, batch_size=args.batch_size, mode='test', num_workers=args.num_workers, seed=args.seed)
        self.w = args.w_dict
        self.adv = 'adv' in self.w and self.w['adv'] > 0
        utils.seed_everything(args.seed)
        model_dict = build_model(args, checkpoint, model=args.model)
        self.model = model_dict['model']
        self.optimizer = model_dict['optimizer']
        self.start_epoch = model_dict['start_epoch']
        self.loss_dict = model_dict['loss_dict']
        self.max_ap_p = 0
        self.max_ap_p_path = None
        self.max_epochs = args.max_epochs
        self.log_interval = args.log_interval
        self.test_interval = args.test_interval
        self.checkpoint_interval = args.checkpoint_interval
        self.eval_ap = args.eval_ap
        self.SSIM = SSIM_Loss(window_size=11, size_average=False, do_3d=True)
        fs = 9
        self.smooth_op = torch.nn.Sequential(torch.nn.MaxPool3d(kernel_size=3, stride=1, padding=1), torch.nn.AvgPool3d(kernel_size=fs, stride=1, padding=fs // 2))
        self.need_space_aug = args.need_space_aug
        self.noise_param = args.noise_param
        self.transforms_dict = {tio.RandomAffine(scales=(0.95, 1.05), degrees=0, translation=0): 0.25, tio.RandomAffine(scales=0, degrees=(5, 5, 5), translation=0): 0.25, tio.RandomAffine(scales=0, degrees=0, translation=(5, 5, 5)): 0.25, tio.RandomElasticDeformation(num_control_points=7, max_displacement=5): 0.25}

    def space_aug(self, x):
        x_aug_list = []
        for s_i in range(len(x)):
            img = torch.tensor(x[s_i])
            transform = tio.OneOf(self.transforms_dict)
            transformed_image = transform(img)
            t_img = transformed_image.numpy()
            x_aug_list.append(t_img[None])
        x_augs = np.concatenate(x_aug_list, axis=0)
        x_augs = np.clip(x_augs, 0.0, 1.0)
        assert x.shape == x_augs.shape
        return x_augs

    def run(self):
        utils.seed_everything(self.seed)
        lr_list = []
        self.logging.info('Learnable para : {:.3f} M'.format(utils.get_learnable_para(self.model) / 1000 / 1000))
        for epoch in range(self.start_epoch, self.max_epochs):
            self.logging.info('======> Epoch : {}, Current_lr : {}'.format(epoch, self.optimizer.param_groups[0]['lr']))
            train_loss = self.train(epoch)
            torch.cuda.empty_cache()
            lr_list.append(self.optimizer.param_groups[0]['lr'])
            self.print_lr(lr_list)
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_model(epoch, save_checkpoint=True)
            if epoch % self.test_interval == 0 and epoch >= 0:
                if self.eval_ap:
                    aps = self.test_ap(epoch)
                    torch.cuda.empty_cache()
                    self.save_loss(epoch, train_loss, aps)
                    self.print_loss(mode='test')
                    self.save_model(epoch, aps=aps)
        torch.cuda.empty_cache()

    def apply_intensity_augmentation(self, x):
        x = x.astype(np.float32)
        batch_size = x.shape[0]
        for i in range(batch_size):
            img = x[i]
            fg_mask = img > 0
            if not np.any(fg_mask):
                continue
            if np.random.rand() < 0.5:
                gamma_range = (0.7, 1.43)
                log_gamma = np.random.uniform(np.log(gamma_range[0]), np.log(gamma_range[1]))
                gamma = np.exp(log_gamma)
                img[fg_mask] = np.power(img[fg_mask], gamma)
            if np.random.rand() < 0.5:
                scale_range = (0.9, 1.1)
                scale_factor = np.random.uniform(scale_range[0], scale_range[1])
                img *= scale_factor
            if np.random.rand() < 0.5:
                shift_factor = np.random.uniform(-0.1, 0.1)
                img[fg_mask] = img[fg_mask] + shift_factor
            img[~fg_mask] = 0.0
            x[i] = np.clip(img, 0.0, 1.0)
        return x

    def local_dae_training_pair(self, x):
        device = next(self.model.parameters()).device
        clean = torch.as_tensor(x, dtype=torch.float32, device=device)
        clean = F.interpolate(clean, self.input_size, mode='trilinear', align_corners=False)
        clean = torch.clip(clean, 0.0, 1.0)
        if clean.ndim != 5 or clean.shape[1] != 1:
            raise ValueError('local DAE training requires single-channel 3D images')
        with torch.no_grad():
            noisy, _ = local_noise_torch(clean, shape_mode=getattr(self.args, 'local_dae_shape_mode', 'ellipsoid'), coverage_mode=getattr(self.args, 'local_dae_coverage_mode', 'current'), target_coverage=getattr(self.args, 'local_dae_target_coverage', 0.25), noise_sigma=self.args.local_dae_noise_sigma, noise_amplitude=self.args.local_dae_noise_amplitude)
        return (clean, noisy)

    def train(self, epoch):
        self.logging.info('======================TRAIN MODE======================')
        train_loss = utils.AverageMeter()
        losses = utils.AverageMeters()
        self.model.train()
        for i, (x, _) in enumerate(self.train_loader):
            if self.need_space_aug:
                x = self.space_aug(x)
            x = self.apply_intensity_augmentation(x)
            if self.input_noise == ['local_dae']:
                x, x_n = self.local_dae_training_pair(x)
            elif self.input_noise == ['none']:
                x = torch.from_numpy(x).float().cuda()
                x = F.interpolate(x, self.input_size, mode='trilinear', align_corners=False)
                x = torch.clip(x, 0.0, 1.0)
                x_n = x.clone()
            else:
                raise ValueError('Supported input_noise values are local_dae and none')
            if torch.isnan(x_n).any() or torch.isinf(x_n).any() or torch.isnan(x).any() or torch.isinf(x).any():
                print('Input contains NaN or Inf, skipping this batch.')
                continue
            with autocast(enabled=self.use_amp):
                loss, rts, rtp, ld = self.model.loss(x_n, x, w=self.w)
            if torch.isnan(loss) or torch.isinf(loss):
                print('Loss is {}, skipping this batch.'.format(loss))
                continue
            elif loss > 2.5:
                print('large loss:', loss)
                continue
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20000, norm_type=2)
                if torch.isnan(grad) or torch.isinf(grad):
                    print('Gradient is {}, skipping this batch.'.format(grad))
                    continue
                elif grad > 200000:
                    print('large grad:', grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                if torch.isnan(grad) or torch.isinf(grad):
                    print('Gradient is {}, skipping this batch.'.format(grad))
                    continue
                elif grad > 200:
                    print('grad:', grad)
                    continue
                self.optimizer.step()
            bs = x.size(0)
            train_loss.update(loss.item(), bs)
            losses.update(ld, bs)
            if i % self.log_interval == 0:
                self.log_loss(epoch, loss.item(), mode='train', iters=i, ld=ld)
            del loss
        self.log_loss(epoch, train_loss.avg, ld=losses.avg_dict)
        draw_list = []
        shape = x.shape[-3:]
        z = shape[0] // 2
        x_n = torch.clip(x_n, 0.0, 1.0)
        x = torch.clip(x, 0.0, 1.0)
        x_n = x_n[:, :, z].cpu()
        x = x[:, :, z].cpu()
        draw_list.append(x)
        draw_list.append(x_n)
        for i in range(len(rts)):
            rt = rts[i]
            for c in range(rt.shape[1]):
                tmp = rt[:, c:c + 1]
                if rt.shape[-3:] < shape:
                    if rtp[i] == 'img':
                        tmp = F.interpolate(tmp, shape, mode='trilinear', align_corners=False)
                    else:
                        tmp = F.interpolate(tmp, shape, mode='nearest')
                if rtp[i] == 'img':
                    tmp = torch.clip(tmp, 0.0, 1.0)
                elif rtp[i] == 'prob':
                    tmp = torch.clip(tmp, 0.0, 1.0)
                else:
                    raise
                tmp = tmp[:, :, z]
                draw_list.append(tmp.cpu())
        imgs_concat = torch.cat(draw_list, dim=-1)
        save_image(imgs_concat, os.path.join(self.fig_dir, 'train-{}.png'.format(epoch)), nrow=1)
        return train_loss.avg

    def test_ap(self, epoch):
        self.logging.info('======================TEST MODE======================')
        self.model.eval()
        pred_p_l1 = []
        pred_p_ssim = []
        pred_p_hybrid = []
        label_p = []
        SSIM = self.SSIM
        x_list = []
        x_cor_list = []
        x_r_list = []
        y_list = []
        for idx, (x, y) in enumerate(self.test_loader):
            x = torch.from_numpy(x).float().cuda()
            y = torch.from_numpy(y).float().cuda()
            target_size = self.output_size
            x = F.interpolate(x, self.input_size, mode='trilinear', align_corners=False)
            y = F.interpolate(y, target_size, mode='nearest')
            with torch.no_grad():
                with autocast(enabled=False):
                    details = self.model(x, return_details=True)
                    x_r = details['reconstruction']
                    a_e = details['edges']
                    c = details['reference_gates'][0]
            x = F.interpolate(x, target_size, mode='trilinear', align_corners=False)
            x_r = F.interpolate(x_r, target_size, mode='trilinear', align_corners=False)
            c = F.interpolate(c, target_size, mode='trilinear', align_corners=False)
            a_e = F.interpolate(a_e, target_size, mode='trilinear', align_corners=False)
            x_list.append(x)
            y_list.append(y)
            x_cor_list.append(a_e)
            x_r_list.append(x_r)
            l1 = (x - x_r).abs().detach()
            ssim = SSIM(x_r, x).detach()
            l1_o = l1.clone().detach()
            ssim_o = ssim.clone().detach()
            l1_w_o = l1.clone().detach() * ssim.clone().detach()
            l1 = (-self.smooth_op(-l1_o.clone().cuda())).abs()
            ssim = (-self.smooth_op(-ssim_o.clone().cuda())).abs()
            l1_w = (-self.smooth_op(-l1_w_o.clone().cuda())).abs()
            m = x.detach() > 0
            m = m.cpu().numpy()
            m = torch.from_numpy(m).cuda()
            l1 = (l1 * m).cpu()
            ssim = (ssim * m).cpu()
            l1_w = (l1_w * m).cpu()
            pred_p_l1.append(l1)
            pred_p_ssim.append(ssim)
            pred_p_hybrid.append(l1_w)
            label_p.append(y.cpu() * 1.0)
        pred_p_l1 = torch.cat(pred_p_l1)
        pred_p_l1_w = torch.cat(pred_p_hybrid)
        pred_p_ssim = torch.cat(pred_p_ssim)
        label_p = torch.cat(label_p)
        x_list = torch.cat(x_list)
        x_cor_list = torch.cat(x_cor_list)
        x_r_list = torch.cat(x_r_list)
        y_list = torch.cat(y_list)
        self.evaluate_regional_l1(x_list, x_r_list, y_list, tg='Final')
        torch.cuda.empty_cache()
        self.logging.info('Images inference is completed, the calculation of AP starts...')
        if self.use_amp:
            ap_p_l1_t2 = self.ap_gpu(pred_p_l1, label_p, cuda=True)
            ap_p_l1_w_t2 = self.ap_gpu(pred_p_l1_w, label_p, cuda=True)
            ap_p_ssim_t2 = self.ap_gpu(pred_p_ssim, label_p, cuda=True)
        else:
            ap_p_l1_t2 = self.ap_gpu(pred_p_l1, label_p)
            ap_p_l1_w_t2 = self.ap_gpu(pred_p_l1_w, label_p)
            ap_p_ssim_t2 = self.ap_gpu(pred_p_ssim, label_p)
        y = self.slices_draw(y).cpu()
        x = self.slices_draw(x).cpu()
        a_e = self.slices_draw(a_e).cpu()
        x_r = self.slices_draw(x_r).cpu()
        c = self.slices_draw(c).cpu()
        l1_o = self.slices_draw(l1_o).cpu()
        l1 = self.slices_draw(l1).cpu()
        l1_w_o = self.slices_draw(l1_w_o).cpu()
        l1_w = self.slices_draw(l1_w).cpu()
        ssim_o = self.slices_draw(ssim_o).cpu()
        ssim = self.slices_draw(ssim).cpu()
        x = torch.clip(x, 0.0, 1.0)
        a_e = a_e / a_e.max().clamp_min(1e-08)
        x_r = torch.clip(x_r, 0.0, 1.0)
        c = torch.clip(c, 0.0, 1.0)
        y = y * 1.0
        l1_o = l1_o / torch.amax(l1_o, dim=(1, 2, 3), keepdim=True)
        l1 = l1 / torch.amax(l1, dim=(1, 2, 3), keepdim=True)
        l1_w_o = l1_w_o / torch.amax(l1_w_o, dim=(1, 2, 3), keepdim=True)
        l1_w = l1_w / torch.amax(l1_w, dim=(1, 2, 3), keepdim=True)
        ssim_o = ssim_o / torch.amax(ssim_o, dim=(1, 2, 3), keepdim=True)
        ssim = ssim / torch.amax(ssim, dim=(1, 2, 3), keepdim=True)
        imgs_concat = torch.cat([x, a_e, c, x_r, ssim_o, ssim, l1_o, l1, l1_w_o, l1_w, y], dim=-1)
        save_image(imgs_concat, os.path.join(self.fig_dir, 'test-{}.png'.format(epoch)), nrow=1)
        self.logging.info('====>  Test (pixel) hybrid AP: {:.4f} l1 AP: {:.4f} ssim AP: {:.4f} '.format(ap_p_l1_w_t2, ap_p_l1_t2, ap_p_ssim_t2))
        aps = {'ap_l1': ap_p_l1_t2, 'ap_hybrid': ap_p_l1_w_t2, 'ap_ssim': ap_p_ssim_t2}
        return aps

    def ap_gpu(self, pred, label, cuda=False):
        bap = BinaryAveragePrecision(thresholds=None)
        if cuda:
            ap = bap(pred.cuda().flatten(), label.cuda().to(torch.int).flatten()).cpu()
        else:
            ap = bap(pred.flatten(), label.to(torch.int).flatten())
        return ap

    def log_loss(self, epoch, loss, iters=-1, ld={}, mode='train'):
        if iters >= 0:
            if mode == 'train':
                info = 'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, iters * self.batch_size, len(self.train_loader.dataset), 100.0 * iters / len(self.train_loader), loss)
            elif mode == 'valid':
                info = 'Valid Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, iters * self.batch_size, len(self.valid_loader.dataset), 100.0 * iters / len(self.valid_loader), loss)
            info += ' ( '
            for key in ld.keys():
                info += '{}: {:.6f} '.format(key, ld[key])
            info += ')'
        else:
            info = '==> Epoch: {} Average loss: {:.6f}'.format(epoch, loss)
        self.logging.info(info)

    def save_loss(self, epoch, train_loss, aps):
        self.loss_dict['epoch'].append(epoch)
        self.loss_dict['train_loss'].append(train_loss)
        self.loss_dict['ap_l1'].append(aps['ap_l1'].numpy())
        self.loss_dict['ap_hybrid'].append(aps['ap_hybrid'].numpy())
        self.loss_dict['ap_ssim'].append(aps['ap_ssim'].numpy())
        np.save(os.path.join(self.save_dir, 'loss.npy'), self.loss_dict)
        with open(os.path.join(self.save_dir, 'loss.txt'), 'a') as f:
            f.write('\nepoch = {}, train loss = {:.4f}\n'.format(epoch, train_loss))

    def print_lr(self, lr_list):
        plt.plot(list(range(1, len(lr_list) + 1)), lr_list)
        plt.title('Learning rate schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate')
        plt.savefig(os.path.join(self.save_dir, 'lr_schedule.png'))
        plt.close()

    def print_loss(self, mode='train'):
        assert mode in ['train', 'test']
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        if mode == 'train':
            ax1.plot(self.loss_dict['epoch'], self.loss_dict['train_loss'], '-', label='train_loss')
            ax1.set_ylabel('loss')
            fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            plt.savefig(os.path.join(self.save_dir, 'train.png'))
        elif mode == 'test':
            epoch = np.array(self.loss_dict['epoch'])
            train_loss = np.array(self.loss_dict['train_loss'])
            ap_hybrid = np.array(self.loss_dict['ap_hybrid'])
            ap_l1 = np.array(self.loss_dict['ap_l1'])
            ap_ssim = np.array(self.loss_dict['ap_ssim'])
            ax1.plot(epoch, train_loss, 'b', label='train_loss')
            ax1.set_ylabel('loss')
            ax2 = ax1.twinx()
            ax2.plot(epoch, ap_hybrid, 'r', label='ap_hybrid', linestyle='--')
            ax2.plot(epoch, ap_l1, 'y', label='ap_l1', linestyle='--')
            ax2.plot(epoch, ap_ssim, 'g', label='ap_ssim', linestyle='--')
            ax2.set_ylabel('AP')
            fig.legend(loc=2, bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes, ncol=2)
            plt.savefig(os.path.join(self.save_dir, 'test.png'))
        plt.close(fig)

    def save_model(self, epoch, aps=None, save_checkpoint=False):
        if save_checkpoint:
            utils.save_checkpoint(path=os.path.join(self.save_dir, 'checkpoint.pth'), model=self.model, epoch=epoch, optimizer=self.optimizer, lr=self.args.lr, weight_decay=self.args.weight_decay)
        if aps is not None:
            endf = 'ap'
            ap_p = aps['ap_hybrid']
            max_ap_p = self.max_ap_p
            if ap_p >= max_ap_p:
                utils.save_checkpoint(path=os.path.join(self.save_dir, 'best_{}_{}.pth'.format(endf, epoch)), model=self.model, epoch=epoch, optimizer=self.optimizer, lr=self.args.lr, weight_decay=self.args.weight_decay)
                with open(os.path.join(self.save_dir, 'best_{}.txt'.format(endf)), 'a') as f:
                    f.write('\nepoch = {}, ap_p = {:.4f}, ap_hybrid = {:.4f}, ap_l1 = {:.4f}, ap_ssim = {:.4f}\n'.format(epoch, ap_p, aps['ap_hybrid'].numpy(), aps['ap_l1'].numpy(), aps['ap_ssim'].numpy()))
                self.max_ap_p = ap_p
                if self.max_ap_p_path is None:
                    self.max_ap_p_path = os.path.join(self.save_dir, 'best_{}_{}.pth'.format(endf, epoch))
                else:
                    os.remove(self.max_ap_p_path)
                    self.max_ap_p_path = os.path.join(self.save_dir, 'best_{}_{}.pth'.format(endf, epoch))

    def slices_draw(self, vol):
        vol = vol.squeeze()
        while vol.ndim < 4:
            vol = vol[None]
        if not hasattr(self, 'slice_ids'):
            self.slice_ids = []
            for i in range(len(vol)):
                if vol[i].max() == 0:
                    self.slice_ids.append(vol.size(1) // 2)
                    continue
                v = vol[i].sum((1, 2))
                idxs = torch.nonzero(v)
                idx = (idxs.max() + idxs.min()).item() // 2
                self.slice_ids.append(idx)
        slices = vol[range(0, len(vol)), self.slice_ids]
        return slices[:, None]

    def evaluate_regional_l1(self, x, x_r, y, tg):
        with torch.no_grad():
            l1_map = torch.abs(x - x_r)
            foreground_mask = (x > 0).float()
            normal_mask = (y == 0).float() * foreground_mask
            anomaly_mask = (y >= 1).float() * foreground_mask
            num_normal_pixels = normal_mask.sum()
            if num_normal_pixels > 0:
                normal_l1 = (l1_map * normal_mask).sum() / num_normal_pixels
            else:
                normal_l1 = torch.tensor(0.0, device=x.device)
            num_anomaly_pixels = anomaly_mask.sum()
            if num_anomaly_pixels > 0:
                anomaly_l1 = (l1_map * anomaly_mask).sum() / num_anomaly_pixels
            else:
                anomaly_l1 = torch.tensor(0.0, device=x.device)
            self.logging.info('--- {} Reconstruction Quality Analysis (Foreground Only) ---'.format(tg))
            self.logging.info(f'Normal Region Avg L1  : {normal_l1.item():.6f}')
            self.logging.info(f'Anomaly Region Avg L1 : {anomaly_l1.item():.6f}')
            self.logging.info(f'---------------------------------------')
        return (normal_l1.item(), anomaly_l1.item())
