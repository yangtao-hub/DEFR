from case_lists import read_names
import fnmatch
import os
import SimpleITK as sitk
import numpy as np
import torch
import torch.utils.data as data
from torch.nn import functional as F
from tqdm import tqdm

class DataFolderDR_cache(data.Dataset):
    background: int = 0

    def __init__(self, image_dir='', case_list=None, label_dir=None, mode='train', filename=[], train_rate=0.8, augmentor=None, norm_method='', pair=False, num_data=32, input_size=[1, 1, 1]):
        super(DataFolderDR_cache, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_list = np.array(sorted(fnmatch.filter(os.listdir(image_dir), '*.nii.gz')))
        self.norm_method = norm_method
        self.pair = pair
        self.input_size = input_size
        if case_list is None:
            raise ValueError('Explicit case_list is required')
        self.img_list = read_names(case_list, image_dir, label_dir)
        self.img_num = len(self.img_list)
        self.mode = mode
        self.augmentor = augmentor
        self.iter_num = len(self.img_list)
        print('Mode: {}, Total number of samples to be generated: {}, Data normalization method: {}'.format(self.mode, self.iter_num, self.norm_method))
        print('load {} dataset to cache'.format(self.mode))
        self.cache_data = []
        if self.mode in ['test', 'test_final', 'test1', 'test2']:
            self.cache_label = []
        for indx in tqdm(range(self.iter_num)):
            img, lab = self.getitem(indx)
            if img.shape[-3:-1] != self.input_size[-3:-1]:
                img = torch.from_numpy(img).float().cuda()
                img = F.interpolate(img, self.input_size, mode='trilinear', align_corners=False)
                if self.mode in ['test', 'test_final', 'test1', 'test2']:
                    lab = torch.from_numpy(lab).float().cuda()
                    lab = F.interpolate(lab, self.input_size, mode='nearest')
                    lab = lab.cpu().numpy()
                    self.cache_label.append(lab)
                img = img.cpu().numpy()
                self.cache_data.append(img)
            else:
                if self.mode in ['test', 'test_final', 'test1', 'test2']:
                    self.cache_label.append(lab)
                self.cache_data.append(img)

    def __len__(self):
        return self.iter_num

    def __getitem__(self, index):
        if self.mode in ['test', 'test_final', 'test1', 'test2']:
            return (self.cache_data[index], self.cache_label[index])
        else:
            return (self.cache_data[index], self.cache_data[index])

    def getitem(self, index):
        if self.pair:
            fp = os.path.join(self.image_dir, self.img_list[index])
            if self.mode in ['test', 'test_final', 'test1', 'test2']:
                img_t1, lab_t1 = self.get_image(fp.replace('T2', 'T1'), index)
                img_t2, lab_t2 = self.get_image(fp, index)
                img = np.stack([img_t1, img_t2], axis=0)
                lab = np.stack([lab_t1, lab_t2], axis=0)
                return (img, lab)
            else:
                img_t1 = self.get_image(fp.replace('T2', 'T1'), index)
                img_t2 = self.get_image(fp, index)
                img = np.stack([img_t1, img_t2], axis=0)
                return (img, img)
        else:
            fp = os.path.join(self.image_dir, self.img_list[index])
            if self.mode in ['test', 'test_final', 'test1', 'test2']:
                img, lab = self.get_image(fp, index)
                return (img[None], lab[None])
            else:
                img = self.get_image(fp, index)
                return (img[None], img[None])

    def get_image(self, fp, index):
        img = sitk.Cast(sitk.ReadImage(fp), sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img).astype('float')
        data = {'image': img}
        if self.mode in ['test', 'test_final', 'test1', 'test2']:
            fp = os.path.join(self.label_dir, self.img_list[index])
            lab = sitk.ReadImage(fp)
            lab = sitk.GetArrayFromImage(lab)
            lab = lab > 0
            data['label'] = lab
        if self.augmentor is not None:
            data = self.augmentor(data)
        for key in data.keys():
            if key == 'image':
                img = data['image']
                if self.norm_method == 'clip':
                    img = np.clip(img, 0, 1)
                elif self.norm_method == 'none':
                    pass
                else:
                    raise ValueError('Unknown data normalization {}'.format(self.norm_method))
            elif key == 'label':
                lab = data['label']
        if self.mode in ['test', 'test_final', 'test1', 'test2']:
            return (img[None], lab[None])
        else:
            return img[None]
