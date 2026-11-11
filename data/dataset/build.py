import torch
import torch.utils.data
from .data_folder_for_DR import DataFolderDR_cache
from ..collate import collate_fn_train, collate_fn_test, worker_init_fn
from functools import partial

def get_dataset(args, mode='train', augmentor=None, num_data=32):
    if mode in ['train', 'valid']:
        image_dir = args.train_img_dir
        label_dir = None
        train_rate = args.train_rate
    elif mode in ['train1']:
        image_dir = args.train_img_dir1
        label_dir = None
        train_rate = args.train_rate
    elif mode in ['train2']:
        image_dir = args.train_img_dir2
        label_dir = None
        train_rate = args.train_rate
    elif mode in ['test']:
        image_dir = args.test_img_dir
        label_dir = args.test_lab_dir
        train_rate = 1
    elif mode in ['test1']:
        image_dir = args.test_img_dir1
        label_dir = args.test_lab_dir1
        train_rate = 1
    elif mode in ['test2']:
        image_dir = args.test_img_dir2
        label_dir = args.test_lab_dir2
        train_rate = 1
    elif mode == 'predict':
        image_dir = args.test_img_dir
        label_dir = None
        train_rate = 1
    dataset = DataFolderDR_cache(image_dir=image_dir, case_list=getattr(args, 'train_case_list' if mode == 'train' else 'monitor_case_list', None), label_dir=label_dir, mode=mode, augmentor=augmentor, train_rate=train_rate, norm_method=args.norm_method, num_data=num_data, pair=args.pair, input_size=args.input_size)
    return dataset

def build_dataloader(dataset, batch_size, mode='train', num_workers=4, seed=0):
    assert mode in ['train', 'valid', 'test', 'predict']
    if mode in ['train', 'valid']:
        cf = collate_fn_train
    else:
        cf = collate_fn_test
    generator = torch.Generator()
    generator.manual_seed(seed)
    img_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=mode == 'train', collate_fn=cf, num_workers=num_workers, pin_memory=True, drop_last=mode == 'train', worker_init_fn=partial(worker_init_fn, seed=seed), generator=generator)
    return img_loader
