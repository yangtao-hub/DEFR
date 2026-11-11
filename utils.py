import os
import shutil
import numpy as np
import torch
import yaml
import random

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class AverageMeters(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg_dict = {}
        self.sum_dict = {}
        self.cnt_dict = {}

    def update(self, val_dict, n=1):
        for key in val_dict.keys():
            if key not in self.avg_dict.keys():
                self.sum_dict[key] = val_dict[key]
                self.cnt_dict[key] = n
            else:
                self.sum_dict[key] += val_dict[key]
                self.cnt_dict[key] += n
            self.sum_dict[key] = self.sum_dict[key] / self.cnt_dict[key]

def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst = os.path.join(path, 'scripts', os.path.basename(script))
            if os.path.isdir(script):
                shutil.copytree(script, dst)
            else:
                shutil.copyfile(script, dst)

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_learnable_para(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_size = sum([np.prod(p.size()) for p in model_parameters])
    return param_size

def write_to_yaml(args, filename):
    params = vars(args)
    with open(filename, 'w') as f:
        yaml.safe_dump(params, f, default_flow_style=False, sort_keys=False)

def save_checkpoint(path, model, epoch, optimizer, **kwargs):
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), **kwargs}, path)
