import argparse, os, time
from torch.utils.tensorboard import SummaryWriter
from scheduler.algorithm.Condition import *
from scheduler.algorithm.TargetLR import *
from scheduler.algorithm.TargetWeva import *
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_loggers(conf):
    log_time = time.strftime('%m%d-%H%M')

    # experiment setting values
    if conf['pretrain']:
        # make pretrained model
        board_name = f"pretrain/{conf['model']}/{conf['dataset']}_lr{conf['lr']}"
        os.makedirs(f"./model/weight/pretrain/{conf['model']}", exist_ok=True)
        checkpt = f"./model/weight/pretrain/{conf['model']}/{conf['dataset']}_{log_time}.pt"

        print('model:', conf['model'], ' dataset:', conf['dataset'], 'pretrain: ', conf['pretrain'])
    else:
        if conf['mode']=='standard':
            setting = f"lr{conf['lr']}"
        elif conf['mode']=='GB':
            if conf['increase_bound']:
                setting = f"lr{conf['lr']}/K{round(conf['K'],3)}_scale{round(conf['scale_factor'],3)}_{conf['bound']}_{conf['thr_init_score']}_increase"
            else:
                setting = f"lr{conf['lr']}/K{round(conf['K'],3)}_scale{round(conf['scale_factor'],3)}_{conf['bound']}_{conf['thr_init_score']}"
        elif conf['mode']=='auto':
            setting = f"lr{conf['lr']}/max{round(conf['max_f'],3)}_min{round(conf['min_f'],3)}"
        elif conf['mode']=='autoGB':
            setting = f"lr{conf['lr']}/max{round(conf['max_f'],3)}_min{round(conf['min_f'],3)}_K{round(conf['K'],3)}_scale{round(conf['scale_factor'],3)}_{conf['bound']}_{conf['thr_init_score']}"
            if conf['increase_bound']:
                setting += '_increase'

        os.makedirs(f"./model/weight/{conf['mode']}/{conf['model']}", exist_ok=True)
        checkpt = f"./model/weight/{conf['mode']}/{conf['model']}/{conf['dataset']}_{log_time}.pt"
        board_name = f"{conf['mode']}/{conf['model']}/{conf['dataset']}_{setting}"

        print('model:', conf['model'], ' dataset:', conf['dataset'], 'fine-tuning mode:', conf['mode'])

    return checkpt, board_name, log_time


def get_size_scalar(torch_tensor):
    return np.prod(np.array(torch_tensor.shape))


def get_class(class_name):
    try:
        cls = globals()[class_name]
        return cls
    except KeyError:
        raise ValueError(f"'{class_name}' is not exist.")


def get_instance(class_name, *args, **kwargs):
    try:
        cls = globals()[class_name]
        return cls(*args, **kwargs)
    except KeyError:
        raise ValueError(f"'{class_name}' is not exist.")