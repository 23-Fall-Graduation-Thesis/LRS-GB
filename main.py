import torch, argparse
from pyprnt import prnt
import numpy as np

from dataset_dir.datasets import datasetload

from utils.utils import *
from model.pretrained_models import select_model
from trainer.Standard_Trainer import Standard_Trainer
from trainer.AutoLR_Trainer import AutoLR_Trainer
from trainer.LRS_GB_Trainer import *
import random

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    # Common Options
    parser.add_argument('--dataset', default='cifar10', help='Dataset type')
    parser.add_argument('--model', default='resnet18', help='Model type')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--pretrain', type=str2bool, nargs='?', const=True, default=False, help="Pretrain")

    # Fine-tuning Options
    parser.add_argument('--mode', type=str, default='standard', help='Standard(standard), LRS-GB(GB), AutoLR(auto)')
    parser.add_argument('--model_path', type=str, default='', help='pretrained model path')

    parser.add_argument('--max_f', default=0.05, type=float, help='max_f for AutoLR')
    parser.add_argument('--min_f', default=1.0, type=float, help='min_f for AutoLR')
    parser.add_argument('--thr_score', default=0.94, type=float, help='score threshold for AutoLR')
    parser.add_argument('--thr_init_score', default=0.9, type=float, help='score threshold for LRS')
    parser.add_argument('--K', default=7.80246991703043, type=float, help='Lipschitz constant') # TODO: add head k
    parser.add_argument('--scale_factor', default=1.27679969876201, type=float, help='layer-wise constraint scaling')
    parser.add_argument('--max_trial', default=20, type=int, help='trial maximum for GB lr update')

    parser.add_argument('--k_multiply', default=1, type=float, help='')
    parser.add_argument('--scale_multiply', default=1, type=float, help='')

    parser.add_argument('--norm', type=str, default='L2', help='weight calculation using L1 norm or L2 norm')
    
    parser.add_argument('--use_AutoLR', type=str2bool, default=False, help='when you using LRS_GB, both using AutoLR Condition?')
    
    return parser.parse_args()

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # arguments parsing
    args = arg_parse(argparse.ArgumentParser())
    
    # random seed
    # np.random.seed(2023)
    # torch.manual_seed(2023)
    set_seed()
    
    # cuda device
    conf = dict()
    conf['device'] = torch.device("cuda:" + str(args.device))
    conf = dict(conf, **args.__dict__)
    
    # dataset load
    trainloader, validloader, testloader, num_class = datasetload(conf['dataset'], conf['batch_size'])

    conf['K'] *= conf['k_multiply']
    conf['scale_factor'] *= conf['scale_multiply']
    
    print()
    prnt(conf)

    checkpt, board_name, log_time = set_loggers(conf)

    model = select_model(conf['model'], num_class, pretrained_model=(not conf['pretrain']), checkpt=conf['model_path'])


    if conf['mode'] == 'standard' or conf['pretrain']:
        trainer = Standard_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    elif conf['mode'] == 'auto':
        trainer = AutoLR_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    elif conf['mode'] == 'GB':
        trainer = LRS_GB_Score_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    else :
        pass
    
    start_time, end_test_time = trainer.train_model(conf['epoch'], conf['lr'])

    print("'\nEntire Training finish with,")
    prnt(conf)
    print("\nStart:\t", start_time)
    print("End:\t", end_test_time)