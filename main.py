import torch, argparse
from pyprnt import prnt
import numpy as np

from dataset_dir.datasets import datasetload

from utils.utils import *
from model.pretrained_models import select_model
from trainer.Standard_Trainer import Standard_Trainer
from trainer.AutoLR_Trainer import AutoLR_Trainer
from trainer.GB_with_Weva_Trainer import GB_with_Weva_Trainer
import random

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    # Common Options
    parser.add_argument('--dataset', default='cifar100', help='Dataset type')
    parser.add_argument('--model', default='resnet18', help='Model type')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=30, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--pretrain', type=str2bool, nargs='?', const=True, default=False, help="Pretrain")

    # Fine-tuning Options
    parser.add_argument('--mode', type=str, default='standard', help='Standard(standard), LRS-GB(GB), AutoLR(auto), Auto-start-GB(autoGB), GB-with-weva(GBweva)')
    parser.add_argument('--model_path', type=str, default='', help='pretrained model path')

    ## AutoLR Options
    parser.add_argument('--max_f', default=0.05, type=float, help='max_f for AutoLR')
    parser.add_argument('--min_f', default=1.0, type=float, help='min_f for AutoLR')
    parser.add_argument('--thr_score', default=0.94, type=float, help='score threshold for AutoLR')

    ## LRS-GB with target weva Options
    parser.add_argument('--K', default=7.8, type=float, help='Lipschitz constant') # TODO: add head k
    parser.add_argument('--scale_factor', default=1.27, type=float, help='layer-wise constraint scaling')
    parser.add_argument('--thr_init_score', default=0.97, type=float, help='score threshold for LRS')
    parser.add_argument('--target_func', default='constant', type=str, help='constant, linear, inverse, cosine, step')
    parser.add_argument('--norm', type=str, default='L2', help='weight calculation using L1 norm or L2 norm')

    # parser.add_argument('--bound', default='diff', type=str, help='diff or weva') # select weva
    # parser.add_argument('--increase_bound', type=str2bool, default=False, help='') # delete GB with increasing bound
    # parser.add_argument('--inc_type', default='log1', type=str, help='increase_bound type - linear(lin1, lin2), log(log1, log2), step')
    
    parser.add_argument('--opt', type=str2bool, default=False, help='using hyperopt?')
    
    return parser.parse_args()

def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # arguments parsing
    args = arg_parse(argparse.ArgumentParser())
    # random seed
    set_seed()
    
    # cuda device
    conf = dict()
    conf['device'] = torch.device("cuda:" + str(args.device))
    conf = dict(conf, **args.__dict__)
    
    # dataset load
    trainloader, validloader, testloader, num_class = datasetload(conf['dataset'], conf['batch_size'])
    
    if not args.opt:
        print()
        prnt(conf)

    checkpt, board_name, log_time = set_loggers(conf)

    model = select_model(conf['model'], num_class, pretrained_model=(not conf['pretrain']), checkpt=conf['model_path'])


    if conf['mode'] == 'standard' or conf['pretrain']:
        trainer = Standard_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    elif conf['mode'] == 'auto':
        trainer = AutoLR_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    # elif conf['mode'] == 'GB':
    #     trainer = LRS_GB_Score_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    # elif conf['mode'] == 'autoGB':
    #     trainer = Auto_Start_GB_Score_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    elif conf['mode'] == 'GBweva':
        trainer = GB_with_Weva_Trainer(model, conf, (trainloader, validloader, testloader), (checkpt, board_name, log_time))
    else :
        pass
    
    start_time, end_test_time = trainer.train_model(conf['epoch'], conf['lr'])

    if not args.opt:
        print("\nEntire Training finish with,")
        prnt(conf)
        print("\nStart:\t", start_time)
        print("End:\t", end_test_time)