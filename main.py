import torch, argparse
from pyprnt import prnt
import numpy as np

from dataset_dir.datasets import datasetload

from utils.utils import *
from model.pretrained_models import select_model
from Trainer.Standard_Trainer import Standard_Trainer
from Trainer.AutoLR_Trainer import AutoLR_Trainer

def arg_parse(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='Dataset type')
    parser.add_argument('--model', default='alexnet', help='Model type')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--pretrain', type=str2bool, nargs='?', const=True, default=False, help="Pretrain")
    parser.add_argument('--mode', type=str, default='ours', help='Standard(standard), Ours(ours), AutoLR(auto)')
    
    parser.add_argument('--epoch', type=int, default=50, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')

    parser.add_argument('--model_path', type=str, default='', help='pretrained model path')

    parser.add_argument('--max_f', default=0.4, type=float, help='max_f')
    parser.add_argument('--min_f', default=2, type=float, help='min_f')

    return parser.parse_args()

if __name__ == '__main__':
    # arguments parsing
    args = arg_parse(argparse.ArgumentParser())
    
    # random seed
    np.random.seed(2023)
    torch.manual_seed(2023)
    
    # cuda device
    conf = dict()
    conf['device'] = torch.device("cuda:" + str(args.device))
    conf = dict(conf, **args.__dict__)
    
    # dataset load
    trainloader, validloader, testloader, num_class = datasetload(conf['dataset'], conf['batch_size'])
    
    print()
    prnt(conf)

    model = select_model(conf['model'], num_class, pretrained=conf['pretrain'], checkpt=conf['model_path'])

    checkpt, board_name, writer, setting = set_loggers(conf)

    # print('Experiment Setting:', setting, '|Croess-Entropy Loss|SGD optimizer')

    if conf['mode'] == 'standard':
        trainer = Standard_Trainer(model, conf['device'], trainloader, validloader, testloader, checkpt, board_name, writer)
    elif conf['mode'] == 'auto':
        trainer = AutoLR_Trainer(model, conf['device'], trainloader, validloader, testloader, checkpt, board_name, writer, conf['max_f'], conf['min_f'])
    start_time, end_test_time = trainer.train_model(conf['epoch'], conf['lr'])

    print("'\nEntire Training finish with,")
    prnt(conf)
    print("\nStart:\t", start_time)
    print("End:\t", end_test_time)