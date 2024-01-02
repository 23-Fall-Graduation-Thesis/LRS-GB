import argparse, os, torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
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
    # experiment setting values
    setting = "epoch"+str(conf['epoch'])+"_lr"+str(conf['lr'])
    if conf['pretrain']:
        # make pretrained model
        checkpt = f"./model/weight/pretrain/{conf['model']}/{conf['dataset']}_{setting}.pt"
        board_name = str(conf['model'])+"/"+str(conf['dataset'])+"_"+setting
        writer = SummaryWriter("./results/log/pretrain/"+board_name)
        
        print('model:', conf['model'], ' dataset:', conf['dataset'], 'pretrain: ', conf['pretrain'])
    else:
        # save name setting for fine-tuning model
        checkpt = f"./model/weight/{conf['mode']}/{conf['model']}/{conf['dataset']}_{setting}.pt"
        board_name = f"{conf['mode']}/{conf['model']}/{conf['dataset']}_{setting}"
        writer = SummaryWriter(f"./results/log/{board_name}")
        os.makedirs(f"./model/weight/{conf['mode']}/{conf['model']}", exist_ok=True)
        
        print('model:', conf['model'], ' dataset:', conf['dataset'], 'fine-tuning mode:', conf['mode'])
        
    return checkpt, board_name, writer, setting

def set_optimizer(model, pretrain, mode):
    if mode == 'standard' or pretrain:
        optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['lr'])
    elif mode == 'auto':
        pass
    elif mode == 'ours':
        pass
    else:
        raise ValueError(f'Invalid finetune mode input.')

def get_size_scalar(torch_tensor):
    return np.prod(np.array(torch_tensor.shape))

def compute_weight_variation(modelA, modelB, layer_names):
    scale = 1000000

    L1_varation = []

    # if model_name in ['resnet18', "resnet34", "resnet50", "resnet101", "resnet152", "WRN50", "WRN101"]:
    for layer_name in layer_names:
        current_layerA = modelA
        current_layerB = modelB
        layer_name_split = layer_name.split(".")
        for name in layer_name_split:
            if name.isdigit():
                # 정수로 변환하여 순차적으로 인덱스로 접근
                index = int(name)
                current_layerA = current_layerA[index]
                current_layerB = current_layerB[index]
            else:
                # 그 외의 경우에는 getattr 사용
                current_layerA = getattr(current_layerA, name)
                current_layerB = getattr(current_layerB, name)

        if len(list(current_layerA.children())) == 0:
            # layer 안에 자식이 없는 경우
            if ('conv' in str(current_layerA).lower() or 'linear' in str(current_layerA).lower()) and hasattr(current_layerA, 'weight'):
                nweight = get_size_scalar(current_layerA.weight)
                variation = torch.pow(torch.norm(current_layerA.weight.cpu() - current_layerB.weight.cpu(), 2), 2) # TODO AutoLR 확인 필요 
                L1_varation.append(variation.detach().numpy()/nweight*scale)
        else:
            nweight, variation = 0
            for (sub_layerA, sub_paramA), (sub_layerB, sub_paramB) in zip(current_layerA.named_children(), current_layerB.named_children()):
                if  ('conv' in str(sub_layerA).lower() or 'linear' in str(sub_layerA).lower()) and hasattr(sub_paramA, 'weight'):
                    nweight += get_size_scalar(sub_paramA.weight)
                    variation += torch.pow(torch.norm(sub_paramA.weight.cpu() - sub_paramB.weight.cpu(), 2), 2)
                L1_varation.append(variation.detach().numpy()/nweight*scale)

    return L1_varation
