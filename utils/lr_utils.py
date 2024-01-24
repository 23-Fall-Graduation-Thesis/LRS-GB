import torch
import numpy as np

SCALE = 1000000

def get_size_scalar(torch_tensor):
    return np.prod(np.array(torch_tensor.shape))

def compute_weight_variation(modelA, modelB, layer_name_dict):
    L1_variation = [[0, 0] for _ in range(max(layer_name_dict.values())+1)]
    
    for (layer_name, paramA), (_, paramB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        curA = modelA # model_temp
        curB = modelB # model_try
        layer_name_splited = layer_name.split('.')[:-1] # .weight, .bias 제거 (이름만 확인)
        for idx, name in enumerate(layer_name_splited):
            if name.isdigit():
                # 정수로 변환하여 순차적으로 인덱스로 접근
                curA = curA[int(name)]
                curB = curB[int(name)]
            else:
                # 그 외의 경우에는 getattr 사용
                curA = getattr(curA, name)
                curB = getattr(curB, name)
            cur_name = '.'.join(layer_name_splited[:idx+1])
            if cur_name in layer_name_dict:
                var_idx = layer_name_dict.get(cur_name)
        if (type(curA) in [torch.nn.Conv2d, torch.nn.Linear]) and ('downsample' not in layer_name):
            L1_variation[var_idx][0] += get_size_scalar(paramA)
            L1_variation[var_idx][1] += torch.pow(torch.norm(paramA.cpu() - paramB.cpu(), 2), 2).detach().numpy()
    
    L1_variation_calc = [(x[1]**0.5/x[0]*SCALE) for x in L1_variation]

    return L1_variation_calc

#TODO:
def compute_weight_difference_and_variation(modelA, modelB, layer_name_dict):
    L1_variation = [[0, 0] for _ in range(max(layer_name_dict.values())+1)]
    distance_calc = [[] for _ in range(max(layer_name_dict.values())+1)]
    param_num_list = [0 for _ in range(max(layer_name_dict.values())+1)]
    for (layer_name, paramA), (_, paramB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        curA = modelA
        curB = modelB
        layer_name_splited = layer_name.split('.')[:-1] # .weight, .bias 제거 (이름만 확인)
        for idx, name in enumerate(layer_name_splited):
            if name.isdigit():
                # 정수로 변환하여 순차적으로 인덱스로 접근
                curA = curA[int(name)]
                curB = curB[int(name)]
            else:
                # 그 외의 경우에는 getattr 사용
                curA = getattr(curA, name)
                curB = getattr(curB, name)
            cur_name = '.'.join(layer_name_splited[:idx+1])
            if cur_name in layer_name_dict:
                var_idx = layer_name_dict.get(cur_name)
        if (type(curA) in [torch.nn.Conv2d, torch.nn.Linear]) and ('downsample' not in layer_name):
            weight_distance = paramA.cpu() - paramB.cpu()
            distance_calc[var_idx].append(weight_distance)
            size = get_size_scalar(paramA)
            L1_variation[var_idx][0] += size
            param_num_list[var_idx] += size
            L1_variation[var_idx][1] += torch.pow(torch.norm(weight_distance, 2), 2).detach().numpy()
    
    L1_variation_calc = [(x[1]**0.5/x[0]*SCALE) for x in L1_variation]
    
    return L1_variation_calc, distance_calc, param_num_list

def diff_to_weva(diff_list, n):
    temp = 0
    for diff in diff_list:
        temp += torch.pow(torch.norm(diff, 2), 2).detach().numpy()
    weva = temp**0.5/n*SCALE
    
    return weva

def layer_block_info(model_name):
    if model_name == "alexnet":
        # return [['features.0'], ['features.3'], ['features.6'], ['features.8'], ['features.10'], 
        #         ['classifier.1'], ['classifier.4'], ['classifier.6']]
        return [['features.0'], ['features.3'], ['features.6'], ['features.8'], ['features.10'], 
                ['classifier']]
    elif model_name == "resnet18":
        return [['conv1', 'bn1'], ['layer1.0'], ['layer1.1'], ['layer2.0'], ['layer2.1'],
                ['layer3.0'], ['layer3.1'], ['layer4.0'], ['layer4.1'], ['fc']]
    elif model_name == "resnet50":
        return [['conv1', 'bn1'], ['layer1.0'], ['layer1.1'], ['layer1.2'], ['layer2.0'], ['layer2.1'],
                ['layer2.2'], ['layer2.3'], ['layer3.0'], ['layer3.1'], ['layer3.2'], ['layer3.3'],
                ['layer3.4'], ['layer3.5'], ['layer4.0'], ['layer4.1'], ['layer4.2'], ['fc']]
        

def get_num_layer(model_name):
    if model_name == "alexnet":
        return 5
    elif model_name == "resnet18":
        return 9
    elif model_name == "resnet50":
        return 17
    

def get_frob_norm(w):
    return torch.sqrt(torch.sum(torch.pow(w, 2.0)))

def get_lone_norm(w):
    if len(w.shape) != 2 and len(w.shape) != 4:
        assert len(w.shape) == 1
        return torch.sum(torch.abs(w))
    
    axes=1
    if len(w.shape) == 4:
        axes=[1, 2, 3]
    return torch.sum(torch.abs(w), dim=axes, keepdims=True)

def get_linf_norm(w):
    if len(w.shape) != 2 and len(w.shape) != 4:
        assert len(w.shape) == 1
        return torch.max(torch.abs(w))
    
    axes=1
    if len(w.shape) == 4:
        axes=[1, 2, 3]
    return torch.max(torch.sum(torch.abs(w), dim=axes))
