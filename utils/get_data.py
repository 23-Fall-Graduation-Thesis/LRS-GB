import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

def get_weights(model, layer_name):
    flag = False
    for name, layer in model.named_modules():
        if name == layer_name:
            flag = True
            weights = layer.weight.data.clone()
            break
    if not flag :
        print(f'Undefined Layer.')
        return -1
    return weights


def get_feature_map(activation, model, input, layer_name):
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        flag = False
        if flag == False:
            if name == layer_name:
                flag = True
                layer.register_forward_hook(get_activation(layer_name))
                break
        else:
            print(f"Layer {layer_name} not found in the model.")
    model.eval()
    output = model(input)
    act = activation[layer_name].squeeze()
    return act


def get_numerical_weight(model):
    means = []
    variances = []
    weights_data = []
    layer_index = []

    conv_layer_num = 0

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_layer_num += 1
            weights = layer.weight.data.cpu().numpy().flatten()
            means.append(weights.mean().item())
            variances.append(weights.var().item())
            weights_data.extend(weights)
            layer_index.extend([conv_layer_num] * len(weights))

    weight_df = pd.DataFrame({'Layer': layer_index, 'Weights': weights_data})

    return means, variances, weight_df


def get_feature_from_dataset(MODEL, batch_size, testloader, layer_name):
    
    features = None
    labels = None
    preds = None
    if batch_size > len(testloader.dataset) :
        batch_size = len(testloader.dataset)
    shuffle_testloader = DataLoader(testloader.dataset, batch_size=batch_size, shuffle=True)
    
    with torch.no_grad():
        MODEL.eval()
        count = 0
        for data, target in shuffle_testloader:
            count += 1
            activation = {}
            feature = get_feature_map(activation, MODEL, data, layer_name)
            features = feature.view(batch_size, -1)
            labels = target
            output = MODEL(data)
            preds = output.argmax(dim=1, keepdim=True).squeeze()
            if count != 0 :
                break
    
    return features, labels, preds


def get_performance_df(BEST_VALUES):
    grouped_data = {}
    for key, value in BEST_VALUES.items():
        group_key, sub_key = key.split('-')
        if group_key not in grouped_data:
            grouped_data[group_key] = {}
        grouped_data[group_key][sub_key] = value

    df_dataset = {}
    for group_key, values in grouped_data.items():
        df_dataset[group_key] = pd.DataFrame(values)

    grouped_data = {}
    for key, value in BEST_VALUES.items():
        sub_key, group_key = key.split('-')
        if group_key not in grouped_data:
            grouped_data[group_key] = {}
        grouped_data[group_key][sub_key] = value
    
    df_freezing = {}
    for group_key, values in grouped_data.items():
        df_freezing[group_key] = pd.DataFrame(values)

    return df_dataset, df_freezing