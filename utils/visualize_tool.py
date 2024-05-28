import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from utils.visualize_grad import VanillaBackprop
from utils.get_data import *
from utils.util_functions import *


def visualize_tensor(tensor, isnomalized=True):
    numpy = tensor.numpy().transpose(1, 2, 0)
    if isnomalized :
        numpy = (numpy - numpy.min()) / (numpy.max() - numpy.min())
    plt.imshow(numpy)
    plt.show()


def plot_filters(model_info, tensor, layer_name, conf, save=True, show=False, ncols=32 , nchannel=5):
    n, _, _, _ = tensor.shape
    nrows = n // ncols + (1 if n % ncols else 0)

    for c in range(nchannel):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
        for i in range(n):
            ax = axes[i // ncols, i % ncols]
            kernel = tensor[i, c, :, :] if n > 1 else tensor[i, :, :]
            ax.imshow(kernel, cmap='viridis')
            ax.axis('off')
        alias = get_alias(layer_name, model_info, conf['model'])
        plt.suptitle('Layer: '+alias+" (# channel: "+str(c)+")")
        if save :
            save_plot_result(fig, "VisFilter", '#layer_'+alias+"_#channel_"+str(c), conf)
        if show :
            plt.show()
        else:
            plt.close()


def visualize_filters(model_info, model, layer_name, conf=None, save=True, show=False, ncols=32, nchannel=5, showAll=False):
    weights = get_weights(model, layer_name)
    n, c, w, h = weights.shape
    if not showAll:
        c = min(c, nchannel)
    plot_filters(model_info, weights, layer_name, conf, save, show, ncols, c)



def visualize_feature_map(activation, model, input, layer_name, conf=None, save=False, show=True, ncols=32):
    act = get_feature_map(activation, model, input, layer_name)
    # print(act.shape)
    nrows = max(act.size(0) // ncols, 1) 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols, nrows))
    for i in range(act.size(0)):
        ax = axes[i // ncols, i % ncols]
        kernel = act[i]
        ax.imshow(kernel, cmap='viridis')
        ax.axis('off')
    plt.suptitle('Layer: '+layer_name)
    if save :
        save_plot_result(fig, "VisActivation", '#layer_'+layer_name, conf)
    if show :
        plt.show()
    else:
        plt.close()


def visualize_weight_distribution(model, conf=None, save=False, show=True, violin_sample=1000):
    
    means, variances, weight_df = get_numerical_weight(model)
    
    plt.figure(figsize=(7, 4))
    plt.plot(means, label='Mean')
    plt.plot(variances, label='Variance')
    plt.xlabel('conv layer idx')
    plt.ylabel('value')
    plt.title('Weights Statistics')
    plt.legend()
    if save :
        fig = plt.gcf()
        save_plot_result(fig, "VisWeightDist", 'statistics', conf)
    if show :
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(7, 4))
    sampled_df = weight_df.groupby('Layer').apply(lambda x: x.sample(violin_sample)).reset_index(drop=True)
    plt.xlabel('conv #')
    plt.ylabel('weights')
    plt.xticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'])
    sns.set()
    sns.set_theme(style="darkgrid")
    sns.violinplot(x='Layer', y='Weights', data=sampled_df)
    if save :
        fig = plt.gcf()
        save_plot_result(fig, "VisWeightDist", 'violinplot', conf)
    if show :
        plt.show()
    else:
        plt.close() 



def visualize_class_activation_images(org_img, activation_map, conf, layer_num, show=True, save=False):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """ 
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
        
    if save :
        save_plot_result(heatmap, "LayerCAM", 'Heatmap_#layer_'+str(layer_num), conf, isimage=True)
        save_plot_result(heatmap_on_image, "LayerCAM", 'Grad_On_Image_#layer_'+str(layer_num), conf, isimage=True)
        save_plot_result(activation_map, "LayerCAM", 'Grayscale_#layer_'+str(layer_num), conf, isimage=True)
    
    if show:
        images = [heatmap, heatmap_on_image, activation_map]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        for ax, img in zip(axes, images):
            ax.imshow(img)
            ax.axis('off') 
        plt.show()
    else :
        plt.close() 


def visualize_gradXimage(prep_img, target_class, model, conf, show=True, save=False):
    VBP = VanillaBackprop(model)
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)

    grad_times_image = vanilla_grads * prep_img.detach().numpy()[0]
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    save_gradient_images(grayscale_vanilla_grads, conf, show, save)


def visualize_feature_distribution(embedding, labels, preds, conf, layer_name, num_class=30, show=True, save=False):
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    df['labels'] = labels
    df['preds'] = preds
    
    df['labels'] = pd.to_numeric(df['labels'], errors='coerce')
    if (10 < num_class) and (conf['dataset'] == "cifar10" or conf['dataset'] == "svhn"):
        num_class = 10

    filtered_df = df[df['labels'] < num_class]
    
    sns.set()
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=filtered_df, x='x', y='y', hue='preds', style='labels', palette='bright', legend=False)
    plt.title("Feature Distribution")
    if save:
        fig = plt.gcf()
        save_plot_result(fig, "FeatureDist", '#layer_'+layer_name, conf)
    if show :
        plt.show()
    else :
        plt.close() 


def plot_comparison_each_dataset_only_two(df_dataset, show=True, save=False):
    datasets = ['cifar100', 'cars', 'cub']
    keys = ['best_train_acc', 'best_train_loss', 'best_val_acc', 'best_val_loss']
    onlytwo = ['11000', '10100', '10010', '10001', '01100', '01010', '01001', '00110', '00101', '00011']
    
    for dataset in datasets:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        axes = axes.flatten()
        group_df = df_dataset[dataset]
        for i, key in enumerate(keys) :
            row_values = group_df.loc[key]
            if key in ['best_train_acc', 'best_val_acc'] : accloss = 'Accuracy'
            else : accloss = 'Loss'
            
            filtered_values = row_values[row_values.index.isin(onlytwo)]
            
            heatmap_data = np.zeros((5, 5))
            # print(filtered_values.index, filtered_values.values)
            for idx, value in zip(filtered_values.index, filtered_values.values):
                row_idx = idx.index('1') 
                col_idx = idx.index('1', row_idx + 1)
                heatmap_data[row_idx, col_idx] = value
                heatmap_data[col_idx, row_idx] = value
            
            df_heatmap = pd.DataFrame(heatmap_data, index=[f"B{i+1}" for i in range(5)], columns=[f"B{i+1}" for i in range(5)])
            df_heatmap.fillna(0, inplace=True)
            df_heatmap[df_heatmap == 0] = np.nan
            mask = np.isnan(heatmap_data)
            
            sns.set()
            sns.set_theme(style="white")
            sns.heatmap(ax=axes[i], data=df_heatmap, annot=True, fmt=".3f", cmap='viridis', annot_kws={"size": 10}, mask=mask)
            axes[i].set_title("2-layer fintuning "+str(key)+" on "+str(dataset))
        
        if save:
            save_plot_result(fig, "Comparison", "2-layer Finetuning profiles on "+str(dataset))
        if show :    
            plt.show()
        else :
            plt.close() 



def plot_comparison_each_dataset_only_one(df_dataset, show=True, save=False):
    datasets = ['cifar100', 'cars', 'cub']
    keys = ['best_train_acc', 'best_train_loss', 'best_val_acc', 'best_val_loss']
    onleyone = ['00001', '00010', '00100', '01000', '10000']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12)) 
    axes = axes.flatten()

    for dataset in datasets:
        group_df = df_dataset[dataset]
        for i, key in enumerate(keys) :
            row_values = group_df.loc[key]
            if key in ['best_train_acc', 'best_val_acc'] : accloss = 'Accuracy'
            else : accloss = 'Loss'
            
            filtered_values = row_values[row_values.index.isin(onleyone)]
            
            long_form = pd.DataFrame({
                'Freezing': filtered_values.index,
                accloss : filtered_values.values
            })
            
            sns.set()
            sns.set_theme(style="darkgrid")
            sns.set_palette("muted")
            #sns.set_theme(palette="viridis")
            sns.lineplot(ax=axes[i], data=long_form, x='Freezing', y=accloss, label=dataset, marker='o', dashes=True)
            
            axes[i].set_title("Comparison of "+str(key))
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend()
    plt.tight_layout()
    if save:
        save_plot_result(fig, "Comparison", "Finetuning profiles for different select Only one layer and datasets")
    if show :    
        plt.show()
    else :
        plt.close() 


def plot_comparison_each_dataset(df_dataset, show=True, save=False):
    datasets = ['cifar100', 'cars', 'cub']
    keys = ['best_train_acc', 'best_train_loss', 'best_val_acc', 'best_val_loss']
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12)) 
    axes = axes.flatten()

    for dataset in datasets:
        group_df = df_dataset[dataset]
        for i, key in enumerate(keys) :
            row_values = group_df.loc[key]
            if key in ['best_train_acc', 'best_val_acc'] : accloss = 'Accuracy'
            else : accloss = 'Loss'
            
            long_form = pd.DataFrame({
                'Freezing': row_values.index,
                accloss : row_values.values
            })
            
            sns.set()
            sns.set_theme(style="darkgrid")
            sns.set_palette("muted")
            #sns.set_theme(palette="viridis")
            sns.lineplot(ax=axes[i], data=long_form, x='Freezing', y=accloss, label=dataset, marker='o', dashes=True)
            
            axes[i].set_title("Comparison of "+str(key))
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend()
    plt.tight_layout()
    if save:
        save_plot_result(fig, "Comparison", "Finetuning profiles for different select layer and datasets")
    if show :    
        plt.show()
    else :
        plt.close() 