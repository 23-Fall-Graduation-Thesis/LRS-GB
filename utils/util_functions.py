import os
import re
import numpy as np
import copy
import torch
from PIL import Image
import umap.umap_ as umap
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
from numba import NumbaDeprecationWarning
import re

warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)


# for prototype
def name_parser(path):
    """object: 주어진 경로로부터 모델의 특징을 추출 
    input: path (log or weight)
    ouput: mode <- str, model <- str, dataset <- str, freeze <- str"""
    parts = path.split("/")[-1].split("_")
    
    mode_ins = parts[2].split(".")[0]
    model_ins = parts[0]
    dataset_ins = parts[1]
    
    conf = dict(
        mode=mode_ins, 
        model=model_ins, 
        dataset=dataset_ins, 
    )
    return conf


def get_alias(layer_name, model_info, model="alexnet"):
    if model == "alexnet":
        idx = int(layer_name.split(".")[1])
        key_list = [key for key, value in model_info.items() if value == idx]
        return key_list[0]
    elif model == "resnet18" or model == "resnet50":
        if layer_name == 'conv1':
            idx = 0
        else:
            idx = int(layer_name.split(".")[0][-1]) * 10 + int(layer_name.split(".")[1])
        key_list = [key for key, value in model_info.items() if value == idx]
        return key_list[0]
    else:
        return -1


def tensor_to_ndarray(tensor, isnomalized=True):
    ndarray = tensor.numpy().transpose(1, 2, 0)
    if isnomalized :
        ndarray = (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())
    return ndarray


def tensor_to_img(tensor, isnomalized=True):
    ndarray = tensor_to_ndarray(tensor, isnomalized)
    ndarray = (ndarray * 255).astype(np.uint8)
    image = Image.fromarray(ndarray)
    return image


def get_info(data):
    img_tensor = data[0]
    target_class = data[1]
    origin_img = tensor_to_img(img_tensor)
    prep_img = preprocess_image(origin_img)
    
    return origin_img, prep_img, target_class


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_gradient_images(gradient, conf, show=True, save=False):
    """objective: Exports the original gradient image
    - input: 
        - gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        - for file_name... (dataset, freeze) (str): File name to be exported
    """
    gradient = gradient - gradient.min()
    if save : 
        save_plot_result(gradient, "GradXImage", "vanila_grad", conf, isimage=True)
    if show:
        plt_gradient = np.squeeze(gradient)
        plt.imshow(plt_gradient, cmap='gray')
        plt.axis('off')
        plt.show()


def apply_colormap_on_image(org_im, activation, colormap_name):
    """objective: Apply heatmap on image
    - input: org_img (PIL img): Original image; activation_map (numpy arr): Activation map (grayscale) 0-255; colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def apply_heatmap(R, sx, sy):
    """objective: Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial
    """
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    heatmap = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    plt.close() 
    return heatmap
    # plt.show()


def preprocess_image(pil_im, resize_im=True):
    """objective: Processes image for CNNs
    - input: PIL_img (PIL_img): PIL Image or numpy array to process; resize_im (bool): Resize to 224 or not
    - output: im_as_var (torch variable): Variable that contains processed float tensor
    """
    # Mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def get_umap_embedding(features, n_components=2, n_neighbors=10, min_dist=0.5, metric="cosine", random_state=50):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    embedding = reducer.fit_transform(features)
    
    return embedding


def save_plot_result(fig, vis_method, alias, conf=None, isimage=False):
    # vis_method: 
    directory_path = "./results/figure/"
    if conf is not None:
        directory_path = os.path.join(directory_path, conf['mode'], conf['model'], conf['dataset'], vis_method)
    else :
        directory_path = os.path.join(directory_path, vis_method)
        
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    file_name = f"{alias}.png"
    
    file_path = os.path.join(directory_path, file_name)
    if not isimage:
        fig.savefig(file_path)
    else :
        save_image(fig, file_path)
    
    print(f"Plot saved to {file_path}")
    plt.close()