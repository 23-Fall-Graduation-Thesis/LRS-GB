import numpy as np
import torch
import torch.nn as nn
from PIL import Image 

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad
    
    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for name, module in self.model.named_children(): # layern
            if name == 'fc':
                break
            if isinstance(module, nn.Sequential):
                for sub_module_name, sub_module in module.named_children(): # layern.k
                    for ssub_module_name, ssub_module in sub_module.named_children(): # layern.k.conv2
                        if isinstance(ssub_module, nn.Sequential):
                            for sssub_module_name, sssub_module in ssub_module.named_children(): # layern.k.downsampling.0
                                if not isinstance(ssub_module, nn.Sequential):
                                    print(sssub_module_name)
                                    x = sssub_module(x)
                        else:
                            x = ssub_module(x)
                            if f"{name}.{sub_module_name}.{ssub_module_name}" == self.target_layer:
                                x.register_hook(self.save_gradient)
                                conv_output = x
            else:
                x = module(x)
                if name == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x
        
        # for name, module in self.model.named_modules():
        #     if isinstance(module, nn.Sequential):
        #         continue
        #     if name == 'fc':
        #         break
        #     x = module(x)
        #     if name == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class LayerCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = guided_gradients
        weights[weights < 0] = 0 # discard negative gradients
        # Element-wise multiply the weight with its conv output and then, sum
        cam = np.sum(weights * target, axis=0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.LANCZOS))/255

        return cam


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr