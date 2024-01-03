import torch
from torchvision import models
from model.Conv4 import Conv4

# pretrained models which are trained by ImageNet
def select_model(model_name, num_class=10, pretrained_model=True, checkpt=''):
    # model that we defined
    if model_name == 'Conv4':
        model = Conv4(num_class)
        if pretrained_model:
            model.load_state_dict(torch.load(checkpt))
    else:
        # model that torchvision.models provided
        ## load pre-trained weight (ImageNet)
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.resnet18(weights=weights)
            feature_in = model.fc.in_features
            model.fc = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.resnet34(weights=weights)
            feature_in = model.fc.in_features
            model.fc = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.resnet50(weights=weights)
            feature_in = model.fc.in_features
            model.fc = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.resnet101(weights=weights)
            feature_in = model.fc.in_features
            model.fc = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'resnet152':
            weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.resnet152(weights=weights)
            feature_in = model.fc.in_features
            model.fc = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'alexnet':
            weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.alexnet(weights=weights)
            feature_in = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'vgg16':
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.vgg16_bn(weights=weights)
            feature_in = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'vgg19':
            weights = models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.vgg19_bn(weights=weights)
            feature_in = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'WRN50':
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.wide_resnet50_2(weights=weights)
            feature_in = model.fc.in_features
            model.fc = torch.nn.Linear(feature_in, num_class)
        elif model_name == 'WRN101':
            weights = models.Wide_ResNet101_2_Weights.IMAGENET1K_V1 if pretrained_model else None
            model = models.wide_resnet101_2(weights=weights)
            feature_in = model.fc.in_features
            model.fc = torch.nn.Linear(feature_in, num_class)
        else:
            raise ValueError(f'Invalid mode name.')
        
        ## load pre-trained model (Not ImageNet)
        if checkpt != '':
            model.load_state_dict(torch.load(checkpt))

    return model
        