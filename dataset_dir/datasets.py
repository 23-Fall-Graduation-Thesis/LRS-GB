import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from dataset_dir.cub200 import Cub2011

from scipy.io import loadmat
from pathlib import Path
import os
from PIL import Image
import numpy as np

def datasetload(dataset_name, batch_size):
    if dataset_name == 'cifar10':
        return Cifar10(batch_size)
    elif dataset_name == 'cifar100':
        return Cifar100(batch_size)
    elif dataset_name == 'svhn':
        return SVHN(batch_size)
    elif dataset_name == 'cub':
        return CUB200(batch_size)
    elif dataset_name == 'cars':
        return Cars(batch_size)


def Cifar10(batch_size):
    n_class = 10

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, 
                                            download=True, transform=transform)
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size
    trainset, validset = random_split(train_dataset, [train_size, valid_size])
    
    trainloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    validloader = DataLoader(validset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    # classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    
    return trainloader, validloader, testloader, n_class

def Cifar100(batch_size):
    n_class = 100

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform)
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size
    trainset, validset = random_split(train_dataset, [train_size, valid_size])
    
    trainloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    validloader = DataLoader(validset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    return trainloader, validloader, testloader, n_class
    

def SVHN(batch_size):
    n_class = 10
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    svhn = torchvision.datasets.SVHN(root='./data/svhn', transform=transform, download=True)
    train_indices = torch.arange(0, 50000)
    valid_indices = torch.arange(50000, 60000)
    test_indices = torch.arange(60000, 70000)
    train_svhn = torch.utils.data.Subset(svhn, train_indices)
    valid_svhn = torch.utils.data.Subset(svhn, valid_indices)
    test_svhn = torch.utils.data.Subset(svhn, test_indices)
    
    trainloader = DataLoader(train_svhn, batch_size, shuffle=True, drop_last=True, num_workers=0)
    validloader = DataLoader(valid_svhn, batch_size, shuffle=True, drop_last=True, num_workers=0)
    testloader = DataLoader(test_svhn, batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    return trainloader, validloader, testloader, n_class

def CUB200(batch_size):
    n_class = 200

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = Cub2011(root='./data/cub-200-2011', transform=transform, download=True, train=True)
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size
    trainset, validset = random_split(train_dataset, [train_size, valid_size])
    
    trainloader = DataLoader(trainset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    validloader = DataLoader(validset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    test_dataset = Cub2011(root='./data/cub-200-2011', transform=transform, download=True, train=False)
    testloader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    return trainloader, validloader, testloader, n_class

def Cars(batch_size):
    n_class = 196
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    for dirname, _, filenames in os.walk('/dataset_dir'):
        for filename in filenames:
            pass
    
    root_dir = Path('./dataset_dir/stanford-cars-dataset')
    cars_annos = root_dir / 'cars_annos.mat'
    cars_test = root_dir / 'cars_test' / 'cars_test'
    cars_train = root_dir / 'cars_train' / 'cars_train'
    
    root_dir = Path("./dataset_dir/stanford-cars-dataset-meta")
    cars_annos_train = root_dir / "devkit" / "cars_train_annos.mat"
    cars_annos_test = root_dir / "cars_test_annos_withlabels (1).mat"

    cars_annos_train_mat, cars_annos_test_mat = loadmat(cars_annos_train), loadmat(cars_annos_test)
    
    # Datasets - create custom dataset and a dictionary which relates image path to label
    training_image_label_dictionary, testing_image_label_dictionary = {}, {}

    for arr in cars_annos_train_mat['annotations'][0]:
        image, label = arr[-1][0], arr[-2][0][0] - 1
        training_image_label_dictionary[image] = label

    for arr in cars_annos_test_mat['annotations'][0]:
        image, label = arr[-1][0], arr[-2][0][0] - 1
        testing_image_label_dictionary[image] = label
    
    train_dataset = StanfordCarsCustomDataset(cars_train, training_image_label_dictionary, transform)
    dataset_size = len(train_dataset)
    train_size = int(dataset_size * 0.8)
    valid_size = dataset_size - train_size
    trainset, validset = random_split(train_dataset, [train_size, valid_size])
    
    test_dataset = StanfordCarsCustomDataset(cars_test, testing_image_label_dictionary, transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, validloader, testloader, n_class

class StanfordCarsCustomDataset(Dataset):
    def __init__(self, directory, image_label_dict, transforms):
        super().__init__()

        self.images = [os.path.join(directory, f) for f in os.listdir(directory)]
        self.transforms = transforms
        self.image_label_dict = image_label_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get image
        image = self.images[index]
        img_pil = Image.open(image).convert('RGB')
        img_trans = self.transforms(img_pil)

        # Parse out the label from cars_meta and cars_x_annos files
        image_stem = image.split("\\")[-1]
        img_label = self.image_label_dict[image_stem]

        return img_trans, img_label