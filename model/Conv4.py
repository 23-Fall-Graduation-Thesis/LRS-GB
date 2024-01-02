import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv4(nn.Module):
    def __init__(self, num_class):
        super(Conv4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.fc = nn.Linear(2 * 2 * 128, num_class)
        
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)
        
        x = x.view(-1, 2 * 2 * 128) # batch size x flattened feature size
        x = F.relu(self.fc(x))
        
        return x
        