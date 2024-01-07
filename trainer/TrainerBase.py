from abc import ABC, abstractmethod
import torch, copy
from torch.autograd import Variable
from utils.utils import compute_weight_variation
from tqdm import tqdm
import torch.nn as nn

class TrainerBase(ABC):
    def __init__(self, model, device, trainloader, validloader, testloader, checkpt, board_name, writer):
        self.model = model.to(device) # current model
        self.device = device

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader

        self.checkpt = checkpt
        self.board_name = board_name
        self.writer = writer

    @abstractmethod
    def train_model(self):
        pass

    def train(self):
        self.model.train()
        
        train_loss = 0
        loss = 0
        train_acc = 0
        for data, target in self.trainloader :
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            self.optimizer.step()
        
        train_loss = train_loss / len(self.trainloader.dataset)
        train_acc = train_acc / len(self.trainloader.dataset) * 100
        
        return train_loss, train_acc

    def validation(self):
        self.model.eval()
        
        valid_loss = 0
        valid_acc = 0
        with torch.no_grad():
            for data, target in self.validloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                valid_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                valid_acc += pred.eq(target.view_as(pred)).sum().item()
        
        valid_loss = valid_loss / len(self.validloader.dataset)
        valid_acc = valid_acc / len(self.validloader.dataset) * 100
        
        return valid_loss, valid_acc
    
    def test(self):
        self.model.load_state_dict(torch.load(self.checkpt))
        self.model.eval()
        
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_acc += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss = test_loss / len(self.testloader.dataset)
        test_acc = test_acc / len(self.testloader.dataset) * 100
        
        return test_loss, test_acc
    
    def train_1epoch(self, modelB, optimizer_try, layer_names):
        modelA = copy.deepcopy(modelB)
        modelB = modelB.to(self.device)

        train_loss = 0.0
        train_acc = 0

        for data, target in tqdm(self.trainloader):
            # get the inputs, wrap them in Variable
            data, target = Variable(data.to(self.device)), Variable(target.to(self.device))
            optimizer_try.zero_grad()
            # forward
            output = modelB(data)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer_try.step()

        train_loss = train_loss / len(self.trainloader.dataset)
        train_acc = train_acc / len(self.trainloader.dataset) * 100

        weva = compute_weight_variation(modelA, modelB, layer_names)

        return weva, train_loss, train_acc