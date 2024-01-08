import torch.optim as optim
from scheduler.algorithm.Condition import *
from scheduler.algorithm.TargetLR import *
from scheduler.algorithm.TargetWeva import * 
from abc import ABC, abstractmethod
from typing import *
from utils import get_instance

class SchedulerBase(ABC):
    def __init__(self,  model, init_lr, instances: Dict[str, str]):
        """부모 클래스를 초기화합니다.

        Args:
            model (_type_): 학습중인 모델
            init_lr (_type_): 초기 learning rate
            instances (Dict[str, str]): 알고리즘으로 사용할 방식들의 이름
        """
        self.get_model_layer_names(model)
        self.optimizer_binding(model, [init_lr])
        try :
            self.weva_manager = get_instance(instances["weva_method"])
            self.lr_manager = get_instance(instances["lr_method"])
            self.condition_manager = get_instance(instances["condition_method"])
        except ValueError as e:
            print(e)
    
    
    def get_model_layer_names(self, model):
        # model layer names
        self.layer_names = dict()
        for name, _ in model.named_parameters():
            name_split = name.split(".")
            layer_name = ".".join(name_split[:-1] if len(name_split)<3 else name_split[:2])
            '''
            alexnet : features.0, features.3, classifier.4 ..
            resnet : layer1.1, layer2.0, fc ... (block)
            '''
            if layer_name not in self.layer_names:
                self.layer_names[layer_name] = len(self.layer_names)
    
    
    def get_lr(self, optimizer):
        lrs = []
        for i in range(len(optimizer.param_groups)):
            lrs.append(optimizer.param_groups[i]['lr'])
        
        return lrs


    def optimizer_binding(self, model, now_lr):
        # TODO : add pruning options
        # ignored_params = list(map(id, model.model.layer2.parameters())) + list(map(id, model.model.layer3.parameters())) + \
        #                 list(map(id, model.model.layer4.parameters())) + list(map(id, model.model.fc.parameters())) \
        #                 + list(map(id, model.classifier.parameters())) + list(map(id, model.model.layer1.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        if len(now_lr) == 1:
            now_lr *= len(self.layer_names)

        param_list = []
        for layer_name in self.layer_names.keys():
            current_layer = model
            layer_name_split = layer_name.split(".")
            for name in layer_name_split:
                if name.isdigit():
                    # 정수로 변환하여 순차적으로 인덱스로 접근
                    index = int(name)
                    current_layer = current_layer[index]
                else:
                    # 그 외의 경우에는 getattr 사용
                    current_layer = getattr(current_layer, name)
            param = current_layer.parameters()

            param_list.append({'params': param, 'lr': now_lr[self.layer_names[layer_name]]})

        # param_list = []
        # for name, param in model.named_parameters():
        #     layer_name = ".".join(name.split(".")[:-1])
        #     # self.layer_names.add(layer_name)
        #     param_list.append({'params': param, 'lr': now_lr[self.layer_names[layer_name]]})
        optimizer_try = optim.SGD(param_list, momentum=0.9, weight_decay=5e-4, nesterov=True)  # for CUB

        return optimizer_try
    
    
    @abstractmethod
    def adjust_lr(self, weva_table, lr_table, score, n_epoch):
        pass
    
    
    @abstractmethod
    def try_lr_update(self, weva_try, epoch, now_lr):
        pass
    