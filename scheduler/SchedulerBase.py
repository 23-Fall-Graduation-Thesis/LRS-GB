import torch.optim as optim
from scheduler.algorithm.Condition import AutoLRCondition, LRSGBCondition
from scheduler.algorithm.TargetLR import AutoLRTargetLR, LRSGBTargetLR
from scheduler.algorithm.TargetWeva import AutoLRTargetWeva, LRSGBTargetWeight
from abc import ABC, abstractmethod
from typing import *
from utils.utils import get_instance
from utils.lr_utils import layer_block_info

class SchedulerBase(ABC):
    def __init__(self, model, model_name, init_lr, instances: Dict[str, str]):
        """부모 클래스를 초기화합니다.

        Args:
            model (_type_): 학습중인 모델
            init_lr (_type_): 초기 learning rate
            instances (Dict[str, str]): 알고리즘으로 사용할 방식들의 이름
        """
        self.layer_name_list = layer_block_info(model_name)
        self.get_model_layer_names()
        self.optimizer_binding(model, [init_lr])
        try :
            self.weva_manager = get_instance(instances["weva_method"])
            self.lr_manager = get_instance(instances["lr_method"])
            self.condition_manager = get_instance(instances["condition_method"])
        except ValueError as e:
            print(e)
    
    
    def get_model_layer_names(self):
        # model layer names
        self.layer_name_dict = dict()
        for idx in range(len(self.layer_name_list)):
            for layer_name in self.layer_name_list[idx]:
                self.layer_name_dict[layer_name] = idx
    
    
    def get_lr(self, optimizer):
        lrs = []
        for i in range(len(optimizer.param_groups)):
            lrs.append(optimizer.param_groups[i]['lr'])
        
        return lrs


    def optimizer_binding(self, model, now_lr):
        # # TODO : add pruning options
        # ignored_params = list(map(id, model.model.layer2.parameters())) + list(map(id, model.model.layer3.parameters())) + \
        #                 list(map(id, model.model.layer4.parameters())) + list(map(id, model.model.fc.parameters())) \
        #                 + list(map(id, model.classifier.parameters())) + list(map(id, model.model.layer1.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        if len(now_lr) == 1:
            # TODO
            now_lr *= len(self.layer_name_list)

        param_list = []
        for idx in range(len(self.layer_name_list)):
            param = []
            for layer_name in self.layer_name_list[idx]:
                cur_layer = model
                layer_name_split = layer_name.split(".")
                for name in layer_name_split:
                    if name.isdigit():
                        # 정수로 변환하여 순차적으로 인덱스로 접근
                        index = int(name)
                        cur_layer = cur_layer[index]
                    else:
                        # 그 외의 경우에는 getattr 사용
                        cur_layer = getattr(cur_layer, name)
                param.extend(cur_layer.parameters())

            param_list.append({'params': param, 'lr': now_lr[idx]})

        optimizer_try = optim.SGD(param_list, momentum=0.9, weight_decay=5e-4, nesterov=True)  # for CUB

        return optimizer_try
    
    
    @abstractmethod
    def adjustLR(self, weva_table, lr_table, n_epoch): 
        """새로운 target_lr을 계산합니다.

        Args:
            weva_table (_type_): _description_
            lr_table (_type_): _description_
            n_epoch (_type_): _description_
        
        Calls:
            self.weva_manager.cal_target_weva() / cal_target_init_weva()
            self.cal_target_lr() / cal_target_init_lr() / select_lr()
        """
        pass
    
    
    @abstractmethod
    def try_lr_update(self, weva_try, epoch, now_lr):
        """weva를 기반으로 조건을 확인합니다.

        Args:
            weva_try (_type_): 시도하는 weva
            epoch (_type_): 현재 epoch
            now_lr (_type_): 시도하는 lr
        
        Calls:
            self.condition_manager.checK_condition()
        
        Returns:
            Traial_error: 수정이 필요한지를 나타냅니다. (True: 필요 / False: 필요하지 않음)
        """
        pass
    