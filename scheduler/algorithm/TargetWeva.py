from abc import ABC, abstractmethod

class TargetWevaBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    # def model_layer_names(self, model):
    #     # TODO
    #     return []
    
    # def get_lr(self,):
    #     lrs = []
    #     for i in range(len(self.optimizer.param_groups)):
    #         lrs.append(self.optimizer.param_groups[i]['lr'])
    #     return lrs
