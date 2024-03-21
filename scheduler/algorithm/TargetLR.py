from abc import ABC, abstractmethod

class TargetLRBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def cal_target_lr(self):
        pass
    
    @abstractmethod
    def cal_target_init_lr(self):
        pass
    

class AutoLRTargetLR(TargetLRBase):
    def __init__(self):
        super().__init__()

    # calculate target lr based target weight variance
    def cal_target_lr(self, now_weva, now_lr, target_weva, cls_lr):
        target_lr = now_weva[:]
        Gvalue = []
        for i in range(len(now_lr)):
            Gvalue.append(now_weva[i]/now_lr[i])

        for i in range(len(target_lr)):
            target_lr[i] = (target_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]

        target_lr.append(cls_lr) # classifier lr 고정해서 사용

        return target_lr
    
    
    def cal_target_init_lr(self):
        pass


# Trial 1
class GBwithAutoLRTargetLR(AutoLRTargetLR):
    def __init__(self):
        super().__init__()


    def cal_target_init_lr(self, now_weva, now_lr, now_init_weva, target_init_weva, cls_lr):
        target_init_lr = []
        for i in range(len(now_lr)):
            temp = (now_lr[i] * (target_init_weva[i] - now_init_weva[i] + now_weva[i])) / now_weva[i]
            target_init_lr.append(temp)

        target_init_lr.append(cls_lr)
        
        return target_init_lr
    
    
    def select_lr(self, GB_update, target_lr, target_init_lr):
        if GB_update :
            return target_init_lr
        else:
            return target_lr
        
# only GB - Score
class LRSGBTargetLR(TargetLRBase):
    def __init__(self):
        super().__init__()

    def cal_target_lr(self):
        pass

    def cal_target_init_lr(self, now_weva, now_lr, now_init_weva, target_init_weva, cls_lr):
        target_init_lr = []
        for i in range(len(now_lr)):
            temp = (now_lr[i] * (target_init_weva[i] - now_init_weva[i] + now_weva[i])) / now_weva[i]
            target_init_lr.append(temp)

        target_init_lr.append(cls_lr)
        
        return target_init_lr