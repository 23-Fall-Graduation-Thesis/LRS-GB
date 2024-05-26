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


class AdvAutoLRTargetLR(TargetLRBase):
    def __init__(self):
        super().__init__()
        # self.alpha = 0.9   # for constant momentum
        self.epoch = -1
        self.oscillation = [] # target을 기준으로 통과한 횟수 (진동 횟수 count하여 momentum의 정도 결정)

    # calculate target lr based target weight variance
    def cal_target_lr(self, now_weva, now_lr, target_weva, cls_lr, epoch):
        if epoch != self.epoch:
            self.epoch = epoch
            self.oscillation = now_weva[:]
            for i in range(len(self.oscillation)):
                self.oscillation[i] = [target_weva[i] - now_weva[i], 0] 
            
        target_lr = now_weva[:]
        Gvalue = []
        for i in range(len(now_lr)):
            Gvalue.append(now_weva[i]/now_lr[i])

        # use lr momentum
        for i in range(len(target_lr)):
            # target_lr[i] = self.alpha * now_lr[i] + (1-self.alpha) * ((target_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]) #apply constant momentum
            # target_lr[i] = scores[i] * now_lr[i] + (1-scores[i]) * ((target_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]) # apply score momentum
            # # to accelerate convergence
            # if scores[i] < 0.5:
            #     target_lr[i] = (target_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]
            # else:
            #     mom = 0.5 + scores[i]/2
            #     target_lr[i] = mom * now_lr[i] + (1-mom) * ((target_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]) # apply score momentum
            if (target_weva[i] - now_weva[i]) * self.oscillation[i][0] < 0:
                self.oscillation[i][1] += 1 # 진동 횟수 1회 추가
            mom = 1 - pow(0.5, self.oscillation[i][1]) # 진동 횟수가 클수록 now_lr로부터 많이 변화하지 않도록
            target_lr[i] = mom * now_lr[i] + (1-mom) * ((target_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]) # apply score momentum
            self.oscillation[i][0] = target_weva[i] - now_weva[i] # update

        target_lr.append(cls_lr) # classifier lr 고정해서 사용

        return target_lr
    
    
    def cal_target_init_lr(self):
        pass


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