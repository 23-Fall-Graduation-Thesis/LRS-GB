from abc import ABC, abstractmethod
import torch
from utils.lr_utils import get_size_scalar, diff_to_weva, get_frob_norm, get_lone_norm, get_linf_norm

class TargetWevaBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def cal_target_weva(self, weva_table, n_epoch):
        """target weight variation을 계산하여, 반환합니다.

        Args:
            weva_table (_type_): weight variation이 저장된 리스트
            n_epoch (_type_): 현재 epoch
        
        Returns:
            target_weva: target weight variation
        """
        pass

    @abstractmethod
    def cal_target_init_weva(self, now_init_weva, n_epoch, constraints):
        """target init weight variation을 계산하여, 반환합니다.

        Args:
            now_init_weva (_type_): 현재 weight와 pre-trained weight와의 variation
            n_epoch (_type_): 현재 epoch
            constraints (_type_): distance based regularization에 사용하는 constraints
        
        Returns:
            target_init_weva: target init weight variation
        """
        pass


class AutoLRTargetWeva(TargetWevaBase):
    def __init__(self):
        pass


    def init(self, max_f, min_f):
        super().__init__()
        self.max_f = max_f
        self.min_f = min_f


    def cal_target_weva(self, weva_table, n_epoch):
        now_weva = weva_table[-1][:-1]
        if len(weva_table) <= 1:
            # Here we make desired weight variation(weva)
            
            # interval -> d_t of AutoLR algorithm
            max_weva = max(now_weva)
            min_weva = min(now_weva)
            if n_epoch == 0: 
                max_weva = max(now_weva)*self.max_f
                min_weva = min(now_weva)*self.min_f
            print('Bound condition of weigh variation are Max: {:.6f} Min: {:.6f}'.format(max_weva, min_weva))
            interval = (max_weva - min_weva) / (len(now_weva) - 1)

            # Eq. 12, 13, 14 of AutoLR algorithm
            if n_epoch == 0:
                # v_bar_t가 없는 경우, epoch 0
                bias = min_weva
                target_weva = [] # v_bar_t
                for i in range(len(now_weva)):
                    target_weva.append(bias + i * interval)
            else:
                # v_bar_t가 있는 경우, 중심을 기준으로 update
                target_weva = now_weva[:]
                center = int(len(now_weva)/2)
                for i in range(center, 0, -1):
                    if target_weva[i] < target_weva[i-1]:
                        target_weva[i - 1] = target_weva[i] - interval
                for i in range(center, len(now_weva)-1, 1):
                    if target_weva[i] > target_weva[i + 1]:
                        target_weva[i + 1] = target_weva[i] + interval
            
            return target_weva
        else:
            
            return False
        
        
    def cal_target_init_weva(self, now_init_weva, n_epoch, constraints):
        pass

# Trial 1
class LRSGBTargetWeva(AutoLRTargetWeva):
    def __init__(self):
        pass
    
    def init(self, max_f, min_f, constraints, trial_effect=0.05):
        super().__init__()
        super().init(max_f, min_f)
        self.constraints = constraints
        self.trial = 0
        self.trial_effect = trial_effect

    
    def reset_trial(self):
        self.trial = 0
    
    
    def cal_target_init_weva(self, now_init_weva, n_epoch):
        if n_epoch == 0:
            #TODO:  
            target_init_weva = now_init_weva[:]
            for i in range(len(self.constraints)):
                if now_init_weva[i] > self.constraints[i]:
                    target_init_weva[i] = self.constraints[i] - self.trial * self.trial_effect
            self.trial += 1
        else:
            target_init_weva = now_init_weva[:]
            for i in range(len(self.constraints)):
                if now_init_weva[i] > self.constraints[i]:
                    target_init_weva[i] = self.constraints[i] - self.trial * self.trial_effect
            self.trial += 1
        
        return target_init_weva
    
# Trial 2
class LRSRSLTargetWeva(AutoLRTargetWeva):
    def __init__(self):
        pass
    
    def init(self, max_f, min_f, K, scale_factor):
        super().__init__()
        super().init(max_f, min_f)
        self.trial = 0
        self.K = K
        self.scale_factor = scale_factor

    def reset_trial(self):
        self.trial = 0
    
    #NOTE: 모든 WEIGHT 수정
    def cal_target_init_weva(self, weight_difference, param_num_list):
        target_init_weva = []
        #print(len(weight_difference), len(param_num_list))
        for i, diff_list in enumerate(weight_difference[:-1]):
            target_temp = []
            #print("diff len", len(diff_list))
            for diff in diff_list:
                norms = get_lone_norm(diff)
                K = self.K * pow(self.scale_factor, int(i/2))
                target_temp.append(diff * (1.0 / torch.maximum(torch.tensor(1.0, device=norms.device), norms / K)))
            n = param_num_list[i]
            target_init_weva.append(diff_to_weva(target_temp, n))
        self.trial += 1
        #print("result: ", len(target_init_weva))
        return target_init_weva