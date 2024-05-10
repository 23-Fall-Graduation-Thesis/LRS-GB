from abc import ABC, abstractmethod
import torch
import numpy as np
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



# GBweva
class LRSGBwithTargetWeight(TargetWevaBase):
    def __init__(self):
        pass
    
    def init(self, K, scale_factor, bound, target_func):
        super().__init__()
        self.K = K
        self.scale_factor = scale_factor
        self.bound = bound
        self.target_func = self.set_target_func(target_func)

    # def cal_target_weva(self, weva_table, n_epoch, all_epoch):
    #     now_weva = weva_table[-1][:-1]
    #     target_weva = []

    #     for i in range(len(now_weva)):
    #         # target_weva.append(self.gen_bound[i]/all_epoch) 
    #         target_weva.append(self.target_func(n_epoch, all_epoch))
            
    #     return target_weva

    def cal_target_weva(self, n_epoch, all_epoch, param_num_list):
        target_weva = []
        K = self.K * self.target_func(n_epoch, all_epoch)

        for i in range(len(param_num_list)):
            target_weva.append(K * pow(self.scale_factor, int(i/2)))
            
        return target_weva
    
    def cal_target_init_weva(self):
        return None
        
    # def cal_target_init_weva(self, init_diff_table, n_epoch, all_epoch, param_num_list):
    #     if len(init_diff_table) <= 1:
    #         # 새로 target init weight variation을 계산해야 하는 경우
    #         now_init_diff = init_diff_table[-1]
    #         # print(now_init_diff)
    #         self.gen_bound = []
    #         self.target_weva = []

    #         for i, diff_list in enumerate(now_init_diff[:-1]):
    #             n = param_num_list[i]
    #             K = self.K * pow(self.scale_factor, int(i/2))
    #             # print(K)

    #             if self.bound == 'diff':
    #                 target_temp = []
    #                 for diff in diff_list:
    #                     norms = torch.norm(diff, p=2, dim=(1,2,3), keepdim=True) # 일단 지금은 L2 norm이라 이렇게 맞춤 -> L1 norm일 경우 여기와 diff to weva 수정 필요
    #                     # target_temp.append(diff * (1.0 / torch.maximum(torch.tensor(1.0, device=norms.device), norms / K)))
    #                     #NOTE: 제한된 바운드를 구한거니까 여기도 바로 k값을 그대로 적용
    #                     target_temp.append(diff * (1.0 / (norms / K)))
        
    #                 self.gen_bound.append(diff_to_weva(target_temp, n))
    #                 self.target_weva.append(diff_to_weva(target_temp, n)/all_epoch)

    #             elif self.bound == 'weva':
    #                 weva = diff_to_weva(diff_list, n)
    #                 #NOTE: 제한된 바운드를 구한거니까 k값을 그대로 적용
    #                 self.gen_bound.append(K)
    #                 self.target_weva.append(K/all_epoch)
    #                 # weva 자체가 norm이기 때문에 bound를 적용하면, 그냥 계산된 오름차순이 generalization bound가 됨

    #         return self.target_weva
        
    def set_target_func(self, target_func):
        def constant(cur_epoch, tot_epoch):
            # result = [1 for y in tot_epoch]
            # sum_K = sum(result)
            # scaled_y = [y*self.K/sum_K for y in result]
            # return scaled_y[cur_epoch]
            return 1/tot_epoch
        if target_func == 'constant':
            return constant
        
        def cosine(cur_epoch, tot_epoch):
            cur_epochs = np.arange(0, tot_epoch)
            result = np.cos((np.pi)*(cur_epochs)/tot_epoch) + 1
            sum_K = sum(result)
            scaled_y = [y/sum_K for y in result]
            return scaled_y[cur_epoch]
        if target_func == 'cosine':
            return cosine
        
        def linear(cur_epoch, tot_epoch):
            cur_epochs = np.arange(0, tot_epoch)
            result = tot_epoch - cur_epochs
            sum_K = sum(result)
            scaled_y = [y/sum_K for y in result]
            return scaled_y[cur_epoch]
        if target_func == 'linear':
            return linear
        
        def inverse(cur_epoch, tot_epoch):
            cur_epochs = np.arange(0, tot_epoch)
            result = 1/(cur_epochs+1)
            sum_K = sum(result)
            scaled_y = [y/sum_K for y in result]
            return scaled_y[cur_epoch]
        if target_func == 'inverse':
            return inverse
        
        def step(cur_epoch, tot_epoch):
            cur_epochs = np.arange(0, tot_epoch)
            result = []
            for cur_epochs in range(tot_epoch):
                if cur_epochs < int(0.5 * tot_epoch):
                    result.append(1)
                elif cur_epochs < int(0.8 * tot_epoch):
                    result.append(0.5)
                else:
                    result.append(0.25)
            sum_K = sum(result)
            scaled_y = [y/sum_K for y in result]
            return scaled_y[cur_epoch]
        if target_func == 'step':
            return step