from typing import *
from scheduler.SchedulerBase import SchedulerBase
import torch.optim as optim

class LRS_GB_Score(SchedulerBase):
    def __init__(self, model, model_name, init_lr, thr_init_score, K, scale_factor, bound, min_f, max_f, instances : Dict[str, str] = None):
        if instances is None:
            instances = dict(
                weva_method = "LRSGBwithAutoLRInitTargetWeight",
                lr_method = "LRSGBTargetLR",
                condition_method = "LRSGBCondition"
            )
        super().__init__(model, model_name, init_lr, instances)
        self.target_init_weva_set = []

        self.scale = 1000000
        self.gamma = 0.2
        self.cls_lr = 0.01

        self.e_drop = 40
        self.e_end = 50
        self.mlast = 3

        self.weva_manager.init(K, scale_factor, bound, min_f, max_f)
        self.condition_manager.init(thr_init_score)

    def set_initial_weva(self, init_diff_table, n_epoch, param_num_list):
        if len(init_diff_table)<2:
            print('setting epoch target weva')
            target_init_weva = self.weva_manager.cal_target_init_weva(init_diff_table, n_epoch, param_num_list)
            self.target_init_weva_set.append(target_init_weva)
    
    def adjustLR(self, weva_table, now_init_weva, init_diff_table, lr_table, n_epoch, param_num_list):
        now_lr = lr_table[-1][:-1]
        now_init_weva = now_init_weva[:-1] 
        now_weva = weva_table[-1][:-1]

        target_init_weva = self.weva_manager.cal_target_init_weva(init_diff_table, n_epoch, param_num_list)
        if not target_init_weva:
            target_init_weva = self.target_init_weva_set[-1]
        else:
            self.target_init_weva_set.append(target_init_weva)
        
        target_init_lr = self.lr_manager.cal_target_init_lr(now_weva, now_lr, now_init_weva, target_init_weva, self.cls_lr)
        
        return target_init_lr
    
    
    def try_lr_update(self, init_weva_try):
        check_GB, init_score = self.condition_manager.check_condition(init_weva_try, self.target_init_weva_set)
        if check_GB:
            Trial_error = False
            # TODO: weight decay
        else:
            Trial_error = True
        return Trial_error, init_score
    
    def decay_lr(self, epoch, now_lr):
        if epoch == self.e_drop - 1:
            for i in range(len(now_lr)):
                now_lr[i] = now_lr[i] * self.gamma
        
        return now_lr