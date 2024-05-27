from typing import *
from scheduler.SchedulerBase import SchedulerBase
import torch.optim as optim

class GB_with_Weva(SchedulerBase):
    def __init__(self, model, model_name, init_lr, thr_init_score, K, scale_factor, bound, all_epoch, target_func, instances : Dict[str, str] = None):
        if instances is None:
            instances = dict(
                weva_method = "LRSGBwithTargetWeight",
                lr_method = "AutoLRTargetLR",
                condition_method = "LRSGBCondition"
            )
        super().__init__(model, model_name, init_lr, instances)
        # self.gen_bound = []
        self.all_epoch = all_epoch
        
        self.scale = 1000000
        self.gamma = 0.2
        self.cls_lr = 0.01

        self.e_drop = int(self.all_epoch * 0.8) # 30의 0.8
        self.e_end = self.all_epoch
        self.mlast = 3

        self.weva_manager.init(K, scale_factor, bound, target_func)
        self.condition_manager.init(thr_init_score)
        self.target_weva_set = []

    # def set_initial_weva(self, init_diff_table, n_epoch, param_num_list):
    #     if len(init_diff_table)<2:
    #         print('setting epoch target weva')
    #         target_weva = self.weva_manager.cal_target_init_weva(init_diff_table, n_epoch, self.all_epoch, param_num_list)
    #         self.target_weva_set.append(target_weva)
    
    def set_target_weva(self, init_diff_table, n_epoch, all_epoch, param_num_list):
        if len(init_diff_table)<2:
            print('setting epoch target weva')
            target_weva = self.weva_manager.cal_target_weva(n_epoch, all_epoch, param_num_list)
            self.target_weva_set.append(target_weva)
                
    
    def adjustLR(self, weva_table, init_diff_table, lr_table, n_epoch, param_num_list):
        now_lr = lr_table[-1][:-1]
        now_weva = weva_table[-1][:-1]

        # target_weva = self.weva_manager.cal_target_weva(weva_table, n_epoch, self.all_epoch)
        # if not target_weva:
        #     target_weva = self.target_weva_set[-1]
        # else:
        #     self.target_weva_set.append(target_weva)
        target_weva = self.target_weva_set[-1]
        
        target_lr = self.lr_manager.cal_target_lr(now_weva, now_lr, target_weva, self.cls_lr)
        
        return target_lr
    
    
    def try_lr_update(self, weva_try):
        check_GB, init_score = self.condition_manager.check_condition(weva_try, self.target_weva_set)
        if check_GB:
            Trial_error = False
        else:
            Trial_error = True
        return Trial_error, init_score
    
    
    def decay_lr(self, epoch, now_lr):
        if epoch == self.e_drop - 1:
            for i in range(len(now_lr)):
                now_lr[i] = now_lr[i] * self.gamma
        
        return now_lr