from typing import *
from scheduler.SchedulerBase import SchedulerBase
import torch.optim as optim

# Trial 1
class GB_with_AutoLR(SchedulerBase):
    def __init__(self, model, model_name, init_lr, max_f, min_f, thr_score, constraints: List[float], instances: Dict[str, str] = None, use_AutoLR: bool = False):
        if instances is None:
            instances = dict(
                weva_method = "GBwithAutoLRTargetWeva",
                lr_method = "GBwithAutoLRTargetLR",
                condition_method = "GBwithAutoLRcondition"
            )
        super().__init__(model, model_name, init_lr, instances)
        self.target_weva_set = []
        self.target_init_weva_set = []

        self.scale = 1000000
        self.gamma = 0.2
        self.cls_lr = 0.01

        self.e_drop = 40
        self.e_end = 50
        self.mlast = 3

        self.weva_manager.init(max_f, min_f, constraints)
        self.condition_manager.init(thr_score, constraints)

    
    def adjustLR(self, weva_table, now_init_weva, lr_table, n_epoch, GB_update):
        now_weva = weva_table[-1][:-1]
        now_lr = lr_table[-1][:-1]
        now_init_weva = now_init_weva[:-1] 
        target_weva = self.weva_manager.cal_target_weva(weva_table, n_epoch)
        if not target_weva:
            target_weva = self.target_weva_set[-1]
        else:
            self.target_weva_set.append(target_weva)
        
        target_init_weva = self.weva_manager.cal_target_init_weva(now_init_weva, n_epoch)
        self.target_init_weva_set.append(target_init_weva)
        
        target_lr = self.lr_manager.cal_target_lr(now_weva, now_lr, target_weva, self.cls_lr)
        target_init_lr = self.lr_manager.cal_target_init_lr(now_weva, now_lr, now_init_weva, target_init_weva, self.cls_lr)
        
        adjust_lr = self.lr_manager.select_lr(GB_update, target_lr, target_init_lr)
        
        return adjust_lr
    
    
    def try_lr_update(self, weva_try, init_weva_try):
        check_autoLR, check_GB, score = self.condition_manager.check_condition(weva_try, init_weva_try)
        # if (check_autoLR and check_GB) or (GB_update and check_GB) :
        #     Trial_error = False
        #     if epoch == self.e_drop - 1:
        #         for i in range(len(now_lr)):
        #             now_lr[i] = now_lr[i] * self.gamma
        # else:
        #     Trial_error = True
        
        return check_autoLR, check_GB, score 
    
    
    def decay_lr(self, epoch, now_lr):
        if epoch == self.e_drop - 1:
            for i in range(len(now_lr)):
                now_lr[i] = now_lr[i] * self.gamma
        
        return now_lr
    

class GB_Score_with_AutoLR(SchedulerBase):
    def __init__(self, model, model_name, init_lr, max_f, min_f, thr_score, thr_init_score, K, scale_factor, instances : Dict[str, str] = None, use_AutoLR: bool = False):
        if instances is None:
            instances = dict(
                weva_method = "GBScorewithAutoLRTargetWeva",
                lr_method = "GBwithAutoLRTargetLR",
                condition_method = "GBScorewithAutoLRcondition"
            )
        super().__init__(model, model_name, init_lr, instances)
        self.target_weva_set = []
        self.target_init_weva_set = []

        self.scale = 1000000
        self.gamma = 0.2
        self.cls_lr = 0.01

        self.e_drop = 40
        self.e_end = 50
        self.mlast = 3
        
        self.use_AutoLR = use_AutoLR

        self.weva_manager.init(max_f, min_f, K, scale_factor)
        self.condition_manager.init(thr_score, thr_init_score)

    
    def adjustLR(self, weva_table, now_init_weva, init_diff, lr_table, n_epoch, param_num_list, GB_update):
        now_lr = lr_table[-1][:-1]
        now_init_weva = now_init_weva[:-1] 
        now_weva = weva_table[-1][:-1]
        if self.use_AutoLR:
            target_weva = self.weva_manager.cal_target_weva(weva_table, n_epoch)
            if not target_weva:
                target_weva = self.target_weva_set[-1]
            else:
                self.target_weva_set.append(target_weva)
            
        target_init_weva = self.weva_manager.cal_target_init_weva(init_diff, param_num_list)
        self.target_init_weva_set.append(target_init_weva)
        
        if self.use_AutoLR:
            target_lr = self.lr_manager.cal_target_lr(now_weva, now_lr, target_weva, self.cls_lr)
        target_init_lr = self.lr_manager.cal_target_init_lr(now_weva, now_lr, now_init_weva, target_init_weva, self.cls_lr)
        
        if self.use_AutoLR:
            adjust_lr = self.lr_manager.select_lr(GB_update, target_lr, target_init_lr)
        else:
            adjust_lr = target_init_lr
        
        return adjust_lr
    
    
    def try_lr_update(self, weva_try, init_weva_try):
        check_autoLR, check_GB, score, init_score = self.condition_manager.check_condition(weva_try, init_weva_try, self.target_init_weva_set)
        # if (check_autoLR and check_GB) or (GB_update and check_GB) :
        #     Trial_error = False
        #     if epoch == self.e_drop - 1:
        #         for i in range(len(now_lr)):
        #             now_lr[i] = now_lr[i] * self.gamma
        # else:
        #     Trial_error = True
        
        if not self.use_AutoLR:
            check_autoLR = True
        
        return check_autoLR, check_GB, score, init_score
    
    
    def decay_lr(self, epoch, now_lr):
        if epoch == self.e_drop - 1:
            for i in range(len(now_lr)):
                now_lr[i] = now_lr[i] * self.gamma
        
        return now_lr