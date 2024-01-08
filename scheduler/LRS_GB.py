from typing import *
from scheduler.SchedulerBase import SchedulerBase
import torch.optim as optim

class LRS_GB(SchedulerBase):
    def __init__(self, model, model_name, init_lr, max_f, min_f, thr_score, constraints: List[float], instances: Dict[str, str] = None):
        if instances is None:
            instances = dict(
                weva_method = "LRSGBTargetWeva",
                lr_method = "LRSGBTargetLR",
                condition_method = "LRSGBCondition"
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
        
        
        target_weva = self.weva_manager.cal_target_weva(weva_table, n_epoch)
        if not target_weva:
            target_weva = self.target_weva_set[-1]
        else:
            self.target_weva_set.append(target_weva)
        
        target_init_weva = self.weva_manager.cal_target_init_weva(now_init_weva, n_epoch)
        self.target_init_weva_set.append(target_init_weva)
        
        now_init_weva = now_init_weva[:-1]
        target_init_weva = target_init_weva[:-1]
        
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