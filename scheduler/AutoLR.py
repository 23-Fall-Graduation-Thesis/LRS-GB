from scheduler.SchedulerBase import SchedulerBase
import torch.optim as optim

class AutoLR(SchedulerBase):
    def __init__(self, model, model_name, init_lr, max_f, min_f, thr_score):
        instances = dict(
            weva_method = "AutoLRTargetWeva",
            lr_method = "AutoLRTargetLR",
            condition_method = "AutoLRCondition"
        )
        super().__init__(model, model_name, init_lr, instances)
        self.target_weva_set = []

        self.scale = 1000000
        self.gamma = 0.2
        self.cls_lr = 0.01

        self.e_drop = 40
        self.e_end = 50
        self.mlast = 3

        self.weva_manager.init(max_f, min_f)
        self.condition_manager.init(thr_score)


    def adjustLR(self, weva_table, lr_table, n_epoch):
        now_weva = weva_table[-1][:-1]
        now_lr = lr_table[-1][:-1]
        
        # calculate target weight variance
        target_weva = self.weva_manager.cal_target_weva(weva_table, n_epoch)
        if not target_weva:
            target_weva = self.target_weva_set[-1]
        else:
            self.target_weva_set.append(target_weva)
        
        # calculate learning rate based target weight variance
        target_lr = self.lr_manager.cal_target_lr(now_weva, now_lr, target_weva, self.cls_lr)
        
        return target_lr
    
    
    def try_lr_update(self, weva_try, epoch, now_lr):
        # if threshold condition is satisfied, then stop next trial
        check, score = self.condition_manager.check_condition(weva_try)
        if check:
            Trial_error = False
            if epoch == self.e_drop - 1:
                for i in range(len(now_lr)):
                    now_lr[i] = now_lr[i] * self.gamma
        else:
            Trial_error = True
        
        return Trial_error, score, now_lr