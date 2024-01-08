from scheduler.SchedulerBase import SchedulerBase
from typing import *

class LRS_GB(SchedulerBase):
    def __init__(self,  model, init_lr, instances: Dict[str, str], constraints: List[float]):
        super().__init__(model, init_lr, instances)
        self.constrains = constraints
    
    def adjust_lr(self, weva_table, lr_table, score, n_epoch):
        pass
    
    def try_lr_update(self, weva_try, epoch, now_lr):
        pass