from scheduler.SchedulerBase import SchedulerBase

import torch.optim as optim
from utils.lr_utils import layer_block_info

class AutoLR(SchedulerBase):
    def __init__(self, model, model_name, init_lr, max_f, min_f, instances):
        super().__init__(model, model_name, init_lr, instances)
        self.max_f = max_f
        self.min_f = min_f
        self.desired_weva_set = []

        self.scale = 1000000
        self.gamma = 0.2
        self.cls_lr = 0.01

        self.e_drop = 40
        self.e_end = 50
        self.mlast = 3

        self.thr_score = 0.94


    def adjustLR(self, weva_table, lr_table, score, n_epoch):
        # calculate new lr
        # now_weva = weva_table[-1][1:-3] # 맨 앞 : base_params, 맨 뒤 2개 layer pruning, classifier lr 고정
        now_weva = weva_table[-1][:-1] # 맨 뒤의 classifier lr만 고정했다고 가정하고 진행
        now_lr = lr_table[-1][:-1]
        if len(weva_table) <= 1:
            # Here we make desired weight variation(weva)
            max_weva = max(now_weva)
            min_weva = min(now_weva)
            if n_epoch == 0: 
                max_weva = max(now_weva)*self.max_f
                min_weva = min(now_weva)*self.min_f
            print('Bound condition of weigh variation are Max: {:.6f} Min: {:.6f}'.format(max_weva,min_weva))
            interval = (max_weva - min_weva) / (len(now_weva) - 1) # d_t

            if n_epoch == 0:
                # v_bar_t가 없는 경우, epoch 0
                bias = min_weva
                desired_weva = [] # v_bar_t
                for i in range(len(now_weva)):
                    desired_weva.append(bias + i * interval)
            else:
                # v_bar_t가 있는 경우, 중심을 기준으로 update
                desired_weva = now_weva[:]
                center = int(len(now_weva)/2)
                for i in range(center, 0, -1):
                    if desired_weva[i] < desired_weva[i-1]:
                        desired_weva[i - 1] = desired_weva[i] - interval
                for i in range(center, len(now_weva)-1, 1):
                    if desired_weva[i] > desired_weva[i + 1]:
                        desired_weva[i + 1] = desired_weva[i] + interval

            self.desired_weva_set.append(desired_weva)
        else:
            # epoch마다 초기화되는 weva_table이 없는 경우 만들고, 있는 경우 그대로 사용
            desired_weva = self.desired_weva_set[-1]

        target_lr = now_weva[:]
        Gvalue = []
        for i in range(len(now_lr)):
            Gvalue.append(now_weva[i]/now_lr[i])

        for i in range(len(target_lr)):
            target_lr[i] = (desired_weva[i] - now_weva[i]) / Gvalue[i] + now_lr[i]

        target_lr.append(self.cls_lr) # classifier lr 고정해서 사용
        adjust_lr = target_lr

        return adjust_lr
    
    
    def try_lr_update(self, weva_try, epoch, now_lr):
        score = round(self.isSort(weva_try), 3)
        if score >= self.thr_score:
            Trial_error = False
            if epoch == self.e_drop - 1:
                for i in range(len(now_lr)):
                    now_lr[i] = now_lr[i] * self.gamma
        else:
            Trial_error = True
        
        return Trial_error, score, now_lr
    
    
        # AutoLR utils
    def weva2index(self, weva):
        # weva = weva[1:-self.mlast]
        weva_index = [weva.index(x) for x in sorted(weva)]
        return weva_index

    def isSort(self, weva):
        weva_index = self.weva2index(weva[:-1]) # exclude classifier layer
        score = self.get_score(weva_index)
        return score

    def get_score(self, A):
        diff = 0.
        for index, element in enumerate(A):
            diff += abs(index - element)
        return 1.0 - diff / len(A) ** 2 * 2