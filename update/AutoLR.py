from update.UpdateBase import LRUpdate

import torch.optim as optim

class AutoLR(LRUpdate):
    def __init__(self, model, init_lr, max_f, min_f):
        self.max_f = max_f
        self.min_f = min_f
        self.desired_weva_set = []

        self.scale = 1000000
        self.gamma = 0.2
        self.cls_lr = 0.01
        self.thr_score = 0.94

        self.e_drop = 40
        self.e_end = 50
        self.mlast = 3

        self.get_model_layer_names(model)
        self.optimizer_binding(model, [init_lr])

    def get_model_layer_names(self, model):
        # model layer names
        self.layer_names = dict()
        for name, _ in model.named_parameters():
            layer_name = ".".join(name.split(".")[:-1])
            self.layer_names[layer_name] = len(self.layer_names)-1
    
    def get_lr(self, optimizer):
        lrs = []
        for i in range(len(optimizer.param_groups)):
            lrs.append(optimizer.param_groups[i]['lr'])
        return lrs

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
    
    # AutoLR utils
    def weva2index(self, weva):
        weva = weva[1:-self.mlast]
        weva_index = [weva.index(x) for x in sorted(weva)]
        return weva_index

    def isSort(self, weva):
        weva_index = self.weva2index(weva)
        score = self.get_score(weva_index)
        return score

    def get_score(self, A):
        diff = 0.
        for index, element in enumerate(A):
            diff += abs(index - element)
        return 1.0 - diff / len(A) ** 2 * 2
    
    def optimizer_binding(self, model, now_lr):
        # TODO : add pruning options
        # ignored_params = list(map(id, model.model.layer2.parameters())) + list(map(id, model.model.layer3.parameters())) + \
        #                 list(map(id, model.model.layer4.parameters())) + list(map(id, model.model.fc.parameters())) \
        #                 + list(map(id, model.classifier.parameters())) + list(map(id, model.model.layer1.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        if len(now_lr) == 1:
            now_lr *= len(self.layer_names)

        param_list = []
        for layer_name in self.layer_names.keys():
            current_layer = model
            layer_name_split = layer_name.split(".")
            for name in layer_name_split:
                if name.isdigit():
                    # 정수로 변환하여 순차적으로 인덱스로 접근
                    index = int(name)
                    current_layer = current_layer[index]
                else:
                    # 그 외의 경우에는 getattr 사용
                    current_layer = getattr(current_layer, name)
            param = current_layer.parameters()

            param_list.append({'params': param, 'lr': now_lr[self.layer_names[layer_name]]})

        # param_list = []
        # for name, param in model.named_parameters():
        #     layer_name = ".".join(name.split(".")[:-1])
        #     # self.layer_names.add(layer_name)
        #     param_list.append({'params': param, 'lr': now_lr[self.layer_names[layer_name]]})
        optimizer_try = optim.SGD(param_list, momentum=0.9, weight_decay=5e-4, nesterov=True)  # for CUB

        return optimizer_try
    
    def try_lr_update(self, weva_try, epoch, now_lr):
        score = round(self.isSort(weva_try),3)
        if score >= self.thr_score:
            Trial_error = False
            if epoch == self.e_drop - 1:
                for i in range(len(now_lr)):
                    now_lr[i] = now_lr[i] * self.gamma
        else:
            Trial_error = True
        
        return Trial_error, score, now_lr