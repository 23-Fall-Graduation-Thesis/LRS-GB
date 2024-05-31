from trainer.TrainerBase import TrainerBase
from scheduler.GB_with_weva_score_sum import GB_with_Weva_Score_Sum

import copy, torch, math, csv, os, random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from utils.lr_utils import compute_weight_variation, get_num_layer, compute_weight_difference_and_variation, compute_L1_weight_variation, compute_L1_weight_difference_and_variation, increase_K

class GB_with_Weva_Score_Sum_Trainer(TrainerBase):

    def __init__(self, model, conf, loaders, loggers):
        super().__init__(model, conf['model'], conf['device'], loaders, loggers)
        self.pretrain_model = copy.deepcopy(model) 
        self.thr_init_score = conf['thr_init_score']
        self.K = conf['K']
        self.scale_factor = conf['scale_factor']
        self.bound = None #TODO: delete
        # self.bound = conf['bound']
        # self.increase_bound = conf['increase_bound']
        # self.inc_type = conf['inc_type']
        self.all_epoch = conf['epoch']
        self.target_func = conf['target_func']

        self.method = 'GBweva_score_sum'
        self.isTry = conf['isTry']
        
        if conf['norm'] == 'L1' :
            self.get_weva =  compute_L1_weight_variation
            self.get_weva_and_diff = compute_L1_weight_difference_and_variation
        elif conf['norm'] == 'L2':
            self.get_weva =  compute_weight_variation
            self.get_weva_and_diff = compute_weight_difference_and_variation

    def set_seed(self, seed=2023):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_model(self, epochs, init_lr):
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
    
        self.model.train()
        # weva_success = []
        weva_success = []
        lr_success = []
        ntrial_success = []

        # lr_scheduler = GB_with_Weva(self.model, self.model_name, init_lr, self.thr_init_score, self.K, self.scale_factor, self.bound, self.all_epoch, self.target_func) #TODO instance arg_parser
        lr_scheduler = GB_with_Weva_Score_Sum(self.model, self.model_name, init_lr, self.thr_init_score, self.K, self.scale_factor, self.bound, self.all_epoch, self.target_func, self.isTry) #TODO instance arg_parser
        self.optimizer = lr_scheduler.optimizer_binding(self.model, [init_lr])
        
        best = 0
        best_epoch = 0
        bad_count = 0
        strict = False

        train_logs = []
        valid_logs = []
        # manager = Manager(self.board_name.replace('/', '_'), 9)
        
        # epoch-wise csv result
        model_name = self.board_name.split('/')[0]
        dataset = self.board_name.split('/')[1]
        mode = self.board_name.split('/')[2]
        setting = self.board_name.split('/')[-1]
        os.makedirs(f"./results/csvs/{model_name}/{dataset}/{mode}", exist_ok=True)
        epoch_log_filename = './results/csvs/' + model_name + '/' + dataset + '/' + mode + '/' + setting + '.csv'
        with open(epoch_log_filename, 'a', newline='') as f:
            wr = csv.writer(f)
            # epoch, trial, target_weva, tryweva, try_initweva, current_LR
            wr.writerow(['epoch', 'trial', 'score', 'target_weva', 'tryweva', 'try_initweva', 'current_lr'])

        for epoch in range(epochs):
            print('Epoch:{:04d}'.format(epoch+1))
            # if not strict :
            #     # decreasing thr_score 
            #     if epoch >= 1 and lr_scheduler.thr_score > 0.8:
            #         lr_scheduler.thr_score = lr_scheduler.thr_score*0.99

            Trial_error = True
            
            trial = 0
            init_weva_table = []
            init_diff_table = []
            weva_table = []
            lr_table = []

            now_lr = lr_scheduler.get_lr(self.optimizer)
            # model_temp = copy.deepcopy(self.model)

            while Trial_error:
                # set seed for fix dataloader
                self.set_seed(2023+epoch)
                trial = trial + 1
                if trial > 50:
                    print('WARNING: trial is larger than 50')
                    with open(f"./results/trial.csv", 'a', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow([self.method, model_name, dataset, epoch, '-', '-', '-', self.thr_init_score, self.K, self.scale_factor, self.target_func, self.log_time])
        
                    exit()
                
                model_temp = copy.deepcopy(self.model)
                model_try = copy.deepcopy(self.model)
                optimizer_try = lr_scheduler.optimizer_binding(model_try, now_lr)
                train_loss, train_acc, model_try = self.train_1epoch(model_try, optimizer_try)
    
                weva_try, diff_try, param_num_list = self.get_weva_and_diff(model_temp, model_try, lr_scheduler.layer_name_dict)
                init_weva_try, init_diff_try, param_num_list = self.get_weva_and_diff(self.pretrain_model, model_try, lr_scheduler.layer_name_dict)
                init_diff_table.append(init_diff_try)

                # if never caculated target init weva, 
                # lr_scheduler.set_initial_weva(init_diff_table, epoch, param_num_list)
                lr_scheduler.set_target_weva(init_diff_table, epoch, epochs, param_num_list)
                Trial_error, init_score, target_weva = lr_scheduler.try_lr_update(weva_try, init_weva_try, self.isTry)

                optimizer_try_lrs = lr_scheduler.get_lr(optimizer_try)
                
                # check loss NaN
                if math.isnan(train_loss):
                    print('WARNING: non-finite loss, ending training ')
                    with open(f"./results/nan.csv", 'a', newline='') as f:
                        wr = csv.writer(f)
                        # wr.writerow([self.model_name, self.dataset])
                        wr.writerow([self.method, model_name, dataset, epoch, '-', '-', '-', self.thr_init_score, self.K, self.scale_factor, self.target_func, init_weva_try, weva_try[:-1], optimizer_try_lrs[:-1], self.log_time])
                        
                    exit()

                # Success 
                if Trial_error == False:
                    self.model = copy.deepcopy(model_try)
                    train_logs.append([train_acc, train_loss])
                    self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                    weva_success.append(copy.deepcopy(weva_try))
                    lr_success.append(optimizer_try_lrs)
                    ntrial_success.append(trial)
                    
                    lr_scheduler.condition_bound_update(copy.deepcopy(weva_try), weva_target=copy.deepcopy(target_weva))
                    # manager.record(epoch, now_lr[:-1], weva_try[:-1])
                else:
                    weva_table.append(weva_try)
                    init_weva_table.append(init_weva_try)
                    lr_table.append(optimizer_try_lrs)
                    now_lr = lr_scheduler.adjustLR(weva_table, init_diff_table, lr_table, epoch, param_num_list)

                print('trial: {}, init score: {:.4f}, Train Loss: {:.8f} Acc: {:.8f}'.format(trial, init_score, train_loss, train_acc))

                epoinitLfmt = ['{:.6f}']*(len(init_weva_try)-1)
                epoinitLfmt =' '.join(epoinitLfmt)
                values2 = []
                for i in range(len(init_weva_try)-1):
                    values2.append(init_weva_try[i])
                epoinitLfmt = '   TryInitWeVa :' + epoinitLfmt
                # print(epoinitLfmt.format(*values2))

                epoLfmt = ['{:.6f}']*(len(weva_try)-1)
                epoLfmt =' '.join(epoLfmt)
                values1 = []
                for i in range(len(weva_try)-1):
                    values1.append(weva_try[i])
                epoLfmt = '       TryWeVa :' + epoLfmt
                print(epoLfmt.format(*values1))

                de_weva = lr_scheduler.target_weva_set[-1][:-1]
                epoinitLfmt = ['{:.6f}'] * len(de_weva)
                epoinitLfmt = ' '.join(epoinitLfmt)
                values3 = []
                for i in range(len(de_weva)):
                    values3.append(de_weva[i])
                epoinitLfmt = '    TargetWeVa :' + epoinitLfmt
                print(epoinitLfmt.format(*values3))

                scores = lr_scheduler.condition_manager.get_layer_score(bool=False)
                epoLfmt = ['{:.6f}'] * (len(scores))
                epoLfmt = ' '.join(epoLfmt)
                values5 = []
                for i in range(len(scores)):
                    values5.append(scores[i])
                epoLfmt = '         Score :' + epoLfmt
                print(epoLfmt.format(*values5))

                epoLfmt = ['{:.6f}'] * (len(optimizer_try_lrs)-1)
                epoLfmt = ' '.join(epoLfmt)
                values4 = []
                for i in range(len(optimizer_try_lrs)-1):
                    values4.append(optimizer_try_lrs[i])
                epoLfmt = '  LearningRate :' + epoLfmt
                print(epoLfmt.format(*values4))

                if Trial_error == False:
                    with open(epoch_log_filename, 'a', newline='') as f:
                        wr = csv.writer(f)
                        # epoch, trial, target_weva, tryweva, try_initweva, current_learning rate
                        wr.writerow([epoch, trial, list(values3), list(values1), list(values2), list(values4)])

                print()
                
            if epoch % 5 == 0 or epoch == epochs-1:
                valid_loss, valid_acc = self.validation()
                valid_logs.append([epoch, valid_acc, valid_loss, train_acc-valid_acc])
                
                print('validation loss:{:.3f}'.format(valid_loss), 'acc:{:.2f}'.format(valid_acc))
                print()

                if valid_acc >= best:
                    best = valid_acc
                    best_gap = train_acc - valid_acc
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), self.checkpt) 
                    bad_count = 0
                else:
                    bad_count += 1
                
                # if bad_count == 30:
                #     break

            # now_lr = lr_scheduler.decay_lr(epoch, now_lr)
            torch.save(self.model.state_dict(), self.checkpt_last)

        end_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_time)
        
        start_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nStart testing at', start_test_time)
        
        print('\nbest test')
        test_loss, test_acc = self.test(self.checkpt)
        print('Load {}th epoch'.format(best_epoch))
        print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))
        
        #! print('\nlast test')
        test_last_loss, test_last_acc = self.test(self.checkpt_last)
        #! print('test loss:{:.3f}'.format(test_last_loss), 'acc:{:.2f}'.format(test_last_acc))
        
        end_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_test_time)

        writer = SummaryWriter(f"./results/tensor_log/{self.board_name}")
        
        for epoch, (acc, loss) in enumerate(train_logs):
            writer.add_scalar('Acc/train', acc, epoch)
            writer.add_scalar('Loss/train', loss, epoch)

        for epoch, acc, loss, gap in valid_logs:
            writer.add_scalar('Acc/val', acc, epoch)
            writer.add_scalar('Loss/val', loss, epoch)
            writer.add_scalar('Generalization_GAP', gap, epoch)

        log_filename = './results/' + dataset + '_log_new.csv'
        with open(log_filename, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([self.method, model_name, dataset, '-', '-', '-', '-', self.thr_init_score, self.K, self.scale_factor, self.target_func, test_acc, best_gap, test_last_acc, train_acc-valid_acc, self.log_time])
            
        return start_time, end_test_time