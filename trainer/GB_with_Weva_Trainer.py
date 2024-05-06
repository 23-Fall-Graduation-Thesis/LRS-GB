from trainer.TrainerBase import TrainerBase
from scheduler.GB_with_weva import GB_with_Weva
from Manager import Manager

import copy, torch, math, csv
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from utils.lr_utils import compute_weight_variation, get_num_layer, compute_weight_difference_and_variation, compute_L1_weight_variation, compute_L1_weight_difference_and_variation, increase_K

class GB_with_Weva_Trainer(TrainerBase):

    def __init__(self, model, conf, loaders, loggers):
        super().__init__(model, conf['model'], conf['device'], loaders, loggers)
        self.pretrain_model = copy.deepcopy(model) 
        self.thr_init_score = conf['thr_init_score']
        self.K = conf['K']
        self.scale_factor = conf['scale_factor']
        self.bound = conf['bound']
        self.increase_bound = conf['increase_bound']
        self.all_epoch = conf['epoch']
        
        if conf['norm'] == 'L1' :
            self.get_weva =  compute_L1_weight_variation
            self.get_weva_and_diff = compute_L1_weight_difference_and_variation
        elif conf['norm'] == 'L2':
            self.get_weva =  compute_weight_variation
            self.get_weva_and_diff = compute_weight_difference_and_variation

    def train_model(self, epochs, init_lr):
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
    
        self.model.train()
        # weva_success = []
        weva_success = []
        lr_success = []
        ntrial_success = []

        lr_scheduler = GB_with_Weva(self.model, self.model_name, init_lr, self.thr_init_score, self.K, self.scale_factor, self.bound, self.all_epoch) #TODO instance arg_parser
        self.optimizer = lr_scheduler.optimizer_binding(self.model, [init_lr])
        
        best = 0
        best_epoch = 0
        bad_count = 0
        strict = False

        train_logs = []
        valid_logs = []
        manager = Manager(self.board_name.replace('/', '_'), 9)

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
                trial = trial + 1
                model_temp = copy.deepcopy(self.model)
                model_try = copy.deepcopy(self.model)
                optimizer_try = lr_scheduler.optimizer_binding(model_try, now_lr)
                train_loss, train_acc, model_try = self.train_1epoch(model_try, optimizer_try)
    
                # weight variation and difference between pretrained model
                # if epoch == 1:
                #     model_try = self.model
                weva_try, diff_try, param_num_list = self.get_weva_and_diff(model_temp, model_try, lr_scheduler.layer_name_dict)
                init_weva_try, init_diff_try, param_num_list = self.get_weva_and_diff(self.pretrain_model, model_try, lr_scheduler.layer_name_dict)
                init_diff_table.append(init_diff_try)

                # if never caculated target init weva, 
                lr_scheduler.set_initial_weva(init_diff_table, epoch, param_num_list)
                Trial_error, init_score = lr_scheduler.try_lr_update(weva_try)

                
                # check loss NaN
                if math.isnan(train_loss):
                    print('WARNING: non-finite loss, ending training ')
                    model_name = self.board_name.split('/')[1]
                    dataset = self.board_name.split('/')[2].split('_')[0]
                    with open(f"./results/nan.csv", 'a', newline='') as f:
                        wr = csv.writer(f)
                        # wr.writerow([self.model_name, self.dataset])
                        wr.writerow([model_name, dataset, 'GB', '-', '-', '-', self.K, self.scale_factor, self.bound, self.thr_init_score, self.log_time])
                    exit()

                optimizer_try_lrs = lr_scheduler.get_lr(optimizer_try)

                # Success 
                if Trial_error == False:
                    self.model = copy.deepcopy(model_try)
                    train_logs.append([train_acc, train_loss])
                    self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                    weva_success.append(copy.deepcopy(weva_try))
                    lr_success.append(optimizer_try_lrs)
                    ntrial_success.append(trial)
                    manager.record(epoch, now_lr[:-1], weva_try[:-1])
                else:
                    weva_table.append(weva_try)
                    init_weva_table.append(init_weva_try)
                    lr_table.append(optimizer_try_lrs)
                    now_lr = lr_scheduler.adjustLR(weva_table, init_diff_table, lr_table, epoch, param_num_list)

                print('trial: {}, init score: {:.4f}, Train Loss: {:.8f} Acc: {:.8f}'.format(trial, init_score, train_loss, train_acc))

                epoLfmt = ['{:.6f}']*(len(weva_try)-1)
                epoLfmt =' '.join(epoLfmt)
                values = []
                for i in range(len(weva_try)-1):
                    values.append(weva_try[i])
                epoLfmt = '       TryWeVa :' + epoLfmt
                print(epoLfmt.format(*values))

                epoinitLfmt = ['{:.6f}']*(len(init_weva_try)-1)
                epoinitLfmt =' '.join(epoinitLfmt)
                values = []
                for i in range(len(init_weva_try)-1):
                    values.append(init_weva_try[i])
                epoinitLfmt = '   TryInitWeVa :' + epoinitLfmt
                print(epoinitLfmt.format(*values))

                de_weva = lr_scheduler.target_weva_set[-1]
                epoinitLfmt = ['{:.6f}'] * len(de_weva)
                epoinitLfmt = ' '.join(epoinitLfmt)
                values = []
                for i in range(len(de_weva)):
                    values.append(de_weva[i])
                epoinitLfmt = '    TargetWeVa :' + epoinitLfmt
                print(epoinitLfmt.format(*values))

                epoLfmt = ['{:.6f}'] * (len(optimizer_try_lrs)-1)
                epoLfmt = ' '.join(epoLfmt)
                values = []
                for i in range(len(optimizer_try_lrs)-1):
                    values.append(optimizer_try_lrs[i])
                epoLfmt = '  LearningRate :' + epoLfmt
                print(epoLfmt.format(*values))

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

            now_lr = lr_scheduler.decay_lr(epoch, now_lr)

        end_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_time)
        
        start_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nStart testing at', start_test_time)
        
        test_loss, test_acc = self.test()
        print('Load {}th epoch'.format(best_epoch))
        print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))
        
        end_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_test_time)

        writer = SummaryWriter(f"./results/log/{self.board_name}")
        
        for epoch, (acc, loss) in enumerate(train_logs):
            writer.add_scalar('Acc/train', acc, epoch)
            writer.add_scalar('Loss/train', loss, epoch)

        for epoch, acc, loss, gap in valid_logs:
            writer.add_scalar('Acc/val', acc, epoch)
            writer.add_scalar('Loss/val', loss, epoch)
            writer.add_scalar('Generalization_GAP', gap, epoch)

        model_name = self.board_name.split('/')[1]
        dataset = self.board_name.split('/')[2].split('_')[0]

        with open(f"./results/log.csv", 'a', newline='') as f:
            wr = csv.writer(f)
            if self.increase_bound:
                mode='GB_with_Weva_increase'
            else:
                mode='GB_with_Weva'
            wr.writerow([model_name, dataset, mode, '-', '-', '-', self.K, self.scale_factor, self.bound, self.thr_init_score, best, valid_acc, test_acc, best_gap, train_acc-valid_acc, self.log_time])

        return start_time, end_test_time