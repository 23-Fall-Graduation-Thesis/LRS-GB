from trainer.TrainerBase import TrainerBase
from scheduler.LRS_GB import LRS_GB

import copy, torch
from datetime import datetime
from utils.lr_utils import compute_weight_variation

class LRS_GB_Trainer(TrainerBase):
    def __init__(self, model, model_name, device, loaders, loggers, max_f, min_f, constraints, thr_score=0.94):
        super().__init__(model, model_name, device, loaders, loggers)
        self.pretrain_model = copy.deepcopy(model) 
        self.max_f = max_f
        self.min_f = min_f
        self.constraints = constraints
        self.thr_score = thr_score

    def train_model(self, epochs, init_lr):
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
    
        self.model.train()
        init_weva_success = []
        weva_success = []
        lr_success = []
        ntrial_success = []

        lr_scheduler = LRS_GB(self.model, self.model_name, init_lr, self.max_f, self.min_f, self.thr_score, self.constraints) #TODO instance arg_parser
        self.optimizer = lr_scheduler.optimizer_binding(self.model, [init_lr])
        
        best = 99999999
        best_epoch = 0
        bad_count = 0
        strict = False

        for epoch in range(epochs):
            print('Epoch:{:04d}'.format(epoch+1))
            # if not strict :
            #     # decreasing thr_score 
            #     if epoch >= 1 and lr_scheduler.thr_score > 0.8:
            #         lr_scheduler.thr_score = lr_scheduler.thr_score*0.99

            Trial_error = True
            Auto_update = False
            GB_update = False   # if GB considerate, then directly update.
            
            trial = 0
            init_weva_table = []
            weva_table = []
            lr_table = []

            now_lr = lr_scheduler.get_lr(self.optimizer)

            while Trial_error:
                trial = trial + 1
                model_temp = copy.deepcopy(self.model)
                model_try = copy.deepcopy(self.model)
                optimizer_try = lr_scheduler.optimizer_binding(model_try, now_lr)
                train_loss, train_acc, model_try = self.train_1epoch(model_try, optimizer_try)
                weva_try = compute_weight_variation(model_temp, model_try, lr_scheduler.layer_name_dict)
                init_weva_try = compute_weight_variation(self.pretrain_model, model_try, lr_scheduler.layer_name_dict)
                check_autoLR, check_GB, score  = lr_scheduler.try_lr_update(weva_try, init_weva_try)
                
                optimizer_try_lrs = lr_scheduler.get_lr(optimizer_try)
                if not GB_update:
                    if not check_autoLR :
                        Auto_update = True
                        weva_table.append(weva_try)
                        lr_table.append(optimizer_try_lrs)
                        now_lr = lr_scheduler.adjustLR(weva_table, init_weva_try, lr_table, epoch, GB_update=False)
                    
                    else:
                        if Auto_update:
                            self.model = copy.deepcopy(model_try)
                            self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                            weva_success.append(copy.deepcopy(weva_try))
                            lr_success.append(optimizer_try_lrs)
                            ntrial_success.append(trial)
                        
                        if not check_GB:
                            GB_update = True
                            init_weva_table.append(init_weva_try)
                            lr_table.append(optimizer_try_lrs)
                            now_lr = lr_scheduler.adjustLR(weva_table, init_weva_try, lr_table, epoch, GB_update=True)
                        else :
                            Trial_error = False
                            self.model = copy.deepcopy(model_try)
                            self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                            init_weva_success.append(copy.deepcopy(init_weva_try))
                            lr_success.append(optimizer_try_lrs)
                            ntrial_success.append(trial)
                # if Trial_error == False :
                #     # Success (score >= threshold score)
                #     self.model = copy.deepcopy(model_try)
                #     self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                #     weva_success.append(copy.deepcopy(weva_try))
                #     lr_success.append(optimizer_try_lrs)
                #     ntrial_success.append(trial)
                # else:
                #     weva_table.append(weva_try)
                #     lr_table.append(optimizer_try_lrs)
                #     now_lr = lr_scheduler.adjustLR(weva_table, lr_table, epoch)
                #     # now_lr.insert(0, now_lr[0]*self.conv1_factor) # for base_params (pruned layers) -> 우리는 base params 없다고 가정


                print('trial: {}, score: {}, check_GB: {}, Train Loss: {:.8f} Acc: {:.8f}'.format(trial, score, check_GB, train_loss, train_acc))

                epoLfmt = ['{:.6f}']*(len(weva_try)-1)
                epoLfmt =' '.join(epoLfmt)
                values = []
                for i in range(len(weva_try)-1):
                    values.append(weva_try[i])
                epoLfmt = '    WeVa :' + epoLfmt
                print(epoLfmt.format(*values))

                if Trial_error == True:
                    de_weva = lr_scheduler.target_weva_set[-1]
                    de_init_weva = lr_scheduler.target_init_weva_set[-1]
                    epoLfmt = ['{:.6f}'] * len(de_weva)
                    epoLfmt = ' '.join(epoLfmt)
                    values = []
                    for i in range(len(de_weva)):
                        values.append(de_weva[i])
                    epoLfmt = ' desWeVa :' + epoLfmt
                    print(epoLfmt.format(*values))
                    
                    epoinitLfmt = ['{:.6f}'] * len(de_init_weva)
                    epoinitLfmt = ' '.join(epoinitLfmt)
                    values = []
                    for i in range(len(de_init_weva)):
                        values.append(de_init_weva[i])
                    epoinitLfmt = 'initWeVa :' + epoinitLfmt
                    print(epoinitLfmt.format(*values))

                epoLfmt = ['{:.6f}'] * (len(optimizer_try_lrs)-1)
                epoLfmt = ' '.join(epoLfmt)
                values = []
                for i in range(len(optimizer_try_lrs)-1):
                    values.append(optimizer_try_lrs[i])
                epoLfmt = '      LR :' + epoLfmt
                print(epoLfmt.format(*values))
                print()
                

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            lr_scheduler.weva_manager.reset_trial()
            
            if epoch % 5 == 0:
                valid_loss, valid_acc = self.validation()
                print('validation loss:{:.3f}'.format(valid_loss), 'acc:{:.2f}'.format(valid_acc))
                self.writer.add_scalar('Loss/val', valid_loss, epoch)
                self.writer.add_scalar('Acc/val', valid_acc, epoch)
                self.writer.add_scalar('Generalization_GAP', train_acc - valid_acc, epoch)
                print()

            now_lr = lr_scheduler.decay_lr(epoch, now_lr)
            
            if valid_loss <= best:
                best = valid_loss
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.checkpt) 
                bad_count = 0
            else:
                bad_count += 1
            
            if bad_count == 30:
                break

        end_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nFinish training at', end_time)
        
        start_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart testing at', start_test_time)
        
        test_loss, test_acc = self.test()
        print('Load {}th epoch'.format(best_epoch))
        print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))
        
        end_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nFinish training at', end_test_time)
        return start_time, end_test_time
