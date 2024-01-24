from trainer.TrainerBase import TrainerBase
from scheduler.LRS_GB import LRS_GB, LRS_GB_Score

import copy, torch
from datetime import datetime
from utils.lr_utils import compute_weight_variation, get_num_layer, compute_weight_difference_and_variation

class LRS_GB_Trainer(TrainerBase):
    def __init__(self, model, conf, loaders, loggers):
        super().__init__(model, conf['model'], conf['device'], loaders, loggers)
        self.pretrain_model = copy.deepcopy(model) 
        self.max_f, self.min_f = conf['max_f'], conf['min_f']
        num_layer = get_num_layer(conf['model'])
        self.constraints = [conf['reg'] * (conf['scale'] ** idx) for idx in range(num_layer)]
        print(self.constraints)
        self.thr_score = conf['thr_score']

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

            # current learning rate
            now_lr = lr_scheduler.get_lr(self.optimizer)

            # repeat until condition is satisfied
            while Trial_error:
                trial = trial + 1
                model_temp = copy.deepcopy(self.model)
                model_try = copy.deepcopy(self.model)
                
                # setting optimizer with splited layer-wise learning rate
                optimizer_try = lr_scheduler.optimizer_binding(model_try, now_lr)
                
                # try 1 epoch training
                train_loss, train_acc, model_try = self.train_1epoch(model_try, optimizer_try)
                
                # calculate weight variance using current model & tried model
                weva_try = compute_weight_variation(model_temp, model_try, lr_scheduler.layer_name_dict)
                
                # calculate init weight variance using pretrained model & tried model
                init_weva_try = compute_weight_variation(self.pretrain_model, model_try, lr_scheduler.layer_name_dict)
                
                # check autoLR & GB condition, get sorting quality 
                check_autoLR, check_GB, score  = lr_scheduler.try_lr_update(weva_try, init_weva_try)
                
                # get tried optimizer
                optimizer_try_lrs = lr_scheduler.get_lr(optimizer_try)
                
                if not check_autoLR and not GB_update:
                    Auto_update = True
                    weva_table.append(weva_try)
                    lr_table.append(optimizer_try_lrs)
                    now_lr = lr_scheduler.adjustLR(weva_table, init_weva_try, lr_table, epoch, GB_update=False)
                
                else:
                    weva_table.append(weva_try)
                    if Auto_update:
                        self.model = copy.deepcopy(model_try)
                        self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                        weva_success.append(copy.deepcopy(weva_try))
                        lr_success.append(optimizer_try_lrs)
                        ntrial_success.append(trial)
                    
                    if not check_GB:
                        if not GB_update:
                            lr_scheduler.weva_manager.reset_trial()
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
                epoLfmt = '       WeVa :' + epoLfmt
                print(epoLfmt.format(*values))

                epoinitLfmt = ['{:.6f}']*(len(init_weva_try)-1)
                epoinitLfmt =' '.join(epoinitLfmt)
                values = []
                for i in range(len(init_weva_try)-1):
                    values.append(init_weva_try[i])
                epoinitLfmt = '   InitWeVa :' + epoinitLfmt
                print(epoinitLfmt.format(*values))


                if Trial_error == True:
                    de_weva = lr_scheduler.target_weva_set[-1]
                    de_init_weva = lr_scheduler.target_init_weva_set[-1]
                    epoLfmt = ['{:.6f}'] * len(de_weva)
                    epoLfmt = ' '.join(epoLfmt)
                    values = []
                    for i in range(len(de_weva)):
                        values.append(de_weva[i])
                    epoLfmt = '    desWeVa :' + epoLfmt
                    print(epoLfmt.format(*values))
                    
                    epoinitLfmt = ['{:.6f}'] * len(de_init_weva)
                    epoinitLfmt = ' '.join(epoinitLfmt)
                    values = []
                    for i in range(len(de_init_weva)):
                        values.append(de_init_weva[i])
                    epoinitLfmt = 'desInitWeVa :' + epoinitLfmt
                    print(epoinitLfmt.format(*values))

                epoLfmt = ['{:.6f}'] * (len(optimizer_try_lrs)-1)
                epoLfmt = ' '.join(epoLfmt)
                values = []
                for i in range(len(optimizer_try_lrs)-1):
                    values.append(optimizer_try_lrs[i])
                epoLfmt = '         LR :' + epoLfmt
                print(epoLfmt.format(*values))
                print()
                

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            
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

class LRS_GB_Score_Trainer(TrainerBase):

    def __init__(self, model, conf, loaders, loggers):
        super().__init__(model, conf['model'], conf['device'], loaders, loggers)
        self.pretrain_model = copy.deepcopy(model) 
        self.max_f, self.min_f = conf['max_f'], conf['min_f']
        self.thr_score = conf['thr_score']
        self.thr_init_score = conf['thr_init_score']
        self.K = conf['K']
        self.scale_factor = conf['scale_factor']
        self.max_trial = conf['max_trial']


    def train_model(self, epochs, init_lr):
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
    
        self.model.train()
        init_weva_success = []
        weva_success = []
        lr_success = []
        ntrial_success = []

        lr_scheduler = LRS_GB_Score(self.model, self.model_name, init_lr, self.max_f, self.min_f, self.thr_score, self.thr_init_score, self.K, self.scale_factor) #TODO instance arg_parser
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

            GB_best_score = 0.0
            GB_best_lr = None
            
            trial = 0
            init_weva_table = []
            weva_table = []
            lr_table = []

            now_lr = lr_scheduler.get_lr(self.optimizer)
            model_temp = copy.deepcopy(self.model)

            while Trial_error:
                trial = trial + 1
                model_try = copy.deepcopy(self.model)
                optimizer_try = lr_scheduler.optimizer_binding(model_try, now_lr)
                train_loss, train_acc, model_try = self.train_1epoch(model_try, optimizer_try)
                weva_try = compute_weight_variation(model_temp, model_try, lr_scheduler.layer_name_dict)
                init_weva_try, init_diff_try, param_num_list = compute_weight_difference_and_variation(self.pretrain_model, model_try, lr_scheduler.layer_name_dict)
                check_autoLR, check_GB, score, init_score = lr_scheduler.try_lr_update(weva_try, init_weva_try)
                
                optimizer_try_lrs = lr_scheduler.get_lr(optimizer_try)
                
                # if not check_autoLR and not GB_update:
                #     Auto_update = True
                #     weva_table.append(weva_try)
                #     lr_table.append(optimizer_try_lrs)
                #     now_lr = lr_scheduler.adjustLR(weva_table, init_weva_try, init_diff_try, lr_table, epoch, param_num_list, GB_update=False)
                # else:
                #     weva_table.append(weva_try)
                #     if Auto_update:
                #         self.model = copy.deepcopy(model_try)
                #         self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                        # weva_success.append(copy.deepcopy(weva_try))
                        # lr_success.append(optimizer_try_lrs)
                        # ntrial_success.append(trial)
                    
                #     if not check_GB:
                #         lr_scheduler.weva_manager.trial += 1
                #         if not GB_update:
                #             lr_scheduler.weva_manager.reset_trial()
                #         GB_update = True
                #         init_weva_table.append(init_weva_try)
                #         lr_table.append(optimizer_try_lrs)
                #         now_lr = lr_scheduler.adjustLR(weva_table, init_weva_try, init_diff_try, lr_table, epoch, param_num_list, GB_update=True)
                #     else:
                #         Trial_error = False
                #         self.model = copy.deepcopy(model_try)
                #         self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                #         init_weva_success.append(copy.deepcopy(init_weva_try))
                #         lr_success.append(optimizer_try_lrs)
                #         ntrial_success.append(trial)

                if not check_autoLR and not Auto_update:
                    # AutoLR update
                    weva_table.append(weva_try)
                    lr_table.append(optimizer_try_lrs)
                    now_lr = lr_scheduler.adjustLR(weva_table, init_weva_try, init_diff_try, lr_table, epoch, param_num_list, GB_update=False)
                    lr_scheduler.weva_manager.reset_trial()
                else:
                    weva_table.append(weva_try)
                    if not Auto_update:
                        # finish autoLR update, start GB update
                        Auto_update = True
                        self.model = copy.deepcopy(model_try)
                        self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                        GB_best_score = init_score
                        GB_best_lr = now_lr
                        weva_success.append(copy.deepcopy(weva_try))
                        lr_success.append(optimizer_try_lrs)
                        ntrial_success.append(trial)
                        lr_scheduler.weva_manager.reset_trial()
                        print("now GB update")
                    else:
                        # GB update
                        if not check_GB:
                            if GB_best_score < init_score:
                                GB_best_score = init_score
                                GB_best_lr = now_lr
                            init_weva_table.append(init_weva_try)
                            lr_table.append(optimizer_try_lrs)
                            now_lr = lr_scheduler.adjustLR(weva_table, init_weva_try, init_diff_try, lr_table, epoch, param_num_list, GB_update=True)
                            
                            if lr_scheduler.weva_manager.trial >= self.max_trial:
                                GB_update = True
                        if check_GB or GB_update: 
                            Trial_error = False
                            self.model = copy.deepcopy(model_try)
                            self.optimizer = lr_scheduler.optimizer_binding(self.model, GB_best_lr)
                            init_weva_success.append(copy.deepcopy(init_weva_try))
                            lr_success.append(optimizer_try_lrs)
                            ntrial_success.append(trial) 

                print('trial: {}, GB trial: {}, score: {:.4f}, init score: {:.4f}, Train Loss: {:.8f} Acc: {:.8f}'.format(trial, lr_scheduler.weva_manager.trial, score, init_score, train_loss, train_acc))

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


                if Trial_error == True:
                    de_weva = lr_scheduler.target_weva_set[-1]
                    de_init_weva = lr_scheduler.target_init_weva_set[-1]
                    epoLfmt = ['{:.6f}'] * len(de_weva)
                    epoLfmt = ' '.join(epoLfmt)
                    values = []
                    for i in range(len(de_weva)):
                        values.append(de_weva[i])
                    epoLfmt = '    TargetWeVa :' + epoLfmt
                    print(epoLfmt.format(*values))
                    
                    epoinitLfmt = ['{:.6f}'] * len(de_init_weva)
                    epoinitLfmt = ' '.join(epoinitLfmt)
                    values = []
                    for i in range(len(de_init_weva)):
                        values.append(de_init_weva[i])
                    epoinitLfmt = 'TargetInitWeVa :' + epoinitLfmt
                    print(epoinitLfmt.format(*values))

                epoLfmt = ['{:.6f}'] * (len(optimizer_try_lrs)-1)
                epoLfmt = ' '.join(epoLfmt)
                values = []
                for i in range(len(optimizer_try_lrs)-1):
                    values.append(optimizer_try_lrs[i])
                epoLfmt = '  LearningRate :' + epoLfmt
                print(epoLfmt.format(*values))
                print()
                

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            
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
