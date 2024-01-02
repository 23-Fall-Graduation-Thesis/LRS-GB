from Trainer.TrainerBase import TrainerBase
from update.AutoLR import AutoLR
import copy, torch
import torch.nn as nn
from datetime import datetime

class AutoLR_Trainer(TrainerBase):
    def __init__(self, model, device, trainloader, validloader, testloader, checkpt, board_name, writer, max_f, min_f):
        super().__init__(model, device, trainloader, validloader, testloader, checkpt, board_name, writer)
        self.conv1_factor = 0.5 ## ?
        self.max_f = max_f
        self.min_f = min_f

    def train_model(self, epochs, init_lr):
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
   
        self.model.train()
        weva_success = []
        lr_success = []
        ntrial_success = []

        lr_updater = AutoLR(self.model, init_lr, self.max_f, self.min_f)
        self.optimizer = lr_updater.optimizer_binding(self.model, [init_lr])
        
        best = 99999999
        best_epoch = 0
        bad_count = 0

        for epoch in range(epochs):
            # TODO
            # if not strict :
            #     # decreasing thr_score 
            #     if epoch >= 1 and thr_score > 0.8:
            #         thr_score = thr_score*0.99

            Trial_error = True

            trial = 0
            weva_table = []
            lr_table = []

            now_lr = lr_updater.get_lr(self.optimizer)

            while Trial_error:
                trial = trial + 1
                model_try = copy.deepcopy(self.model)
                optimizer_try = lr_updater.optimizer_binding(model_try, now_lr)
                weva_try, train_loss, train_acc = self.train_1epoch(model_try, optimizer_try, lr_updater.layer_names)
                # lr updater에 함수 만들기
                # model_pre = copy.deepcopy(model_try)
                Trial_error, score, now_lr = lr_updater.try_lr_update(weva_try, epoch, now_lr)
                optimizer_try_lrs = lr_updater.get_lr(optimizer_try)
                print_lr = optimizer_try_lrs[1:]

                if Trial_error == False:
                    # Success (score >= threshold score)
                    self.model = copy.deepcopy(model_try)
                    self.optimizer = lr_updater.optimizer_binding(self.model, now_lr)
                    weva_success.append(copy.deepcopy(weva_try))
                    lr_success.append(optimizer_try_lrs)
                    ntrial_success.append(trial)
                else:
                    weva_table.append(weva_try)
                    lr_table.append(optimizer_try_lrs)
                    now_lr = lr_updater.adjustLR(weva_table, lr_table, score, epoch)
                    # now_lr.insert(0, now_lr[0]*self.conv1_factor) # for base_params (pruned layers) -> 우리는 base params 없다고 가정

                #Print current state
                # train_acc = train_acc.float()
                epoch_loss = train_loss /  len(self.trainloader.dataset)
                epoch_acc = train_acc /  len(self.trainloader.dataset)

                print('trial: {}, score: {}, Train Loss: {:.8f} Acc: {:.8f}'.format(trial, score,
                    epoch_loss, epoch_acc))

                weva_try_print = weva_try[1:-3]
                weva_try_print.append(weva_try[-1])

                epoLfmt = ['{:.6f}']*(len(weva_try_print)-1)
                epoLfmt =' '.join(epoLfmt)
                values = []
                for i in range(len(weva_try_print)-1):
                    values.append(weva_try_print[i])
                epoLfmt = '   WeVa :' + epoLfmt
                print(epoLfmt.format(*values))

                if Trial_error == True:
                    de_weva = lr_updater.desired_weva_set[-1]
                    epoLfmt = ['{:.6f}'] * len(de_weva)
                    epoLfmt = ' '.join(epoLfmt)
                    values = []
                    for i in range(len(de_weva)):
                        values.append(de_weva[i])
                    epoLfmt = 'desWeVa :' + epoLfmt
                    print(epoLfmt.format(*values))

                epoLfmt = ['{:.6f}'] * (len(print_lr)-1)
                epoLfmt = ' '.join(epoLfmt)
                values = []
                for i in range(len(print_lr)-1):
                    values.append(print_lr[i])
                epoLfmt = '     LR :' + epoLfmt
                print(epoLfmt.format(*values))
                print('Epoch:{:04d}'.format(epoch+1), 'train loss:{:.3f}'.format(train_loss), 'acc:{:.2f}'.format(train_acc))
                # print('test accuracy : top-1 {:.4f} top-2 {:.4f} top-4 {:.4f} top-8 {:.4f}'.format(results[0]*100,results[1]*100,results[2]*100,results[3]*100))

            if epoch % 5 == 0:
                valid_loss, valid_acc = self.validation()
                print('validation loss:{:.3f}'.format(valid_loss), 'acc:{:.2f}'.format(valid_acc))
                print()

            if valid_loss < best:
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
