from trainer.TrainerBase import TrainerBase
from scheduler.AutoLR import AutoLR
import copy, torch, math, csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.lr_utils import compute_weight_variation

from Manager import Manager

class AutoLR_Trainer(TrainerBase):
    def __init__(self, model, conf, loaders, loggers):
        super().__init__(model, conf['model'], conf['device'], loaders, loggers)
        self.max_f = conf['max_f']
        self.min_f = conf['min_f']
        self.thr_score = conf['thr_score']

    def train_model(self, epochs, init_lr):
        writer = SummaryWriter(f"./results/log/{self.board_name}")
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
    
        self.model.train()
        weva_success = []
        lr_success = []
        ntrial_success = []

        train_logs = []
        valid_logs = []

        # setting scheduler & optimizer
        lr_scheduler = AutoLR(self.model, self.model_name, init_lr, self.max_f, self.min_f, self.thr_score)
        self.optimizer = lr_scheduler.optimizer_binding(self.model, [init_lr])
        
        best = 0
        best_epoch = 0
        bad_count = 0
        strict = False

        manager = Manager(self.board_name.replace('/', '_'), 9)

        for epoch in range(epochs):
            print('Epoch:{:04d}'.format(epoch+1))
            # if not strict :
            #     # decreasing thr_score 
            #     if epoch >= 1 and lr_scheduler.thr_score > 0.8:
            #         lr_scheduler.thr_score = lr_scheduler.thr_score*0.99

            Trial_error = True

            trial = 0
            weva_table = []
            lr_table = []

            # current learning rate
            now_lr = lr_scheduler.get_lr(self.optimizer)

            # repeat until condition is satisfied
            while Trial_error:
                trial = trial + 1
                if trial>100:
                    Trial_error = False
                model_temp = copy.deepcopy(self.model)
                model_try = copy.deepcopy(self.model)
                
                # setting optimizer with splited layer-wise learning rate
                optimizer_try = lr_scheduler.optimizer_binding(model_try, now_lr)
                
                # try 1 epoch training
                train_loss, train_acc, model_try = self.train_1epoch(model_try, optimizer_try)
                
                # calculate weight variance using current model & tried model
                weva_try = compute_weight_variation(model_temp, model_try, lr_scheduler.layer_name_dict)
                
                # check trial condition and update current lr & get tried optimizer
                Trial_error, score, now_lr = lr_scheduler.try_lr_update(weva_try, epoch, now_lr)
                optimizer_try_lrs = lr_scheduler.get_lr(optimizer_try)

                # check loss NaN
                if math.isnan(train_loss):
                    print('WARNING: non-finite loss, ending training ')
                    model_name = self.board_name.split('/')[1]
                    dataset = self.board_name.split('/')[2].split('_')[0]
                    with open(f"./results/nan.csv", 'a', newline='') as f:
                        wr = csv.writer(f)
                        wr.writerow([model_name, dataset, 'auto', self.max_f, self.min_f, self.thr_score, self.log_time])
                    exit()

                # Success (score >= threshold score)
                if Trial_error == False:
                    # update tried model & optimizer
                    self.model = copy.deepcopy(model_try)
                    train_logs.append([train_acc, train_loss])
                    self.optimizer = lr_scheduler.optimizer_binding(self.model, now_lr)
                    weva_success.append(copy.deepcopy(weva_try))
                    lr_success.append(optimizer_try_lrs)
                    ntrial_success.append(trial)
                    manager.record(epoch, now_lr[:-1], weva_try[:-1])
                else:
                    weva_table.append(weva_try)
                    lr_table.append(optimizer_try_lrs)
                    
                    # update learning rate using AutoLR algorithm
                    now_lr = lr_scheduler.adjustLR(weva_table, lr_table, epoch)
                    # now_lr.insert(0, now_lr[0]*self.conv1_factor) # for base_params (pruned layers) -> 우리는 base params 없다고 가정


                print('trial: {}, score: {}, Train Loss: {:.8f} Acc: {:.8f}'.format(trial, score, train_loss, train_acc))

                # print current states
                epoLfmt = ['{:.6f}']*(len(weva_try)-1)
                epoLfmt =' '.join(epoLfmt)
                values = []
                for i in range(len(weva_try)-1):
                    values.append(weva_try[i])
                epoLfmt = '   WeVa :' + epoLfmt
                print(epoLfmt.format(*values))

                if Trial_error == True:
                    de_weva = lr_scheduler.target_weva_set[-1]
                    epoLfmt = ['{:.6f}'] * len(de_weva)
                    epoLfmt = ' '.join(epoLfmt)
                    values = []
                    for i in range(len(de_weva)):
                        values.append(de_weva[i])
                    epoLfmt = 'desWeVa :' + epoLfmt
                    print(epoLfmt.format(*values))

                epoLfmt = ['{:.6f}'] * (len(optimizer_try_lrs)-1)
                epoLfmt = ' '.join(epoLfmt)
                values = []
                for i in range(len(optimizer_try_lrs)-1):
                    values.append(optimizer_try_lrs[i])
                epoLfmt = '     LR :' + epoLfmt
                print(epoLfmt.format(*values))
                print()

            
            
            if epoch % 5 == 0 or epoch == epochs-1:
                valid_loss, valid_acc = self.validation()
                valid_logs.append([epoch, valid_acc, valid_loss, train_acc-valid_acc])
                
                print('validation loss:{:.3f}'.format(valid_loss), 'acc:{:.2f}'.format(valid_acc))
                print()

                if valid_acc >= best:
                    best = valid_acc
                    best_epoch = epoch + 1
                    best_gap = train_acc - valid_acc
                    torch.save(self.model.state_dict(), self.checkpt) 
                    bad_count = 0
                else:
                    bad_count += 1
                
                # if bad_count == 30:
                #     break


        end_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_time)
        
        start_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nStart testing at', start_test_time)
        
        test_loss, test_acc = self.test()
        print('Load {}th epoch'.format(best_epoch))
        print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))
        
        end_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_test_time)


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
            wr.writerow([model_name, dataset, 'auto', init_lr, self.max_f, self.min_f, self.thr_score, '-', '-', '-', best, valid_acc, test_acc, best_gap, train_acc-valid_acc, self.log_time])

        return start_time, end_test_time
        
