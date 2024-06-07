import copy
from trainer.TrainerBase import TrainerBase

import torch, csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.lr_utils import layer_block_info
from utils.lr_utils import compute_weight_variation, get_num_layer, compute_weight_difference_and_variation, compute_L1_weight_variation, compute_L1_weight_difference_and_variation, increase_K


class Standard_Trainer(TrainerBase):
    def __init__(self, model, conf, loaders, loggers):
        super().__init__(model, conf['model'], conf['device'], loaders, loggers)
        self.layer_name_list = layer_block_info(self.model_name)
        self.layer_name_dict = dict()
        for idx in range(len(self.layer_name_list)):
            for layer_name in self.layer_name_list[idx]:
                self.layer_name_dict[layer_name] = idx
        self.pretrain_model = copy.deepcopy(model) 
        
        if conf['norm'] == 'L1' :
            self.get_weva =  compute_L1_weight_variation
            self.get_weva_and_diff = compute_L1_weight_difference_and_variation
        elif conf['norm'] == 'L2':
            self.get_weva =  compute_weight_variation
            self.get_weva_and_diff = compute_weight_difference_and_variation
        
    def train_model(self, epochs, init_lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr)
        self.writer = SummaryWriter(f"./results/log/{self.board_name}")
        
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
        
        best = 0
        best_epoch = 0
        bad_count = 0
        for epoch in range(epochs):
            train_loss, train_acc = self.train()
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            print('Epoch:{:04d}'.format(epoch+1), 'train loss:{:.3f}'.format(train_loss), 'acc:{:.2f}'.format(train_acc))

            if epoch % 5 == 0:
                valid_loss, valid_acc = self.validation()
                self.writer.add_scalar('Loss/val', valid_loss, epoch)
                self.writer.add_scalar('Acc/val', valid_acc, epoch)
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
            torch.save(self.model.state_dict(), self.checkpt_last)
        
        
        init_weva, init_diff, param_num_list = self.get_weva_and_diff(self.pretrain_model, self.model, self.layer_name_dict)

        end_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_time)
        
        start_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nStart testing at', start_test_time)
        
        print('\nbest test')
        test_loss, test_acc = self.test(self.checkpt)
        print('Load {}th epoch'.format(best_epoch))
        print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))
        
        print('\nlast test')
        test_last_loss, test_last_acc = self.test(self.checkpt_last)
        print('test loss:{:.3f}'.format(test_last_loss), 'acc:{:.2f}'.format(test_last_acc))
        
        end_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_test_time)

        model_name = self.board_name.split('/')[0]
        dataset = self.board_name.split('/')[1]
        
        log_filename = './results/' + dataset + '_log_new.csv'
        with open(log_filename, 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['standard', model_name, dataset, init_lr, '-', '-', '-', '-', '-', '-', '-', test_acc, best_gap, test_last_acc, train_acc-valid_acc, init_weva, self.log_time])


        return start_time, end_test_time

        