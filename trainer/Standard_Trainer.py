from Trainer.TrainerBase import TrainerBase

import torch
from datetime import datetime
import torch.nn as nn

class Standard_Trainer(TrainerBase):
    def __init__(self, model, model_name, device, loaders, loggers):
        super().__init__(self, model, model_name, device, loaders, loggers)

    def train_model(self, epochs, init_lr):
        # loss, optimizer define
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr)
        
        start_time = datetime.now().strftime('%m-%d_%H%M%S')
        print('\nStart training at', start_time)
        
        best = 99999999
        best_epoch = 0
        bad_count = 0
        for epoch in range(epochs):
            train_loss, train_acc = self.train()
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Acc/train', train_acc, epoch)
            print('Epoch:{:04d}'.format(epoch+1), 'train loss:{:.3f}'.format(train_loss), 'acc:{:.2f}'.format(train_acc))

            if epoch % 5 == 0:
                valid_loss, valid_acc = self.validation()
                print('validation loss:{:.3f}'.format(valid_loss), 'acc:{:.2f}'.format(valid_acc))
                print()
                self.writer.add_scalar('Loss/val', valid_loss, epoch)
                self.writer.add_scalar('Acc/val', valid_acc, epoch)
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

        