from trainer.TrainerBase import TrainerBase

import torch, csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Standard_Trainer(TrainerBase):
    def __init__(self, model, conf, loaders, loggers):
        super().__init__(model, conf['model'], conf['device'], loaders, loggers)

    def train_model(self, epochs, init_lr):
        # loss, optimizer define
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
            
        end_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_time)
        
        start_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nStart testing at', start_test_time)
        
        test_loss, test_acc = self.test()
        print('Load {}th epoch'.format(best_epoch))
        print('test loss:{:.3f}'.format(test_loss), 'acc:{:.2f}'.format(test_acc))
        
        end_test_time = datetime.now().strftime('%m-%d_%H%M%S')
        #print('\nFinish training at', end_test_time)

        model_name = self.board_name.split('/')[1]
        dataset = self.board_name.split('/')[2].split('_')[0]
        

        with open(f"./results/log.csv", 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([model_name, dataset, 'standard', init_lr, '-', '-', '-', '-', '-', '-', best, valid_acc, test_acc, best_gap, train_acc-valid_acc, self.log_time])


        return start_time, end_test_time

        