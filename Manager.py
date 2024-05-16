from torch.utils.tensorboard import SummaryWriter

class Manager:
    def __init__(self, log_name, block_num):
        self.log_name = log_name
        self.block_num = block_num
        self.writers = []
        for block in range(block_num):
            self.writers.append(SummaryWriter(f'./results/logs/{log_name}/block{block}'))        
    
    def record(self, epoch, lr_list, weva_list):
        for idx, (lr, weva) in enumerate(zip(lr_list, weva_list)):
            self.writers[idx].add_scalar(f'{self.log_name}/lr', lr, epoch)
            self.writers[idx].add_scalar(f'{self.log_name}/weva', weva, epoch)