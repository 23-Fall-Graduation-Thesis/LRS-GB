from update.UpdateBase import UpdateBase

class Ours(UpdateBase):
    def __init__(self, optimizer):
        self.optim = optimizer

    def update_lr(self):
        # not update
        return self.optim
