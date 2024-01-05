from update.UpdateBase import UpdateBase

class Ours(UpdateBase):
    def __init__(self, optimizer):
        self.optim = optimizer

    # TODO : our methods, like AutoLR.py
    # 여기 안에서 Clip, target weight varation 계산, lr 계산 등 진행