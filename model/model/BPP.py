import torch

class BPP(object):
    def __init__(self, model, beta):
        self.model = model
        self.beta = beta
        self.theta_bak = {}

    def theta_backup(self):
        for name, param in self.model.named_parameters():
            self.theta_bak[name] = param.data.clone()

    def theta_update(self):
        for name, param in self.model.named_parameters():
            assert name in self.theta_bak
            param.data = self.beta * param.data + (1 - self.beta) * self.theta_bak[name]
