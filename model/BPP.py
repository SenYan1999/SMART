import torch
import torch.nn.functional as F

class BPP(object):
    def __init__(self, model, beta, mu):
        self.model = model
        self.beta = beta
        self.mu = mu
        self.theta_til = {}
        for name, param in self.model.named_parameters():
            self.theta_til[name] = param.data.clone().cuda()

    def theta_til_backup(self, named_parameters):
        for name, param in named_parameters:
            self.theta_til[name] = (1-self.beta) * param.data.clone() + self.beta * self.theta_til[name]

    def bregman_divergence(self, batch, logits):
        theta_prob = F.softmax(logits, dim=-1)

        param_bak = {}
        for name, param in self.model.named_parameters():
            param_bak[name] = param.data.clone()
            param.data = self.theta_til[name]

        with torch.no_grad():
            theta_til_prob = F.softmax(self.model(*batch), dim=-1)

        for name, param in self.model.named_parameters():
            param.data = param_bak[name]

        for name, param in self.model.named_parameters():
            param.data = param_bak[name]

        bregman_divergence = F.kl_div(theta_prob.log(), theta_til_prob, reduction='batchmean') + \
            F.kl_div(theta_til_prob.log(), theta_prob, reduction='batchmean')

        return bregman_divergence
