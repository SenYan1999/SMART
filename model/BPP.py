import torch
import torch.nn.functional as F

class BPP(object):
    def __init__(self, model, beta, mu):
        self.model = model.clone()
        self.beta = beta
        self.mu = mu

    def theta_til_backup(self, named_parameters):
        for (name_pbb, param_pbb), (name_model, param_model) in zip(self.model.named_parameters(), named_parameters):
            assert name_pbb == name_model
            param_pbb.data = (1-self.beta) * param_model.data.clone() + self.beta * param_pbb.data

    def bregman_divergence(self, batch, logits):
        theta_prob = F.softmax(logits, dim=-1)
        theta_til_prob = F.softmax(self.model(*batch), dim=-1)

        bregman_divergence = F.kl_div(theta_prob.log(), theta_til_prob, reduction='batchmean') + \
            F.kl_div(theta_til_prob.log(), theta_prob, reduction='batchmean')

        return bregman_divergence
