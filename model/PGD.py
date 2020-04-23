import torch

class PGD(object):
    def __init__(self, model, epsilon, alpha):
        super(PGD, self).__init__()
        self.embed_bak = {}
        self.grad_bak = {}
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def attack(self, emb_name='embeddings.', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.embed_bak[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        r = param_data - self.embed_bak[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.embed_bak[param_name] + r

    def restore(self, emb_name = 'embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.embed_bak
                param.data = self.embed_bak[name]
        self.embed_bak = {}

    def grad_backup(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_bak[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.grad_bak
                param.grad = self.grad_bak[name]
