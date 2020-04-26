import torch
import torch.nn.functional as F
import numpy as np
import os

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, accuracy_score

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, model, pgd, pgd_k, bpp, optimizer, task, logger, normal, distributed):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.model = model
        self.pgd = pgd
        self.pgd_k = pgd_k
        self.bpp = bpp
        self.optimizer = optimizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.task = task
        self.logger = logger
        self.normal = normal

        if distributed:
            self.logger.info('Lets use %d GPUs!' % torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)

        self.metrics = {'CoLA': 'MCC', 'QNLI': 'ACC', 'SST-2': 'ACC', 'MNLI': 'ACC', 'WNLI': 'ACC', 'QQP': 'acc',\
                        'MRPC': 'ACC', 'RTE': 'ACC'}

    def calculate_result(self, pred, truth):
        pred = torch.argmax(pred, dim=-1).detach().cpu().numpy().astype(np.float)
        truth = truth.detach().cpu().numpy().astype(np.float)

        if self.task == 'cola':
            result = matthews_corrcoef(truth, pred)
        elif self.task in ['QNLI', 'SST-2', 'WNLI', 'QQP', 'MNLI', 'MRPC', 'RTE']:
            result = accuracy_score(truth, pred)
        else:
            raise('Task error!')

        return result

    def train_epoch_smart(self, epoch):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data))
        self.model.train()

        losses, accs = [], []
        self.bpp.theta_backup()
        for batch in self.train_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.nll_loss(out, labels)
            loss.backward()

            # PGD Adversarial Training
            self.pgd.grad_backup()
            for k in range(self.pgd_k):
                self.pgd.attack(is_first_attack=(k == 0))
                if k != self.pgd_k - 1:
                    self.model.zero_grad()
                else:
                    self.pgd.restore_grad()
                out_adv = self.model(input_ids, attention_mask, token_type_ids)
                loss_adv = F.nll_loss(out_adv, labels)
                loss_adv.backward()

            self.pgd.restore()

            self.optimizer.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            acc = self.calculate_result(out, labels)
            accs.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))
            pbar.update(1)

        # Bregman Proximal Point Optimization
        self.bpp.theta_update()

        pbar.close()
        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def train_epoch_normal(self, epoch):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data))
        self.model.train()

        losses, accs = [], []
        for batch in self.train_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.nll_loss(out, labels)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            acc = self.calculate_result(out, labels)
            accs.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))
            pbar.update(1)

        pbar.close()
        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def evaluate_epoch(self, epoch):
        self.logger.info('Epoch: %2d: Evaluating Model...' % epoch)
        self.model.eval()

        losses, precise, recall, f1s, accs = [], [], [], [], []
        for batch in self.dev_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)

            with torch.no_grad():
                out = self.model(input_ids, attention_mask, token_type_ids)
            loss = F.nll_loss(out, labels)

            acc = self.calculate_result(out, labels)
            losses.append(loss.item())
            accs.append(acc)

        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def train(self, num_epoch, save_path):
        for epoch in range(num_epoch):
            if self.normal:
                self.train_epoch_normal(epoch)
            else:
                self.train_epoch_smart(epoch)
            self.evaluate_epoch(epoch)

            # save state dict
            path = os.path.join(save_path, 'state_%d_epoch.pt' % epoch)
            self.save_dict(path)
            self.logger.info('')

    def save_dict(self, save_path):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state_dict, save_path)

    def load_dict(self, path):
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
