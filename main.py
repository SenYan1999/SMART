import os
import torch

from args import args
from utils import Trainer, GlueDataset, init_logger
from model import Bert, PGD, BPP
from torch.utils.data import DataLoader

def preprocess():
    # get dataset
    train_dataset = GlueDataset(args.data_dir, args.task, args.max_len, args.bert_type, mode='train')
    dev_dataset = GlueDataset(args.data_dir, args.task, args.max_len, args.bert_type, mode='dev')

    # save dataset
    torch.save(train_dataset, os.path.join(args.data_dir, args.task, 'train.pt'))
    torch.save(dev_dataset, os.path.join(args.data_dir, args.task, 'dev.pt'))

def train(logger):
    # prepare dataset and dataloader
    train_dataset = torch.load(os.path.join(args.data_dir, args.task, 'train.pt'))
    dev_dataset = torch.load(os.path.join(args.data_dir, args.task, 'dev.pt'))
    train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
                                        DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    # define model and optimzier
    model = Bert(args.bert_type, train_dataset.num_class)
    pgd = PGD(model, args.epsilon, args.alpha)
    bpp = BPP(model, args.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define trainer and begin training
    trainer = Trainer(train_dataloader, eval_dataloader, model, pgd, args.K, bpp, optimizer, args.task, logger, args.normal)
    trainer.train(args.num_epoch, args.save_path)

if __name__ == '__main__':
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # define logger file
    logger = init_logger(os.path.join(args.log_path, 'log.log'))

    if args.do_prepare:
        preprocess()

    if args.do_train:
        train(logger)

    if not(args.do_train or args.do_prepare):
        print('Nothing have done!')
