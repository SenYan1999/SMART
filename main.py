import os
import torch

from args import args
from utils import Trainer, GlueDataset, init_logger
from apex import amp
from model import Bert, PGD, BPP
from torch.utils.data import DataLoader

def preprocess(logger):
    # get dataset
    train_dataset = GlueDataset(args.data_dir, args.task, args.max_len, args.bert_name, args.bert_type, mode='train')
    dev_dataset = GlueDataset(args.data_dir, args.task, args.max_len, args.bert_name, args.bert_type, mode='dev')

    # save dataset
    torch.save(train_dataset, os.path.join(args.data_dir, args.task, 'train.pt'))
    torch.save(dev_dataset, os.path.join(args.data_dir, args.task, 'dev.pt'))

def train(logger):
    # prepare dataset and dataloader
    train_dataset = torch.load(os.path.join(args.data_dir, args.task, 'train.pt'))
    dev_dataset = torch.load(os.path.join(args.data_dir, args.task, 'dev.pt'))

    # get mini dataset
    if args.restrict_dataset:
        if len(train_dataset) > 100000:
            logger.info('Train_size: %d' % len(train_dataset))
            train_size = int(0.1 * len(train_dataset))
            train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
            dev_size = int(0.1 * len(dev_dataset))
            dev_dataset, _ = torch.utils.data.random_split(dev_dataset, [dev_size, len(dev_dataset) - dev_size])

    # define model and optimzier
    model = Bert(args.bert_name, train_dataset.num_class, args.bert_type)
    pgd = PGD(model, args.epsilon, args.alpha)
    bpp = BPP(model, args.beta, args.mu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler

        logger.info('Lets initialize nccl model.')
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        logger.info('Lets use %d GPUs!' % torch.cuda.device_count())
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)

        train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=DistributedSampler(train_dataset)), \
                                            DataLoader(dev_dataset, batch_size=args.batch_size, sampler=DistributedSampler(dev_dataset))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
                                            DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    if args.fp16:
        amp_model = amp.initialize(model, optimizer, opt_level='O1')
    else:
        amp_model = None

    # define trainer and begin training
    trainer = Trainer(train_dataloader, eval_dataloader, model, pgd, args.K, bpp, optimizer, args.task, logger, \
                      args.normal, args.fp16, device, amp_model)
    trainer.train(args.num_epoch, args.save_path)

if __name__ == '__main__':
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # args.distributed = True
    # args.do_train = True
    # args.fp16 = True
    # args.normal = True
    # args.bert_name = 'albert-xxlarge-v2'
    # args.bert_type= 'albert'

    # define logger file
    logger = init_logger(args.log_path)

    if args.do_prepare:
        preprocess(logger)

    if args.do_train:
        train(logger)

    if not(args.do_train or args.do_prepare):
        print('Nothing have done!')
