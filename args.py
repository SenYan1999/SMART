import argparse

parser = argparse.ArgumentParser()

# main mode
parser.add_argument('--do_prepare', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--normal', action='store_true')

# data prepare
parser.add_argument('--data_dir', type=str, default='glue_data')
parser.add_argument('--task', type=str, default='cola')
parser.add_argument('--max_len', type=int, default=100)

# model
parser.add_argument('--bert_type', type=str, default='bert-base-uncased')

# PGD
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--epsilon', type=float, default=1.)
parser.add_argument('--alpha', type=float, default=0.3)

# BPP
parser.add_argument('--beta', type=float, default=0.8)

# train
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-5)

# save & log
parser.add_argument('--log_path', type=str, default='log/')
parser.add_argument('--save_path', type=str, default='save_model/')

# parse args
args = parser.parse_args()
