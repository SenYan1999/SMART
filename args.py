import argparse

parser = argparse.ArgumentParser()

# main mode
parser.add_argument('--do_prepare', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--normal', action='store_true')
parser.add_argument('--restrict_dataset', action='store_true')
parser.add_argument('--fp16', action='store_true')

# distributed
parser.add_argument('--local_rank', type=int, default=0)

# data prepare
parser.add_argument('--data_dir', type=str, default='glue_data')
parser.add_argument('--task', type=str, default='CoLA')
parser.add_argument('--max_len', type=int, default=150)

# model
parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
parser.add_argument('--bert_type', type=str, default='bert')

# PGD
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--epsilon', type=float, default=1e-5)
parser.add_argument('--alpha', type=float, default=0.3)

# BPP
parser.add_argument('--beta', type=float, default=0.8)
parser.add_argument('--mu', type=float, default=1)

# train
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=6)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--drop_out', type=float, default=0.1)

# save & log
parser.add_argument('--log_path', type=str, default='log/log.log')
parser.add_argument('--save_path', type=str, default='save_model/')

# parse args
args = parser.parse_args()
