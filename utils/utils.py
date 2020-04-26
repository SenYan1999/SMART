import torch
import os
import logging

from torch.utils.data import Dataset
from logging import handlers
from transformers import BertTokenizer
from transformers.data.processors.glue import ColaProcessor, Sst2Processor, MnliProcessor, MrpcProcessor, QnliProcessor, \
    QqpProcessor, WnliProcessor, RteProcessor
from transformers.data.processors.glue import glue_convert_examples_to_features

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger

class GlueDataset(Dataset):
    def __init__(self, data_dir, task, max_len, bert_type, mode='train'):
        self.task = task
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)

        self.data, self.num_class = self._get_data(data_dir, mode)

    def _get_data(self, data_dir, mode):
        # define processors
        processors = {'CoLA': ColaProcessor,
                      'SST-2': Sst2Processor,
                      'MNLI': MnliProcessor,
                      'MRPC': MrpcProcessor,
                      'QNLI': QnliProcessor,
                      'QQP': QqpProcessor,
                      'RTE': RteProcessor,
                      'WNLI': WnliProcessor}

        # get InputExamples from raw file
        p = processors[self.task]()
        if mode == 'train':
            input_examples = p.get_train_examples(data_dir=os.path.join(data_dir, self.task))
        elif mode == 'dev':
            input_examples = p.get_dev_examples(data_dir=os.path.join(data_dir, self.task))
        else:
            raise Exception('mode must be in ["train", "dev"]...')

        # get InputFeatures from InputExamples
        input_features = glue_convert_examples_to_features(input_examples, tokenizer=self.tokenizer, \
                                                           max_length=self.max_len, task=self.task.lower())

        # convert InputFeatures to tensor
        input_ids, attention_mask, token_type_ids, labels = [], [], [], []
        for feature in input_features:
            input_ids.append(feature.input_ids)
            attention_mask.append(feature.attention_mask)
            token_type_ids.append(feature.token_type_ids)
            labels.append(feature.label)
        input_ids, attention_mask, token_type_ids, labels = map(lambda x: torch.LongTensor(x),
                                                                (input_ids, attention_mask, token_type_ids, labels))

        return (input_ids, attention_mask, token_type_ids, labels), len(p.get_labels())

    def __getitem__(self, item):
        out = ()
        for i in self.data:
            out += (i[item],)
        return out

    def __len__(self):
        return self.data[0].shape[0]

if __name__ == '__main__':
    dataset = GlueDataset('../glue_data', 'cola', 100, 'bert-base-uncased')
    print('For debug use')
