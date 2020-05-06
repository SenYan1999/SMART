import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, AlbertModel

class Bert(nn.Module):
    def __init__(self, bert_name, num_class, bert_type='bert', drop_out=0.1):
        super(Bert, self).__init__()
        if bert_type == 'bert':
            self.bert = BertModel.from_pretrained(bert_name)
        elif bert_type == 'albert':
            self.bert = AlbertModel.from_pretrained(bert_name)
        else:
            raise Exception('Please enter the correct bert type.')
        self.drop_out = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)

    def forward(self, input_ids, attention_mask, token_type):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)[1]
        out = self.drop_out(out)
        out = self.classifier(out)

        return out
