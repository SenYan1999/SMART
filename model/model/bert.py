import torch
import torch.nn as nn

from transformers import BertModel

class Bert(nn.Module):
    def __init__(self, bert_type, num_class):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_type)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)

    def forward(self, input_ids, attention_mask, token_type):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)[1]
        out = self.classifier(out)

        logits = torch.nn.functional.log_softmax(out, dim=-1)
        return logits
