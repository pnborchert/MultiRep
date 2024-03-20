import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, desc_max_length, cat_entity_rep=False, multirep=False, desc=False): 
        nn.Module.__init__(self)
        self.max_length = max_length
        self.desc_max_length = desc_max_length
        self.cat_entity_rep = cat_entity_rep
        self.multirep = multirep
        self.desc = desc

        if self.multirep:
            self.bert = BertModel.from_pretrained(pretrain_path, output_hidden_states=True)
        else:
            self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):

        if self.multirep:
            if self.desc:
                x = self.bert(inputs['word'], attention_mask=inputs['mask'])
                x_desc = self.bert(inputs['desc_word'], attention_mask=inputs['desc_mask'])
                return x, x_desc
            else:
                x = self.bert(inputs['word'], attention_mask=inputs['mask'])
                return x
        
        else:

            if not self.cat_entity_rep:
                p = self.bert(inputs['word'], attention_mask=inputs['mask']).pooler_output
                return p

            else:
                h = self.bert(inputs['word'], attention_mask=inputs['mask']).last_hidden_state
                tensor_range = torch.arange(inputs['word'].size()[0])
                h_state = h[tensor_range, inputs["pos1"]]
                t_state = h[tensor_range, inputs["pos2"]]
                state = torch.cat((h_state, t_state), -1)
                return state
    
    def tokenize(self, raw_tokens, pos_head, pos_tail, template=None):
        
        # token -> index
        tokens = ['[CLS]']

        pos_mask_in_index = 1
        # template
        if self.multirep:
            for token in template:
                if token == '<e1>':
                    for t in pos_head:
                        tokens.append(raw_tokens[t])
                elif token == '<e2>':
                    for t in pos_tail:
                        tokens.append(raw_tokens[t])
                elif token == '<mask>':
                    tokens.append('[MASK]')
                    pos_mask_in_index = len(tokens)
                elif token == '<sep>':
                    tokens.append('[SEP]')
                    break
                else:
                    tokens.append(token)

        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        tokens.append('[SEP]')
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        if self.multirep:
            return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, pos_mask_in_index
        else:
            return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask
    
    def tokenize_desc(self, desc_list):
        """
        Tokenize relation descriptions. FewRel `pid2name.json` format contains relation type (short description) and relation description (long description).
        
        Template: "<cls> <mask> : <relation type>, <relation description> <sep>"
        """
        desc = []
        for d in desc_list:
            d = ", ".join(d)
            d = '[MASK] :' + d
            desc.append(d)
        
        enc = self.tokenizer(desc, add_special_tokens=True, max_length=self.desc_max_length, padding="max_length", truncation=True, return_tensors="pt")

        # MASK position tensor
        pos_mask = (enc["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        return enc["input_ids"], enc["attention_mask"], pos_mask

        