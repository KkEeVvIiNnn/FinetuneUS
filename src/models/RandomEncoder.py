import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class RandomEncoder(torch.nn.Module):
    def __init__(self, args):
        super(RandomEncoder, self).__init__()
        self.args = args
        self.inner_size = args.hidden_units
    
    def forward(self, his_items, his_ratings, his_reviews, user_description):
        personalized_rep = torch.rand(his_items.size(0), self.inner_size, dtype=torch.float16, device=his_items.device)
        return personalized_rep
