import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class TextUserEncoder(torch.nn.Module):
    def __init__(self, item_num, args):
        super(TextUserEncoder, self).__init__()
        self.args = args
        self.item_num = item_num
        self.inner_size = args.hidden_units

        self.user_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.user_encoder.requires_grad_(False)
        self.user_wfd_layer = nn.Sequential(
            nn.Linear(self.user_encoder.config.hidden_size, args.hidden_units),
            nn.GELU(),
            nn.Linear(args.hidden_units, args.hidden_units),
        )

        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, his_items, his_ratings, his_reviews, user_description):
        # B * L * H
        personalized_rep = self.user_encoder(**user_description).last_hidden_state[:,0,:]
        personalized_rep = self.user_wfd_layer(personalized_rep)
        return personalized_rep
