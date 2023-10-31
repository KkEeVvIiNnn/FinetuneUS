import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.args = args
        self.item_num = item_num

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.user_encoder = BertModel.from_pretrained("bert-base-uncased")
        # self.user_encoder.requires_grad_(False)
        self.user_wfd_layer = nn.Sequential(
            nn.Linear(self.user_encoder.config.hidden_size, args.hidden_units),
            nn.GELU(),
            nn.Linear(args.hidden_units, args.hidden_units),
        )

        self.sigmoid = torch.nn.Sigmoid()

    def calc_personalized_rep(self, his_items, his_ratings, his_reviews):
        return self.seq2feats(his_items, his_ratings, his_reviews)[:, -1, :]
    
    def forward(self, batch):
        user_description = batch["user_description"]
        # B * L * H
        personalized_rep = self.user_encoder(**user_description).last_hidden_state[:,0,:]
        personalized_rep = self.user_wfd_layer(personalized_rep)

        target_item = batch["target_item"]
        target_rating = batch["target_rating"]
        target_item_emb = self.item_emb(target_item)
        logits = self.sigmoid((personalized_rep * target_item_emb).sum(dim=-1))
        loss = torch.nn.MSELoss()(logits, target_rating)
        return loss

    def point_predict(self, batch):
        user_description = batch["user_description"]
        # B * L * H
        personalized_rep = self.user_encoder(**user_description).last_hidden_state[:,0,:]
        personalized_rep = self.user_wfd_layer(personalized_rep)
        # B * H
        # personalized_rep_for_match = self.match_layer(personalized_rep)
        target_item = batch["target_item"]
        # B * H
        item_emb = self.item_emb(target_item)
        logits = self.sigmoid((personalized_rep * item_emb).sum(dim=-1))
        return logits

    def list_predict(self, batch):
        user_description = batch["user_description"]
        # B * L * H
        personalized_rep = self.user_encoder(**user_description).last_hidden_state[:,0,:]
        personalized_rep = self.user_wfd_layer(personalized_rep)
        # B * H
        # personalized_rep_for_match = self.match_layer(personalized_rep)
        candidate_items = batch["candidate_list"]
        # B * L * H
        item_embs = self.item_emb(candidate_items)
        logits = self.sigmoid(torch.einsum("bi,bji->bj", personalized_rep, item_embs))
        sorted_scores, indices = torch.sort(logits, -1, descending=True)
        return sorted_scores, indices
