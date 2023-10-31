import numpy as np
import torch
from torch.nn import functional as F

class MF(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MF, self).__init__()

        self.user_num = user_num
        self.item_num = item_num

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units, padding_idx=0)
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch):
        users = batch["user"]
        items = batch["item"]
        ratings = batch["rating"]
        user_embs = self.user_emb(users)
        item_embs = self.item_emb(items)
        logits = self.sigmoid((user_embs * item_embs).sum(dim=-1))
        return torch.nn.MSELoss()(logits, ratings)

    def point_predict(self, batch):
        users = batch["user"]
        user_embs = self.user_emb(users)
        target_items = batch["target_item"]
        item_embs = self.item_emb(target_items)
        logits = self.sigmoid((user_embs * item_embs).sum(dim=-1))
        return logits

    def list_predict(self, batch):
        users = batch["user"]
        user_embs = self.user_emb(users)
        # B * H
        candidate_items = batch["candidate_list"]
        # B * L * H
        item_embs = self.item_emb(candidate_items)
        logits = self.sigmoid(torch.einsum("bi,bji->bj", user_embs, item_embs))
        sorted_scores, indices = torch.sort(logits, -1, descending=True)
        return sorted_scores, indices
