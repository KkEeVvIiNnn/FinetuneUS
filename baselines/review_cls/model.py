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
        self.inner_size = args.hidden_units * 3

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.rating_emb = torch.nn.Embedding(2, args.hidden_units)
        self.review_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.review_encoder.requires_grad_(False)
        self.review_wfd_layer = nn.Sequential(
            nn.Linear(self.review_encoder.config.hidden_size, self.inner_size),
            nn.GELU(),
            nn.Linear(self.inner_size, args.hidden_units),
        )
        self.target_wfd_layer = nn.Sequential(
            nn.Linear(args.hidden_units * 2, args.hidden_units),
            nn.GELU(),
            nn.Linear(args.hidden_units, args.hidden_units),
        )

        self.pos_emb = torch.nn.Embedding(args.max_his_len, self.inner_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.inner_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.inner_size,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.inner_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.inner_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.trans_layer = torch.nn.Linear(args.hidden_units*3, args.hidden_units)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, his_items, his_ratings, his_reviews):
        for k, v in his_reviews.items():
            his_reviews[k] = v.reshape(-1, self.args.max_review_len)
        review_reps = self.review_encoder(**his_reviews).last_hidden_state[:,0,:]
        review_reps = review_reps.reshape(-1, self.args.max_his_len, self.review_encoder.config.hidden_size)
        review_reps = self.review_wfd_layer(review_reps)

        seqs = torch.cat(
            [self.item_emb(his_items), self.rating_emb(his_ratings), review_reps],
            dim=-1
        )

        seqs *= (self.inner_size) ** 0.5
        positions = np.tile(np.array(range(his_items.shape[1])), [his_items.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (his_items == 0).type(torch.bool)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        seqs = self.trans_layer(seqs)
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats
        
    def forward(self, batch):
        his_items = batch["his_items"]
        his_ratings = batch["his_ratings"]
        his_reviews = batch["his_reviews"]
        # B * L * H
        personalized_rep = self.seq2feats(his_items, his_ratings, his_reviews)
        # personalized_rep_for_match = self.match_layer(personalized_rep)
        target_items = batch["target_items"]
        target_reviews = batch["target_reviews"]
        items_embs = self.item_emb(target_items)
        for k, v in target_reviews.items():
            target_reviews[k] = v.reshape(-1, self.args.max_review_len)
        review_reps = self.review_encoder(**target_reviews).last_hidden_state[:,0,:]
        review_reps = review_reps.reshape(-1, self.args.max_his_len, self.review_encoder.config.hidden_size)
        review_reps = self.review_wfd_layer(review_reps)

        target_reps = torch.cat([items_embs, review_reps], dim=-1)
        target_reps = self.target_wfd_layer(target_reps)

        masks = target_items.ne(0)
        logits = self.sigmoid((personalized_rep * target_reps).sum(dim=-1))
        loss = (torch.nn.MSELoss(reduction='none')(logits, batch["target_ratings"]) * masks).sum()
        return (loss) / torch.max(torch.tensor(1, device=masks.device), torch.sum(masks))

    def predict(self, batch):
        his_items = batch["his_items"]
        his_ratings = batch["his_ratings"]
        his_reviews = batch["his_reviews"]
        # B * L * H
        personalized_rep = self.seq2feats(his_items, his_ratings, his_reviews)[:,-1,:]
        # personalized_rep_for_match = self.match_layer(personalized_rep)
        
        target_item = batch["target_item"]
        target_review = batch["target_review"]
        items_embs = self.item_emb(target_item)
        review_reps = self.review_encoder(**target_review).last_hidden_state[:,0,:]
        review_reps = self.review_wfd_layer(review_reps)

        target_reps = torch.cat(
            [items_embs, review_reps],
            dim=-1
        )
        target_reps = self.target_wfd_layer(target_reps)

        logits = self.sigmoid((personalized_rep * target_reps).sum(dim=-1))
        return logits