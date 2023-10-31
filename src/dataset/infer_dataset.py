import os
import logging
import random
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from dataset.conversation import get_conv_by_system_prompt

logger = logging.getLogger(__name__)
IGNORE_TOKEN_ID=-100

SIMULATOR_SYSTEM_PROMPT = (
    "You are a user simulator that accurately mimics the user's behavior and preferences based on given historical information. "
    "You will be given a user's history purchasing history at the first turn of the conversation. "
    "After that you should learn this user's preference and behavior based on the history. "
    "Then you are this user, and your response must mimic the tone and behavior of the user. "
    "All words need to be told from that user's perspective."
)
PERSONALIZED_SIMULATOR_SYSTEM_PROMPT = (
    "You are a user simulator that accurately mimics this user's behavior and preferences. "
    "You should learn this user's preference and behavior. "
    "Then you are this user, and your response must mimic the tone and behavior of the user. "
    "All words need to be told from that user's perspective."
)

class InferDataset(Dataset):
    def __init__(self, args, tokenizer, method, split='test'):
        self.method = method
        self.tokenizer = tokenizer
        self.args = args
        if self.method in ["GeneratePrompt", "GenerateAdaptor"]:
            self.review_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        data_names = args.data_names.split(',')
        self.dataset = []
        for data_name in data_names:
            if "rating" in data_name:
                self.rating_targets = []
                cnt = 0
                for line in open(os.path.join(args.data_dir, data_name+f'_{split}.jsonl'), "r"):
                    data = json.loads(line)
                    self.rating_targets.append(data["target_rating"])
                    new_data = data.copy()
                    new_data["target_response"] = "I don't like it."
                    self.dataset.append(new_data)
                    new_data = data.copy()
                    new_data["target_response"] = "I like it."
                    self.dataset.append(new_data)
                    cnt += 1
                    if cnt == 1000:
                        break
            if "list" in data_name:
                cnt = 0
                self.list_targets = []
                self.candidate_items = []
                for line in open(os.path.join(args.data_dir, data_name+f'_{split}.jsonl'), "r"):
                    data = json.loads(line)
                    self.list_targets.append(data["item"])
                    self.candidate_items.append(data["candidate_items"])
                    for title in data["candidate_item_titles"]:
                        new_data = data.copy()
                        new_data["target_response"] = title
                        self.dataset.append(new_data)
                    cnt += 1
                    if cnt == 100:
                        break
            if "review" in data_name:
                cnt = 0
                self.review_targets = []
                for line in open(os.path.join(args.data_dir, data_name+f'_{split}.jsonl'), "r"):
                    data = json.loads(line)
                    self.review_targets.append(data["target_response"])
                    self.dataset.append(data)
                    cnt += 1
                    if cnt == 1000:
                        break

        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        if self.args.prompt_type == "all":
            conv_data = self.process_data(self.dataset[item]["all_conversations"], self.dataset[item]["target_response"], SIMULATOR_SYSTEM_PROMPT)
        else:
            conv_data = self.process_data(self.dataset[item]["task_conversations"], self.dataset[item]["target_response"], PERSONALIZED_SIMULATOR_SYSTEM_PROMPT)

        data = {
            'input_ids': conv_data['input_ids'],
            'attention_mask': conv_data['attention_mask'],
            'labels': conv_data['labels']
        }

        if self.method in ["GeneratePrompt", "GenerateAdaptor"]:
            data = {
                **data,
                "his_items": torch.LongTensor(self.pad_seq(self.dataset[item]["his_items"], self.args.max_his_len, 0)),
                "his_ratings": torch.LongTensor(self.pad_seq(self.dataset[item]["his_ratings"], self.args.max_his_len, 0)),
                "his_reviews": self.process_review(self.pad_seq(self.dataset[item]["his_reviews"], self.args.max_his_len, "")),
                "user_description": self.process_description(self.dataset[item]["user_description"])
            }
        return data

    def pad_seq(self, seq, length, value):
        return [value] * (length-len(seq)) + seq

    def process_data(self, conversation, response, system_prompt):
        # Apply prompt templates
        conv = get_conv_by_system_prompt(system_prompt)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conv.messages = []
        for j, sentence in enumerate(conversation):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2] and len(sentence["value"]) > 0, f"i: {i}, j: {j}, role: {role}, conv.roles: {conv.roles}, sentence: {sentence}"
            conv.append_message(role, sentence["value"])
        conv.append_message(roles["gpt"], response)

        prompt = conv.get_prompt()
        # Tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            verbose=False,
        ).input_ids[0]
        labels = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "

        turns = prompt.split(conv.sep2)
        cur_len = 1
        labels[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(self.tokenizer(turn, verbose=False).input_ids) #there will be <s> at the beginning, so cur_len += turn_len can cover the artificially added </s>

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(self.tokenizer(parts[0], verbose=False).input_ids) - 2

            # Ignore the user instructions
            labels[: cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        labels[cur_len:] = IGNORE_TOKEN_ID
        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels
        )
    
    def process_review(self, reviews):
        inputs = self.review_tokenizer(
            reviews,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_review_len,
            return_tensors="pt",
            verbose=False,
        )
        for key in inputs:
            inputs[key] = inputs[key].unsqueeze(0)
        return inputs
    def process_description(self, description):
        return self.review_tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_user_len,
            return_tensors="pt",
            verbose=False,
        )
    
    def collator(self, features):
        max_length = max([f["input_ids"].size(0) for f in features])
        # max_length = self.tokenizer.model_max_length
        input_ids = torch.stack([F.pad(f["input_ids"], (0, max_length-f["input_ids"].size(0)), value=self.tokenizer.pad_token_id) for f in features])
        attention_mask = torch.stack([F.pad(f["attention_mask"], (0, max_length-f["attention_mask"].size(0))) for f in features])
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        if "labels" in features[0]:
            labels = torch.stack([F.pad(f["labels"], (0, max_length-f["labels"].size(0)), value=IGNORE_TOKEN_ID) for f in features]) 
            return_dict["labels"] = labels
        if "his_items" in features[0]:
            his_items = torch.stack([f["his_items"] for f in features])
            his_ratings = torch.stack([f["his_ratings"] for f in features])
            his_reviews = {}
            for key in features[0]["his_reviews"]:
                his_reviews[key] = torch.stack([f["his_reviews"][key] for f in features])
            user_description = {}
            for key in features[0]["user_description"]:
                user_description[key] = torch.cat([f["user_description"][key] for f in features], dim=0)
            return_dict = {
                **return_dict,
                "his_items": his_items,
                "his_ratings": his_ratings,
                "his_reviews": his_reviews,
                "user_description": user_description
            }
        return return_dict