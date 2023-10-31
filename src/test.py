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


def pad_seq(self, seq, length, value):
    return [value] * (length-len(seq)) + seq

def process_data(conversation, response, system_prompt):
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
    input_ids = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
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
        turn_len = len(tokenizer(turn, verbose=False).input_ids) #there will be <s> at the beginning, so cur_len += turn_len can cover the artificially added </s>

        parts = turn.split(sep)
        parts[0] += sep
        # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
        instruction_len = len(tokenizer(parts[0], verbose=False).input_ids) - 2

        # Ignore the user instructions
        labels[: cur_len + instruction_len] = IGNORE_TOKEN_ID
        cur_len += turn_len

    print(list(zip(input_ids.tolist(), labels.tolist())))

    return dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        labels=labels
    )


tokenizer = AutoTokenizer.from_pretrained(
    "lmsys/vicuna-7b-v1.3",
    cache_dir="/home/v-weixu1/.cache",
    model_max_length=2048,
    use_fast=True,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.unk_token

data = {"user": 1217, "item": 26076, "all_conversations": [{"from": "human", "value": "These products have been bought by you."}, {"from": "gpt", "value": "OK, now I am this user."}, {"from": "human", "value": "A new product is now available for you to buy."}], "target_rating": 1, "target_response": "I like it."}

conv_data = process_data(data["all_conversations"], data["target_response"], SIMULATOR_SYSTEM_PROMPT)
