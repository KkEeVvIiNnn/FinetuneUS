import os
import re
import json
import gzip
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from collections import defaultdict
import importlib
import pickle
from torch.utils.data import Dataset

import torch

NEG_NUM=9
random.seed(43)

prompt_templates = {
    "setup_prompt_begin_end": [
        {
            "begin": "You have bought the following products:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\nNow, let's start the simulation."
        },
        {
            "begin": "The items in your purchase history are:\n[start of purchase history]\n",
            "end": "[end of purchase history]\n\nNow, let's start the simulation."
        },
        {
            "begin": "These are the products you've purchased:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\nNow, we can start simulating."
        },
        {
            "begin": "You have bought the following products:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\nLet's move on to the simulation phase."
        },
        {
            "begin": "These products have been bought by you:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\nLet us proceed with the simulation."
        },
    ],
    "setup_prompt_content": [
        "title: \"{title}\", category: {domain}, your rating: {rating}, your review: {review}\n",
        "product title: \"{title}\", product category: {domain}, your score: {rating}, your review: {review}\n",
        "product name: \"{title}\", product type: {domain}, your rating score: {rating}, your review: {review}\n",
        "title of item: \"{title}\", category of item: {domain}, your given rating: {rating}, your review: {review}\n",
        "name of product: \"{title}\", type of product: {domain}, your rating: {rating}, your review: {review}\n",
    ],
    "task_setup_prompt_begin_end": [
        {
            "begin": "You have bought the following products:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\n"
        },
        {
            "begin": "The items in your purchase history are:\n[start of purchase history]\n",
            "end": "[end of purchase history]\n\n"
        },
        {
            "begin": "These are the products you've purchased:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\n"
        },
        {
            "begin": "You have bought the following products:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\n"
        },
        {
            "begin": "These products have been bought by you:\n[start of historical purchased products]\n",
            "end": "[end of historical purchased products]\n\n"
        },
    ],
    "task_setup_prompt_content": [
        "title: \"{title}\", category: {domain}, your rating: {rating}.\n",
        "product title: \"{title}\", product category: {domain}, your score: {rating}.\n",
        "product name: \"{title}\", product type: {domain}, your rating score: {rating}.\n",
        "title of item: \"{title}\", category of item: {domain}, your given rating: {rating}.\n",
        "name of product: \"{title}\", type of product: {domain}, your rating: {rating}.\n",
    ],
    "confirm_prompt": [
        "OK, now I am this user. I will simulate this user's preferences and talk to you.",
        "Alright, now I am taking on the role of this user. I'll imitate this user's preferences and communicate with you.",
        "Okay, I am currently representing this user. I will emulate this user's preferences and converse with you.",
        "Sure, I am now adopting the persona of this user. I will model this user's preferences and have a dialogue with you.",
        "Okay, I have now transformed into this user. I will reproduce this user's preferences and have a conversation with you."
    ],
    "rating_query_prompt": [
        "Now there is a new product you can purchase: it's title is \"{title}\", and it's category is {domain}.\nDo you like it?",
        "A new product is now available for you to buy: its name is \"{title}\", and it is under the {domain} category.\nWould you like it?",
        "Introducing a new product, \"{title}\", belonging to the {domain} category.\nOnce you've bought it, whether you like it or not?",
        "A new item, \"{title}\", is now available for purchase in the {domain} category.\nWould you like it once you own it?"
    ],
    "list_query_prompt_begin_end": [
        {
            "begin": "Now there are some new products you can purchase:\n[start of candidate products]\n",
            "end": "[end of candidate products]\n\nPlease select the product that you are most interested in to purchase.\nOutput product title only."
        }
    ],
    "list_query_prompt_content": [
        "title: \"{title}\", category: {domain}.\n",
        "product title: \"{title}\", product category: {domain}.\n",
        "product name: \"{title}\", product type: {domain}.\n",
        "title of item: \"{title}\", category of item: {domain}.\n",
        "name of product: \"{title}\", type of product: {domain}.\n",
    ],
    "review_query_prompt": [
        "Now there is a new product you can purchase: it's title is \"{title}\", and it's category is {domain}.\nPlease write a short review to tell {rating} after purchasing it according to your preference.",
        "A new product is now available for you to buy: its name is \"{title}\", and it is under the {domain} category.\nPlease compose a brief review post-purchase to tell {rating}, taking into account your personal preferences.",
        "Introducing a new product, \"{title}\", belonging to the {domain} category.\nPlease create a short review to tell {rating} after buying it.",
        "With a new product called \"{title}\" in the {domain} category now up for purchase.\nPlease author a concise review  to tell {rating} in line with your taste.",
        "A new item, \"{title}\", is now available for purchase in the {domain} category.\nPlease provide a brief review to tell {rating} that takes into consideration your preferences."
    ],
}

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,"r") as fd:
        for line in fd:
            lines.append(line.rstrip("\n"))
    return lines

def parse(path):
    if path.endswith("gz"):
        g = gzip.open(path, "r")
    else:
        g = open(path, "r")
    nan_default = {"NaN": "", "false": "", "true": ""}
    for l in g:
        yield eval(l, nan_default)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rating,list,review", help="tasks for data generation.")
    parser.add_argument("--dataset", type=str, default="Books-Movies_and_TV")
    parser.add_argument("--num_train_user", type=int, default=10000)
    parser.add_argument("--num_test_user", type=int, default=1000)
    return parser.parse_args()

def gen_setup_prompt(his_items, his_ratings, his_reviews):
    setup_prompt_begin_end_template = random.choice(prompt_templates["setup_prompt_begin_end"])
    setup_prompt = setup_prompt_begin_end_template["begin"]
    setup_prompt_content_template = random.choice(prompt_templates["setup_prompt_content"])
    for his_item, his_rating, his_review in zip(his_items, his_ratings, his_reviews):
        meta = item2meta[his_item]
        domain = meta["domain"]
        title = meta['title']
        setup_prompt += setup_prompt_content_template.format(
            domain=domain,
            title=title,
            rating="like" if his_rating else "dislike",
            review=his_review,
        )
    setup_prompt += setup_prompt_begin_end_template["end"]
    confirm_prompt = random.choice(prompt_templates["confirm_prompt"])

    task_setup_prompt_begin_end_template = random.choice(prompt_templates["task_setup_prompt_begin_end"])
    task_setup_prompt = task_setup_prompt_begin_end_template["begin"]
    task_setup_prompt_content_template = random.choice(prompt_templates["task_setup_prompt_content"])
    for his_item, his_rating in zip(his_items, his_ratings):
        meta = item2meta[his_item]
        domain = meta["domain"]
        title = meta['title']
        task_setup_prompt += task_setup_prompt_content_template.format(
            domain=domain,
            title=title,
            rating="like" if his_rating else "dislike",
        )
    task_setup_prompt += task_setup_prompt_begin_end_template["end"]
    return setup_prompt, confirm_prompt, task_setup_prompt

def gen_rating_data(user, his_items, his_ratings, his_reviews, item2meta, target_item, target_rating):
    target_itemmeta = item2meta[target_item]
    rating2result = ["I don't like it.", "I like it."]
    result = rating2result[target_rating]
    data = {
        "user": user,
        "item": target_item,
        "all_conversations": [],
        "task_conversations": [],
        "user_description": "",
        "his_items": his_items,
        "his_ratings": his_ratings,
        "his_reviews": his_reviews,
        "target_rating": target_rating,
        "target_response": result
    }
    setup_prompt, confirm_prompt, task_setup_prompt = gen_setup_prompt(his_items, his_ratings, his_reviews)
    
    query_prompt = random.choice(prompt_templates["rating_query_prompt"]).format(
        domain=target_itemmeta["domain"],
        title=target_itemmeta["title"],
    )
    data["user_description"] = setup_prompt
    data["all_conversations"] = [
        {"from": "human","value": setup_prompt},
        {"from": "gpt","value": confirm_prompt},
        {"from": "human","value": query_prompt},
    ]
    data["task_conversations"] = [
        {"from": "human","value": task_setup_prompt+query_prompt},
    ]
    return data

def gen_review_data(user, his_items, his_ratings, his_reviews, item2meta, target_item, target_rating, target_review):
    target_itemmeta = item2meta[target_item]
    data = {
        "user": user,
        "item": target_item,
        "all_conversations": [],
        "task_conversations": [],
        "user_description": "",
        "his_items": his_items,
        "his_ratings": his_ratings,
        "his_reviews": his_reviews,
        "target_rating": target_rating,
        "target_response": target_review,
    }
    setup_prompt, confirm_prompt, task_setup_prompt = gen_setup_prompt(his_items, his_ratings, his_reviews)

    query_prompt = random.choice(prompt_templates["review_query_prompt"]).format(
        domain=target_itemmeta["domain"],
        title=target_itemmeta["title"],
        rating="you like it" if target_rating else "you don't like it"
    )
    data["user_description"] = setup_prompt
    data["all_conversations"] = [
        {"from": "human","value": setup_prompt},
        {"from": "gpt","value": confirm_prompt},
        {"from": "human","value": query_prompt}
    ]
    data["task_conversations"] = [
        {"from": "human","value": query_prompt}
    ]
    return data

def gen_list_data(user, his_items, his_ratings, his_reviews, item2meta, target_item, candidate_items):
    data = {
        "user": user,
        "item": target_item,
        "all_conversations": [],
        "task_conversations": [],
        "user_description": "",
        "his_items": his_items,
        "his_ratings": his_ratings,
        "his_reviews": his_reviews,
        "candidate_items": candidate_items,
        "candidate_item_titles": [item2meta[item]["title"] for item in candidate_items],
        "target_response": item2meta[target_item]["title"]
    }
    setup_prompt, confirm_prompt, task_setup_prompt = gen_setup_prompt(his_items, his_ratings, his_reviews)

    query_prompt_begin_end_template = random.choice(prompt_templates["list_query_prompt_begin_end"])
    query_prompt = query_prompt_begin_end_template["begin"]
    query_prompt_content_template = random.choice(prompt_templates["list_query_prompt_content"])
    for candidate_item in candidate_items:
        meta = item2meta[candidate_item]
        domain = meta["domain"]
        title = meta['title']
        query_prompt += query_prompt_content_template.format(
            domain=domain,
            title=title
        )
    query_prompt += query_prompt_begin_end_template["end"]
    data["all_conversations"] = [
        {"from": "human","value": setup_prompt},
        {"from": "gpt","value": confirm_prompt},
        {"from": "human","value": query_prompt},
    ]
    data["task_conversations"] = [
        {"from": "human","value": task_setup_prompt + query_prompt},
    ]
    return data

def sample(pool, neg, num):
    neg = set(neg)
    if isinstance(pool, list):
        ret = []
        while len(ret) < num:
            ret.extend(random.sample(pool, num-len(ret)))
            ret = list(set(ret)-neg)
        return ret
    elif isinstance(pool, int):
        ret = []
        while len(ret) < num:
            ret.extend(random.sample(range(1, pool+1), num-len(ret)))
            ret = list(set(ret)-neg)
        return ret

class TrainDataset(Dataset):
    def __init__(self, data, negpool):
        super(TrainDataset, self).__init__()
        self.data = data
        self.negpool = negpool

    def __len__(self):
        return len(self.data)
    
    def get_rating_data(self):
        ret = []
        for data in self.data:
            ret.append(gen_rating_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, data["target_item"], data["target_rating"]))
            if data["target_rating"] == 1:
                ret.append(gen_rating_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, data["target_item"], data["target_rating"]))
                neg_item = sample(self.negpool, data["all_items"], 1)[0]
                ret.append(gen_rating_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, neg_item, 0))
        return ret
    def get_list_data(self):
        ret = []
        for data in self.data:
            if data["target_rating"] == 1:
                candidate_items = data["candidate_list"]
                random.shuffle(candidate_items)
                ret.append(gen_list_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, data["target_item"], candidate_items))
        return ret
    def get_review_data(self):
        ret = []
        for data in self.data:
            if "target_review" in data:
                ret.append(gen_review_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, data["target_item"], data["target_rating"], data["target_review"]))
        return ret
    
class TestDataset(Dataset):
    def __init__(self, data):
        super(TestDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    def get_rating_data(self):
        ret = []
        num_test_user = 0
        for data in self.data:
            ret.append(gen_rating_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, data["target_item"], data["target_rating"]))
            num_test_user += 1
            if num_test_user == args.num_test_user:
                return ret
        return ret
    def get_list_data(self):
        ret = []
        num_test_user = 0
        for data in self.data:
            if data["target_rating"] == 1:
                candidate_items = data["candidate_list"]
                random.shuffle(candidate_items)
                ret.append(gen_list_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, data["target_item"], candidate_items))
                num_test_user += 1
                if num_test_user == args.num_test_user:
                    return ret
        return ret
    def get_review_data(self):
        ret = []
        num_test_user = 0
        for data in self.data:
            if "target_review" in data:
                ret.append(gen_review_data(data["user"], data["his_items"], data["his_ratings"], data["his_reviews"], item2meta, data["target_item"], data["target_rating"], data["target_review"]))
                num_test_user += 1
                if num_test_user == args.num_test_user:
                    return ret
        return ret

import nltk  
import re
  
def retain_previous_sentences(text):
    text = ' '.join([x.strip() for x in text.split('\n')])
    if len(text) and text[-1] not in ['.', '!', '?']:
        text += '.'
    sentences = re.split('([.!?])', text)
    sentences = [sentences[i]+sentences[i+1] for i in range(0, len(sentences)-1, 2)]
    current_sentence = ""
    token_num = 0
    for sentence in sentences:
        if len(sentence):
            token_num += len(nltk.word_tokenize(sentence))
            if token_num <= 50:
                current_sentence = current_sentence+sentence
            else:
                break
    return current_sentence

def build_his_data(user, item_ratings, review_dict):
    his_items = []
    his_ratings = []
    his_reviews = []
    for his_item, his_rating in item_ratings[-20:]:
        if (user, his_item) in review_dict:
            his_review = review_dict[(user, his_item)]
        else:
            his_review = "No review."
        his_items.append(his_item)
        his_ratings.append(his_rating)
        his_reviews.append(his_review)
    return his_items, his_ratings, his_reviews
    
def prepare_training_data(user, item_ratings, negpool, review_dict):
    nfeedback = len(item_ratings)
    all_items = [item for item, rating in item_ratings]
    ret = []
    if nfeedback < 3:
        last_idx = nfeedback
    else:
        last_idx = nfeedback - 2
    for idx in range(1, last_idx):
        target_item, target_rating = item_ratings[idx]
        his_items, his_ratings, his_reviews = build_his_data(user, item_ratings[:idx], review_dict)
        data = {
            "user": user,
            "his_items": his_items,
            "his_ratings": his_ratings,
            "his_reviews": his_reviews,
            "target_item": target_item,
            "target_rating": target_rating,
        }
        if target_rating == 1:
            data["all_items"] = all_items         
            data["candidate_list"] = [target_item] + sample(negpool, all_items, NEG_NUM)
            random.shuffle(data["candidate_list"])
        if (user, target_item) in review_dict:
            data["target_review"] = review_dict[(user, target_item)]
        ret.append(data)
    return ret

def prepare_validate_data(user, item_ratings, negpool, review_dict):
    nfeedback = len(item_ratings)
    all_items = [item for item, rating in item_ratings]
    ret = []
    if nfeedback >= 3:
        target_item, target_rating = item_ratings[-2]
        his_items, his_ratings, his_reviews = build_his_data(user, item_ratings[:-2], review_dict)
        data = {
            "user": user,
            "his_items": his_items,
            "his_ratings": his_ratings,
            "his_reviews": his_reviews,
            "target_item": target_item,
            "target_rating": target_rating,
        }
        if target_rating == 1:            
            data["candidate_list"] = [target_item] + sample(negpool, all_items, NEG_NUM)
            random.shuffle(data["candidate_list"])
        if (user, target_item) in review_dict:
            data["target_review"] = review_dict[(user, target_item)]
        ret.append(data)
    return ret

def prepare_testing_data(user, item_ratings, user2negsamples, review_dict):
    nfeedback = len(item_ratings)
    ret = []
    if nfeedback >= 3:
        target_item, target_rating = item_ratings[-1]
        his_items, his_ratings, his_reviews = build_his_data(user, item_ratings[:-1], review_dict)
        data = {
            "user": user,
            "his_items": his_items,
            "his_ratings": his_ratings,
            "his_reviews": his_reviews,
            "target_item": target_item,
            "target_rating": target_rating,
        }
        if target_rating == 1:            
            data["candidate_list"] = [target_item] + user2negsamples[user][:NEG_NUM]
            random.shuffle(data["candidate_list"])
        if (user, target_item) in review_dict:
            data["target_review"] = review_dict[(user, target_item)]
        ret.append(data)
    return ret

def prepare_indomain_coldstart_test(user, item_ratings, user2negsamples, review_dict):
    nfeedback = len(item_ratings)
    ret = []
    if nfeedback >= 3:
        target_item, target_rating = item_ratings[-1]
        his_items, his_ratings, his_reviews = build_his_data(user, item_ratings[:-1], review_dict)
        data = {
            "user": user,
            "his_items": his_items,
            "his_ratings": his_ratings,
            "his_reviews": his_reviews,
            "target_item": target_item,
            "target_rating": target_rating,
        }
        if target_rating == 1:            
            data["candidate_list"] = [target_item] + user2negsamples[user][:NEG_NUM]
            random.shuffle(data["candidate_list"])
        if (user, target_item) in review_dict:
            data["target_review"] = review_dict[(user, target_item)]
        ret.append(data)
    return ret

def prepare_crossdomain_coldstart_test(user, item_ratings, user2negsamples, review_dict, target_item_rating):
    ret = []
    target_item, target_rating = target_item_rating
    his_items, his_ratings, his_reviews = build_his_data(user, item_ratings, review_dict)
    data = {
        "user": user,
        "his_items": his_items,
        "his_ratings": his_ratings,
        "his_reviews": his_reviews,
        "target_item": target_item,
        "target_rating": target_rating,
    }
    if target_rating == 1:            
        data["candidate_list"] = [target_item] + user2negsamples[user][:NEG_NUM]
        random.shuffle(data["candidate_list"])
    if (user, target_item) in review_dict:
        data["target_review"] = review_dict[(user, target_item)]
    ret.append(data)
    return ret

if __name__ == "__main__":
    args = parse_args()
    '''
    Set seeds
    '''
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)

    sequential_data = f"../data/{args.dataset}/sequential_data.txt"
    metadata = f"../data/{args.dataset}/metadata.jsonl"
    from_dataset, to_dataset = args.dataset.split('-')
    item2meta = ["padding"]
    for line in open(metadata):
        meta = json.loads(line)
        meta = {
            "title": meta["title"],
            "domain": meta["domain"]
        }
        item2meta.append(meta)
    if os.path.exists(f"../data/{args.dataset}/review.pkl"):
        review_dict = pickle.load(open(f"../data/{args.dataset}/review.pkl", "rb"))
    else:
        review_data = f"../data/{args.dataset}/review.jsonl"
        review_dict = {}
        for idx, line in enumerate(tqdm(open(review_data))):
            data = json.loads(line)
            review = retain_previous_sentences(data["review"])
            if len(review):
                review_dict[(int(data["user"]), int(data["item"]))] = review
        pickle.dump(review_dict, open(f"../data/{args.dataset}/review.pkl", 'wb'))
    
    all_indomain_items = [item for item in range(1, len(item2meta)) if item2meta[item]["domain"]==from_dataset]
    all_crossdomain_items = [item for item in range(1, len(item2meta)) if item2meta[item]["domain"]==to_dataset]
    
    indomain_negative_samples_filepath = f'../data/{args.dataset}/indomain_negative_samples.txt'
    if not os.path.exists(indomain_negative_samples_filepath):
        print("constructing indomain_negative_samples")
        outf = open(indomain_negative_samples_filepath, 'w')
        for line in ReadLineFromFile(sequential_data):
            user, item_ratings = line.split('\t', 1)
            user, user_type = user.split(',')
            user = int(user)
            user_type = int(user_type)
            item_ratings = [x.split(',') for x in item_ratings.split('\t')]
            items = [int(item) for item, rating in item_ratings]
            negative_sample = sample(all_indomain_items, items, 99)
            outf.write('\t'.join([f"{user}"]+[str(x) for x in negative_sample])+'\n')
        outf.close()
    all_indomain_negative_samples = open(indomain_negative_samples_filepath, 'r').readlines()
    user2indomain_negative_samples = {}
    for line in all_indomain_negative_samples:
        user, indomain_negative_samples = line.split('\t', 1)
        user = int(user)
        indomain_negative_samples = [int(item) for item in indomain_negative_samples.split('\t')]
        user2indomain_negative_samples[user] = indomain_negative_samples
    
    crossdomain_negative_samples_filepath = f'../data/{args.dataset}/crossdomain_negative_samples.txt'
    if not os.path.exists(crossdomain_negative_samples_filepath):
        print("constructing crossdomain_negative_samples")
        outf = open(crossdomain_negative_samples_filepath, 'w')
        for line in ReadLineFromFile(sequential_data):
            user, item_ratings = line.split('\t', 1)
            user, user_type = user.split(',')
            user = int(user)
            user_type = int(user_type)
            item_ratings = [x.split(',') for x in item_ratings.split('\t')]
            items = [int(item) for item, rating in item_ratings]
            negative_sample = sample(all_crossdomain_items, items, 99)
            outf.write('\t'.join([f"{user}"]+[str(x) for x in negative_sample])+'\n')
        outf.close()
    all_crossdomain_negative_samples = open(crossdomain_negative_samples_filepath, 'r').readlines()
    user2crossdomain_negative_samples = {}
    for line in all_crossdomain_negative_samples:
        user, crossdomain_negative_samples = line.split('\t', 1)
        user = int(user)
        crossdomain_negative_samples = [int(item) for item in crossdomain_negative_samples.split('\t')]
        user2crossdomain_negative_samples[user] = crossdomain_negative_samples

    # generate data
    data_train = []
    data_valid = []
    data_test = []
    data_indomain_coldstart_test = []
    data_crossdomain_coldstart_test = []
    num_train_user = 0
    num_test_user = 0
    num_indomain_coldstart_test_user = 0
    num_crossdomain_coldstart_test_user = 0

    def check_test_complete(test_data):
        num_sample = 0
        for data in test_data:
            if "candidate_list" in data:
                num_sample += 1
        if num_sample >= 1000:
            return True
        return False

    for line in tqdm(ReadLineFromFile(sequential_data)):
        user, item_ratings = line.split('\t', 1)
        user, user_type = user.split(',')
        user = int(user)
        user_type = int(user_type)
        item_ratings = [x.split(',') for x in item_ratings.split('\t')]
        item_ratings = [[int(item), 1 if float(rating) > 3 else 0] for item, rating in item_ratings]
        from_domain_item_ratings = [[item, rating] for item, rating in item_ratings if item2meta[item]["domain"]==from_dataset]
        cross_domain_target_item_rating = None
        for item, rating in item_ratings:
            if item2meta[item]["domain"] == from_dataset:
                cross_domain_target_item_rating = None
            elif cross_domain_target_item_rating == None:
                cross_domain_target_item_rating = item, rating
        if len(from_domain_item_ratings):
            if not check_test_complete(data_indomain_coldstart_test):
                data_indomain_coldstart_test.extend(prepare_indomain_coldstart_test(user, from_domain_item_ratings, user2indomain_negative_samples, review_dict))
            else:
                if num_train_user < args.num_train_user:
                    num_train_user += 1
                    data_train.extend(prepare_training_data(user, from_domain_item_ratings, all_indomain_items, review_dict))
                    data_valid.extend(prepare_validate_data(user, from_domain_item_ratings, all_indomain_items, review_dict))
                    data_test.extend(prepare_testing_data(user, from_domain_item_ratings, user2indomain_negative_samples, review_dict))
                if user_type == 2:
                    if cross_domain_target_item_rating is not None:
                        data_crossdomain_coldstart_test.extend(prepare_crossdomain_coldstart_test(user, from_domain_item_ratings, user2crossdomain_negative_samples, review_dict, cross_domain_target_item_rating))

    train_dataset = TrainDataset(data_train, all_indomain_items)
    valid_dataset = TestDataset(data_valid)
    test_dataset = TestDataset(data_test)
    indoamin_coldstart_test_dataset = TestDataset(data_indomain_coldstart_test)
    crossdoamin_coldstart_test_dataset = TestDataset(data_crossdomain_coldstart_test)

    if "rating" in args.task:
        train_data_path = f"../data/{from_dataset}_rating_train.jsonl"
        valid_data_path = f"../data/{from_dataset}_rating_valid.jsonl"
        test_data_path = f"../data/{from_dataset}_rating_test.jsonl"
        indomain_coldstart_test_data_path = f"../data/{from_dataset}_rating_indomain_coldstart_test.jsonl"
        crossdomain_coldstart_test_data_path = f"../data/{from_dataset}_rating_crossdomain_coldstart_test.jsonl"
        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
        train_fd = open(train_data_path, "w", encoding="utf-8")
        valid_fd = open(valid_data_path, "w", encoding="utf-8")
        test_fd = open(test_data_path, "w", encoding="utf-8")
        indomain_coldstart_test_fd = open(indomain_coldstart_test_data_path, "w", encoding="utf-8")
        crossdomain_coldstart_test_fd = open(crossdomain_coldstart_test_data_path, "w", encoding="utf-8")
        for data in train_dataset.get_rating_data():
            train_fd.write(json.dumps(data)+'\n')
        for data in valid_dataset.get_rating_data():
            valid_fd.write(json.dumps(data)+'\n')
        for data in test_dataset.get_rating_data():
            test_fd.write(json.dumps(data)+'\n')
        for data in indoamin_coldstart_test_dataset.get_rating_data():
            indomain_coldstart_test_fd.write(json.dumps(data)+'\n')
        for data in crossdoamin_coldstart_test_dataset.get_rating_data():
            crossdomain_coldstart_test_fd.write(json.dumps(data)+'\n')

    if "list" in args.task:
        train_data_path = f"../data/{from_dataset}_list_train.jsonl"
        valid_data_path = f"../data/{from_dataset}_list_valid.jsonl"
        test_data_path = f"../data/{from_dataset}_list_test.jsonl"
        indomain_coldstart_test_data_path = f"../data/{from_dataset}_list_indomain_coldstart_test.jsonl"
        crossdomain_coldstart_test_data_path = f"../data/{from_dataset}_list_crossdomain_coldstart_test.jsonl"
        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
        train_fd = open(train_data_path, "w", encoding="utf-8")
        valid_fd = open(valid_data_path, "w", encoding="utf-8")
        test_fd = open(test_data_path, "w", encoding="utf-8")
        indomain_coldstart_test_fd = open(indomain_coldstart_test_data_path, "w", encoding="utf-8")
        crossdomain_coldstart_test_fd = open(crossdomain_coldstart_test_data_path, "w", encoding="utf-8")
        for data in train_dataset.get_list_data():
            train_fd.write(json.dumps(data)+'\n')
        for data in valid_dataset.get_list_data():
            valid_fd.write(json.dumps(data)+'\n')
        for data in test_dataset.get_list_data():
            test_fd.write(json.dumps(data)+'\n')
        for data in indoamin_coldstart_test_dataset.get_list_data():
            indomain_coldstart_test_fd.write(json.dumps(data)+'\n')
        for data in crossdoamin_coldstart_test_dataset.get_list_data():
            crossdomain_coldstart_test_fd.write(json.dumps(data)+'\n')

    if "review" in args.task:
        train_data_path = f"../data/{from_dataset}_review_train.jsonl"
        valid_data_path = f"../data/{from_dataset}_review_valid.jsonl"
        test_data_path = f"../data/{from_dataset}_review_test.jsonl"
        indomain_coldstart_test_data_path = f"../data/{from_dataset}_review_indomain_coldstart_test.jsonl"
        crossdomain_coldstart_test_data_path = f"../data/{from_dataset}_review_crossdomain_coldstart_test.jsonl"
        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
        train_fd = open(train_data_path, "w", encoding="utf-8")
        valid_fd = open(valid_data_path, "w", encoding="utf-8")
        test_fd = open(test_data_path, "w", encoding="utf-8")
        indomain_coldstart_test_fd = open(indomain_coldstart_test_data_path, "w", encoding="utf-8")
        crossdomain_coldstart_test_fd = open(crossdomain_coldstart_test_data_path, "w", encoding="utf-8")
        for data in train_dataset.get_review_data():
            train_fd.write(json.dumps(data)+'\n')
        for data in valid_dataset.get_review_data():
            valid_fd.write(json.dumps(data)+'\n')
        for data in test_dataset.get_review_data():
            test_fd.write(json.dumps(data)+'\n')
        for data in indoamin_coldstart_test_dataset.get_review_data():
            indomain_coldstart_test_fd.write(json.dumps(data)+'\n')
        for data in crossdoamin_coldstart_test_dataset.get_review_data():
            crossdomain_coldstart_test_fd.write(json.dumps(data)+'\n')