import os
import sys
import json
import random
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, set_seed

from utils import *
from model import SASRec

NEG_NUM=9

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--threshold', default=3, type=int)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--max_his_len', default=20, type=int)
parser.add_argument('--max_review_len', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--do_test', action='store_true')
parser.add_argument('--load_state_dict', default="", type=str)
parser.add_argument('--test_review_path', type=str)
parser.add_argument("--save_dir",
                    default="./outputs/",
                    type=str,
                    help="The path to save model.")

args = parser.parse_args()
args.save_dir =  os.path.join(args.save_dir, args.dataset, f"l={args.num_blocks}_h={args.num_heads}_d={args.hidden_units}_lr={args.lr}")
args.log_path =  os.path.join(args.save_dir, "run.log")
args.save_path = os.path.join(args.save_dir, "model.bin")
os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
logging.basicConfig(filename=args.log_path, level=logging.DEBUG)  # debug level - avoid printing log information during training
logger = logging.getLogger(__name__)
logger.info(args)

set_seed(123)
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

review_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def process_review(reviews):
    # return {"a": torch.LongTensor([1])}
    return review_tokenizer(
        reviews,
        padding="max_length",
        truncation=True,
        max_length=args.max_review_len,
        return_tensors="pt",
        verbose=False,
    )

class SASRecTrainingDataset(Dataset):
    def __init__(self, data):
        super(SASRecTrainingDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        batch = {
            "his_items": torch.LongTensor(pad_seq(data["his_items"], args.max_his_len, 0)),
            "his_ratings": torch.LongTensor(pad_seq(data["his_ratings"], args.max_his_len, 0)),
            "his_reviews": process_review(pad_seq(data["his_reviews"], args.max_his_len, "")),
            "target_items": torch.LongTensor(pad_seq(data["target_items"], args.max_his_len, 0)),
            "target_ratings": torch.FloatTensor(pad_seq(data["target_ratings"], args.max_his_len, 0)),
            "target_reviews": process_review(pad_seq(data["target_reviews"], args.max_his_len, "")),
        }
        return batch
    def collate_fn(self, batch):
        B = len(batch)
        batch_entry = {
            "his_items": torch.stack([data["his_items"] for data in batch]),
            "his_ratings": torch.stack([data["his_ratings"] for data in batch]),
            "his_reviews": {},
            "target_items": torch.stack([data["target_items"] for data in batch]),
            "target_ratings": torch.stack([data["target_ratings"] for data in batch]),
            "target_reviews": {},
        }
        for key in batch[0]["his_reviews"]:
            batch_entry["his_reviews"][key] = torch.stack([data["his_reviews"][key] for data in batch])
            batch_entry["target_reviews"][key] = torch.stack([data["target_reviews"][key] for data in batch])
        return batch_entry

class SASRecPointTestingDataset(Dataset):
    def __init__(self, data, split):
        super(SASRecPointTestingDataset, self).__init__()
        self.data = data
        if split == "test":
            self.data = self.data[:1000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        batch = {
            "his_items": torch.LongTensor(pad_seq(data["his_items"], args.max_his_len, 0)),
            "his_ratings": torch.LongTensor(pad_seq(data["his_ratings"], args.max_his_len, 0)),
            "his_reviews": process_review(pad_seq(data["his_reviews"], args.max_his_len, "")),
            "target_item": torch.tensor(data["target_item"], dtype=torch.long),
            "target_rating": torch.tensor(data["target_rating"], dtype=torch.long),
            "target_review": process_review(data["target_review"]),
        }
        return batch
    def collate_fn(self, batch):
        B = len(batch)
        batch_entry = {
            "his_items": torch.stack([data["his_items"] for data in batch]),
            "his_ratings": torch.stack([data["his_ratings"] for data in batch]),
            "his_reviews": {},
            "target_item": torch.stack([data["target_item"] for data in batch]),
            "target_rating": torch.stack([data["target_rating"] for data in batch]),
            "target_review": {},
        }
        for key in batch[0]["his_reviews"]:
            batch_entry["his_reviews"][key] = torch.stack([data["his_reviews"][key] for data in batch])
            batch_entry["target_review"][key] = torch.stack([data["target_review"][key].squeeze(0) for data in batch])
        return batch_entry

def build_his_data(user, item_ratings, review_dict):
    his_items = []
    his_ratings = []
    his_reviews = []
    for his_item, his_rating in item_ratings:
        if (user, his_item) in review_dict:
            his_review = review_dict[(user, his_item)]
        else:
            his_review = "No review."
        his_items.append(his_item)
        his_ratings.append(his_rating)
        his_reviews.append(his_review)
    return his_items, his_ratings, his_reviews
    
def prepare_training_data(user, item_ratings, review_dict):
    nfeedback = len(item_ratings)
    ret = []
    if nfeedback < 3:
        len_for_train = nfeedback
    else:
        len_for_train = nfeedback - 1
    his_items, his_ratings, his_reviews = build_his_data(user, item_ratings[:len_for_train-1], review_dict)
    target_items = [item if (user, item) in review_dict else 0 for item, rating in item_ratings[1:len_for_train]]
    target_ratings = [rating for item, rating in item_ratings[1:len_for_train]]
    target_reviews = [review_dict[(user, item)] if (user, item) in review_dict else "" for item, rating in item_ratings[1:len_for_train]]
    
    ret.append({
        "his_items": his_items,
        "his_ratings": his_ratings,
        "his_reviews": his_reviews,
        "target_items": target_items,
        "target_ratings": target_ratings,
        "target_reviews": target_reviews,
    })
    return ret

def prepare_validate_data(user, item_ratings, review_dict):
    nfeedback = len(item_ratings)
    ret = []
    if nfeedback >= 3:
        target_item, target_rating = item_ratings[-1]
        his_items, his_ratings, his_reviews = build_his_data(user, item_ratings[:-1], review_dict)
        if (user, target_item) in review_dict:
            data = {
                "his_items": his_items,
                "his_ratings": his_ratings,
                "his_reviews": his_reviews,
                "target_item": target_item,
                "target_rating": target_rating,
                "target_review": review_dict[(user, target_item)]
            }
            ret.append(data)   
    return ret

def prepare_testing_data(user, item_ratings, review_dict, test_review_dict):
    nfeedback = len(item_ratings)
    ret = []
    if nfeedback >= 3:
        target_item, target_rating = item_ratings[-1]
        his_items, his_ratings, his_reviews = build_his_data(user, item_ratings[:-1], review_dict)
        if (user, target_item) in test_review_dict:
            data = {
                "his_items": his_items,
                "his_ratings": his_ratings,
                "his_reviews": his_reviews,
                "target_item": target_item,
                "target_rating": target_rating,
                "target_review": test_review_dict[(user, target_item)]
            }
            ret.append(data)   
    return ret

def data_partition(dataset_name):
    usernum = 0
    itemnum = 0
    if "-" in dataset_name:
        from_dataset, to_dataset = dataset_name.split('-')
    else:
        from_dataset = dataset_name
    # assume user/item index starting from 1
    logger.info("reading sequential data")
    sequential_data = open(f'../../data/{dataset_name}/sequential_data.txt', 'r').readlines()
    for line in sequential_data:
        user, item_ratings = line.split('\t', 1)
        user, user_type = user.split(',')
        user = int(user)
        usernum = max(user, usernum)
        item_ratings = [x.split(',') for x in item_ratings.split('\t')]
        item_ratings = [[int(item), 1 if float(rating) > 3 else 0] for item, rating in item_ratings]
        for item, rating in item_ratings:
            itemnum = max(item, itemnum)
    
    logger.info("reading metadata")
    metadata = f"../../data/{dataset_name}/metadata.jsonl"
    item2meta = ["padding"]
    for line in open(metadata):
        meta = json.loads(line)
        meta = {
            "title": meta["title"],
            "domain": meta["domain"]
        }
        item2meta.append(meta)

    logger.info("reading review data")
    review_dict = pickle.load(open(f"../../data/{args.dataset}/review.pkl", "rb"))
    data_train = []
    data_valid = []
    data_test = []
    test_review_dict = {}
    for line in open(args.test_review_path):
        data = json.loads(line)
        review = data["predict"].split("ASSISTANT:")[-1].split('</s>')[0].strip()
        test_review_dict[(data["user"], data["item"])] = review

    logger.info("building datasets")
    num_skip = 0
    for line in tqdm(sequential_data, desc="building datasets", ncols=80):
        user, item_ratings = line.split('\t', 1)
        user, user_type = user.split(',')
        user = int(user)
        usernum = max(user, usernum)
        user_type = int(user_type)
        item_ratings = [x.split(',') for x in item_ratings.split('\t')]
        item_ratings = [[int(item), 1 if float(rating) > 3 else 0] for item, rating in item_ratings]
        from_domain_item_ratings = [[item, rating] for item, rating in item_ratings if item2meta[item]["domain"]==from_dataset][-(args.max_his_len+1):]
        if len(from_domain_item_ratings):
            if num_skip < 1000:
                if from_domain_item_ratings[-1][1] == 1:
                    num_skip += 1
            else:
                data_train.extend(prepare_training_data(user, from_domain_item_ratings, review_dict))
                data_valid.extend(prepare_validate_data(user, from_domain_item_ratings, review_dict))
                data_test.extend(prepare_testing_data(user, from_domain_item_ratings, review_dict, test_review_dict))

    train_dataset = SASRecTrainingDataset(data_train)
    valid_dataset = SASRecPointTestingDataset(data_valid, split="valid")
    test_dataset = SASRecPointTestingDataset(data_test, split="test")
    return train_dataset, valid_dataset, test_dataset, \
        usernum, itemnum, item2meta

def train_step(model, train_data):
    with torch.no_grad():
        for key, value in train_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    train_data[key][sub_key] = sub_value.to(args.device)
            else:
                train_data[key] = value.to(args.device)
    loss = model(train_data)
    return loss

def validate(model, valid_dataset, best_result, key_metric): 
    model.eval()
    result = test_model(model, valid_dataset)
    if best_result == None or result[key_metric] > best_result[key_metric]:
        best_result = result
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    logger.info(f"Current result:{result}, Best Result:{best_result}.")
    return best_result

def fit(model, train_dataset, valid_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=train_dataset.collate_fn) # set num_workers > 1 for speeding up
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    one_epoch_step = len(train_dataset) // args.batch_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # decay the learning rate
    best_result = None
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        avg_loss = 0.0
        model.train()
        epoch_iterator = tqdm(train_dataloader, ncols=80)  # set ncols for the length of the progress bar
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient
            optimizer.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())  # show the learning rate and loss on the progress bar
            if i > 0 and i % (one_epoch_step // 5 + 1) == 0:  # evaluate every 20% data
                best_result = validate(model, valid_dataset, best_result, "AUC")
                model.train()
            avg_loss += loss.item()
        best_result = validate(model, valid_dataset, best_result, "AUC")
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.save_dir, f"model_epoch_{epoch}.bin"))
        model.train()
        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f}".format(avg_loss / cnt))
        scheduler.step()

def train_model(train_dataset, valid_dataset, usernum, itemnum):
    model = SASRec(itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info('* number of parameters: %d' % n_params)  # compute the number of parameters
    model = model.to(args.device)  # move the model to GPU
    # model = torch.nn.DataParallel(model)  # open this if using multi-GPU for training
    fit(model, train_dataset, valid_dataset)

def predict(model, test_dataset):
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=test_dataset.collate_fn)
    preds = []
    trues = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=80, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key, value in test_data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            test_data[key][sub_key] = sub_value.to(args.device)
                    else:
                        test_data[key] = value.to(args.device)
            logits = model.predict(test_data)
            preds.extend(logits.tolist())
            trues.extend(test_data["target_rating"].tolist())
    print(preds[:100], trues[:100])
    return {"AUC": calc_auc(preds, trues)}

def test_model(model, test_dataset):
    result = predict(model, test_dataset)
    return result

if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset, \
        usernum, itemnum, item2meta = data_partition(args.dataset)
    # print first two example in each test dataset
    # for data in point_test_dataset.data[:2]:
    #     print(data)
    #     print(item2meta[data["target_item"]])
    # for data in list_test_dataset.data[:2]:
    #     print(data)
    #     print(item2meta[data["target_item"]])
    #     for item in data["candidate_list"]:
    #         print(item2meta[item], end="")
    #     print()
    # for data in indomain_point_test_dataset.data[:2]:
    #     print(data)
    #     print(item2meta[data["target_item"]])
    # for data in indomain_list_test_dataset.data[:2]:
    #     print(data)
    #     print(item2meta[data["target_item"]])
    #     for item in data["candidate_list"]:
    #         print(item2meta[item], end="")
    #     print()
    
    if args.do_train:
        train_model(train_dataset, valid_dataset, usernum, itemnum)
    if args.do_test:
        model = SASRec(itemnum, args)
        if args.load_state_dict:
            model_state_dict = torch.load(args.load_state_dict)
        else:
            model_state_dict = torch.load(args.save_path)
        model.load_state_dict(model_state_dict)
        model = model.to(args.device)
        result = test_model(model, test_dataset)
        logger.info(f"test result:{result}")