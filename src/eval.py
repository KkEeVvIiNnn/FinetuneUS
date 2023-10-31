import re
import json
import argparse

import numpy as np
from sklearn.metrics import auc as sk_auc

def _binary_clf_curve(trues, preds):
    """Calculate true and false positives per binary classification threshold

    Args:
        trues (numpy.ndarray): the true scores' list
        preds (numpy.ndarray): the predict scores' list

    Returns:
        fps (numpy.ndarray): A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]
        preds (numpy.ndarray): An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i].

    Note:
        To improve efficiency, we referred to the source code(which is available at sklearn.metrics.roc_curve)
        in SkLearn and made some optimizations.

    """
    trues = trues == 1

    desc_idxs = np.argsort(preds)[::-1]
    preds = preds[desc_idxs]
    trues = trues[desc_idxs]

    unique_val_idxs = np.where(np.diff(preds))[0]
    threshold_idxs = np.r_[unique_val_idxs, trues.size - 1]

    tps = np.cumsum(trues)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps


def calc_auc(preds, trues):
    fps, tps = _binary_clf_curve(trues, preds)
    if len(fps) > 2:
        optimal_idxs = np.where(
            np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    result = sk_auc(fpr, tpr)
    return result

def calc_mse(preds, trues):
    mse = np.square(preds - trues).mean()
    return mse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--tasks', type=str, default='direct_recommendation', help='tasks for data generation.')

    return parser.parse_args()

args = parse_args()

total = 0
error = 0
preds = []
trues = []

def extract_last_sen(text):  
    # Pattern to match sentences ending with a period not followed by a digit  
    pattern = r"(?<!\d)\.(?!\d|\s*\d)"  
    sentences = re.split(pattern, text)  
    return [s.strip() for s in sentences if s.strip()][-1]

if "rating" in args.tasks:
    targets = []
    for line in open(f"data/{args.dataset}_rating_test.jsonl"):
        line = json.loads(line)
        target = float(line["target"])
        targets.append(target)
    for idx, line in enumerate(open(f"output/{args.method}/{args.dataset}/predict.jsonl")):
        line = json.loads(line)
        target = targets[idx]
        patterns = [r"\"rating\": (\d(\.\d+)?)"]
        total += 1
        matched = False
        for pattern in patterns:
            match = re.search(pattern, line["predict"])
            if match:
                predict = float(match.group(1))
                preds.append(predict)
                trues.append(1 if target > 3 else 0)
                matched = True
                break
        if matched == False:
            error += 1
            print(line["predict"])
        
    print(f"err:{error/total:.3f}")
    print([(x,y) for x,y in zip(preds, trues)])
    print(f"auc: {calc_auc(np.array(preds)/5, np.array(trues))}")
    print(f"pred distribution: {sorted(preds)}")
    print(f"label distribution:{int(sum(trues))}/{len(trues)}")

if "list_select" in args.tasks:
    total = 0
    recall = {
        "recall@1": 0,
        "recall@3": 0,
        "recall@5": 0
    }
    for line in open(f"outputs/{args.dataset}/list_select/{args.method}/result.jsonl"):
        line = json.loads(line)
        if "select@1" in line:
            total += 1
            for k in [1,3,5]:
                target = line["target"]["title"]
                response = line[f"select@{k}"]["response"]
                if target in response:
                    recall[f"recall@{k}"] += 1

    for k in recall:
        recall[k] /= total

    print(recall)