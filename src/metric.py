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

import jellyfish
def sim(str1, str2):
    return jellyfish.jaro_winkler_similarity(str1, str2)

import re
def extract_content_between_quotes(text):  
    matches = re.findall(r'"(.*?)"', text)  
    return matches

def parse_gen_list_result(result, k):
    result = line["predict"].split("ASSISTANT:")[-1].strip().split("</s>")[0].strip()
    if "\"" in result:
        result = extract_content_between_quotes(result)
    elif '\n' in result:
        result = result.split('\n')
    else:
        result = result.split(',')
    return [item for item in result if item]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--task', type=str)

    return parser.parse_args()

args = parse_args()

if args.task == "rating":
    targets = []
    predicts = []
    for line in open(args.input):
        line = json.loads(line)
        target = float(line["target"])
        targets.append(target)
        predict = line["predict"]
        predicts.append(np.exp(predict[1])/(np.exp(predict[0])+np.exp(predict[1])))

    print(f"auc: {calc_auc(np.array(predicts), np.array(targets))}")

if args.task == "list_gen":
    total = 0
    recall = {
        "recall@1": 0,
        "recall@3": 0,
        "recall@5": 0
    }
    klist = [1, 3, 5]
    for idx, line in enumerate(open(args.input).readlines()):
        line = json.loads(line)
        target = line["target"]
        topk = klist[idx % 3]
        predict = parse_gen_list_result(line["predict"], topk)
        flag = False
        for item in predict[:topk]:
            if sim(item, target) > 0.7:
                flag = True
        if flag:
            recall[f"recall@{topk}"] += 1
        if topk == 1:
            total += 1
        # if topk == 1:
        #     print("--------------------------------\n",target)
        # print(topk, predict)
    for k in recall:
        recall[k] /= total
    print(recall)

if args.task == "list_ppl":
    total = 0
    recall = {
        "recall@1": 0,
        "recall@3": 0,
        "recall@5": 0
    }
    targets = []
    predicts = []
    for line in open(args.input):
        line = json.loads(line)
        target = line["target"]
        targets.append(target)
        predict = list(enumerate(line["predict"]))
        predict.sort(key=lambda x:-x[1])
        predict = [x[0] for x in predict]
        total += 1
        for k in [1, 3, 5]:
            if target in predict[:k]:
                recall[f"recall@{k}"] += 1
    for k in recall:
        recall[k] /= total
    print(recall)
    