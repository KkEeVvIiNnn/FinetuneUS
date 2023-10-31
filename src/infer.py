#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import sys
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import to_device, set_random_seed
from dataset.infer_dataset import InferDataset
from models.LoRA_US import LoRA_US
from models.GeneratePrompt_US import GeneratePrompt_US
from models.GenerateAdaptor_US import GenerateAdaptor_US



def parse_args():
    ## data arguments
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    ## model arguments
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Model method.",
    )
    parser.add_argument(
        "--llm_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--llm_max_length",
        type=int,
        default=2048,
        help="The max length of the large language model.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )

    ## LoRA method arguments
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')


    ## generate method arguments
    parser.add_argument("--max_his_len",
                        type=int,
                        default=20,
                        help="")
    parser.add_argument("--max_review_len",
                        type=int,
                        default=50,
                        help="")
    parser.add_argument("--itemnum",
                        type=int,
                        default=0,
                        help="")
    parser.add_argument("--rating_num",
                        type=int,
                        default=5,
                        help="")
    parser.add_argument("--hidden_units",
                        type=int,
                        default=50,
                        help="")
    parser.add_argument("--num_blocks",
                        type=int,
                        default=2,
                        help="")
    parser.add_argument("--num_heads",
                        type=int,
                        default=1,
                        help="")
    parser.add_argument("--dropout_rate",
                        type=float,
                        default=0.2,
                        help="")
    parser.add_argument("--rec_model",
                        type=str,
                        )
    parser.add_argument("--rec_model_load_state_dict",
                        type=str,
                        )
    parser.add_argument('--freeze_rec', action='store_true')
    parser.add_argument('--freeze_llm', action='store_true')
    
    ## generate prompt arguments
    parser.add_argument("--prefix_length",
                        type=int,
                        default=1,
                        help="")
    parser.add_argument("--prompt_type",
                        type=str,
                        )

    ## infering arguments
    parser.add_argument('--data_dir',
        type=str,
        help="Path to the dataset directory.",
        required=True,
    )
    parser.add_argument('--data_names',
        type=str,
        help="json file names containing the infering data.",
        required=True,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--list_type",
        type=str,
        default="gen",
    )
    parser.add_argument("--load_state_dict",
                        type=str,
                        default=None,
                        help="Where to load the model.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the result.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=128,
                        help="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda")
    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.llm_max_length,
        use_fast=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.unk_token
    llm_config = AutoConfig.from_pretrained(
        args.llm_model_name_or_path,
        cache_dir=args.cache_dir,
    )
    llm_config.pad_token_id = tokenizer.pad_token_id
    llm_config.use_cache = False

    if args.method == "Base":
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_name_or_path,
            config=llm_config,
            cache_dir=args.cache_dir,
        )
    elif args.method == "LoRA":
        model = LoRA_US(args, llm_config)
    elif args.method == "GeneratePrompt":
        model = GeneratePrompt_US(args, llm_config)
    elif args.method == "GenerateAdaptor":
        model = GenerateAdaptor_US(args, llm_config)

    if args.load_state_dict:
        model.load_state_dict(torch.load(args.load_state_dict), strict=False)
    model = model.to(device)

    # Prepare the data
    test_dataset = InferDataset(args, tokenizer, args.method, split="test")

    # Infer!
    print("***** Running inferencing *****")
    if "review" in args.data_names:
        generation_config = GenerationConfig.from_pretrained(
            args.llm_model_name_or_path,
            max_new_tokens=args.max_new_tokens,
        )
        model.eval()
        out_path = os.path.join(args.output_dir, f"{args.data_names}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        outf = open(out_path, "w")
        pbar = tqdm(total=len(test_dataset), desc="inferencing")
        target_reviews = test_dataset.targets
        idx = 0
        for step, batch in enumerate(test_dataset):
            batch = to_device(batch, device)
            outputs = model.generate(**batch, generation_config=generation_config)
            sentences = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            for sentence in sentences:
                data = {
                    "user": test_dataset.dataset[idx]["user"],
                    "item": test_dataset.dataset[idx]["target_itemid"],
                    "rating": test_dataset.dataset[idx]["target_rating"],
                    "predict": sentence,
                    "target": target_reviews[idx]
                }
                idx += 1
                outf.write(json.dumps(data)+'\n')
                outf.flush()
            pbar.update(1)

    if "rating" in args.data_names:
        model.eval()
        out_path = os.path.join(args.output_dir, f"{args.data_names}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        outf = open(out_path, "w")
        pbar = tqdm(total=len(test_dataset), desc="inferencing")
        target_ratings = test_dataset.targets
        pred_scores = []
        idx = 0
        for step, batch in enumerate(test_dataset):
            batch = to_device(batch, device)
            with torch.no_grad():
                score = -model(**batch).loss.float().item()
            pred_scores.append(score)
            if len(pred_scores) == 2:
                data = {
                    "predict": pred_scores,
                    "target": target_ratings[idx]
                }
                idx += 1
                outf.write(json.dumps(data)+'\n')
                outf.flush()
                pred_scores = []
            pbar.update(1)

    if "list" in args.data_names:
        if args.list_type == "gen":
            generation_config = GenerationConfig.from_pretrained(
                args.llm_model_name_or_path,
                max_new_tokens=args.max_new_tokens,
            )
            model.eval()
            out_path = os.path.join(args.output_dir, f"{args.data_names}_gen.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            outf = open(out_path, "w")
            pbar = tqdm(total=len(test_dataset), desc="inferencing")
            target_items = test_dataset.targets
            idx = 0
            for step, batch in enumerate(test_dataset):
                batch = to_device(batch, device)
                outputs = model.generate(**batch, generation_config=generation_config)
                sentences = tokenizer.batch_decode(outputs, skip_special_tokens=False)
                for sentence in sentences:
                    data = {
                        "predict": sentence,
                        "target": target_items[idx]
                    }
                    idx += 1
                    outf.write(json.dumps(data)+'\n')
                    outf.flush()
                pbar.update(1)
        else:
            model.eval()
            out_path = os.path.join(args.output_dir, f"{args.data_names}_ppl.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            outf = open(out_path, "w")
            pbar = tqdm(total=len(test_dataset), desc="inferencing")
            target_idx = test_dataset.targets
            pred_scores = []
            idx = 0
            for step, batch in enumerate(test_dataset):
                batch = to_device(batch, device)
                with torch.no_grad():
                    score = -model(**batch).loss.float().item()
                pred_scores.append(score)
                if len(pred_scores) == 10:
                    data = {
                        "predict": pred_scores,
                        "target": target_idx[idx]
                    }
                    idx += 1
                    outf.write(json.dumps(data)+'\n')
                    outf.flush()
                    pred_scores = []
                pbar.update(1)

if __name__ == "__main__":
    main()