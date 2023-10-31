#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    GenerationConfig
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, save_zero_three_model, get_optimizer_grouped_parameters
from utils.ds_utils import get_train_ds_config
from utils.metrics import calc_auc

from dataset.train_dataset import TrainingDataset, IGNORE_TOKEN_ID
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
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )

    ## generate method arguments
    parser.add_argument("--max_his_len",
                        type=int,
                        default=20,
                        help="")
    parser.add_argument("--max_review_len",
                        type=int,
                        default=50,
                        help="")
    parser.add_argument("--max_user_len",
                        type=int,
                        default=512,
                        help="")
    parser.add_argument("--itemnum",
                        type=int,
                        default=0,
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
    parser.add_argument("--rec_model_load_state_dict",
                        type=str,
                        default=None
                        )
    parser.add_argument("--rec_model",
                        type=str,
                        )
    
    ## generate prompt arguments
    parser.add_argument("--prefix_length",
                        type=int,
                        default=3,
                        help="")
    parser.add_argument("--prompt_type",
                        type=str,
                        )


    ## training arguments
    parser.add_argument('--freeze_rec', action='store_true')
    parser.add_argument('--freeze_llm', action='store_true')
    parser.add_argument('--data_dir',
        type=str,
        help="Path to the dataset directory.",
        required=True,
    )
    parser.add_argument('--data_names',
        type=str,
        help="json file names containing the training data.",
        required=True,
    )
    parser.add_argument(
        "--max_example_num_per_dataset",
        type=int,
        default=500000000,
        help="",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--save_step",
                        type=int,
                        default=500,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload, stage=args.zero_stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.llm_max_length,
        use_fast=True,
        padding_side="right",
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
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

    elif args.method == "LoRA":
        model = LoRA_US(args, llm_config)
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)
    
    elif args.method == "GeneratePrompt":
        model = GeneratePrompt_US(args, llm_config)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": args.weight_decay,
            },
        ]
    elif args.method == "GenerateAdaptor":
        model = GenerateAdaptor_US(args, llm_config)
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_trainable_params}/{total_params}")

    # Prepare the data
    train_dataset = TrainingDataset(args, tokenizer, args.method, split="train")
    eval_dataset = InferDataset(args, tokenizer, args.method, split="valid")
    test_dataset = InferDataset(args, tokenizer, args.method, split="test")

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_dataloader = DataLoader(train_dataset,
                                    collate_fn=train_dataset.collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=eval_dataset.collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset,
                                    collate_fn=test_dataset.collator,
                                    sampler=test_sampler,
                                    batch_size=args.per_device_eval_batch_size)


    def list_metric(targets, candidate_lists, all_scores):
        candidate_num = len(candidate_lists[0])
        result = {
            "recall@1": 0,
            "recall@3": 0,
            "recall@5": 0
        }
        for idx, candidate_list in enumerate(candidate_lists):
            candidate_list = list(zip(range(candidate_num), candidate_list))
            scores = all_scores[idx*candidate_num:(idx+1)*candidate_num]
            sorted_list = sorted(candidate_list, key=lambda x:scores[x[0]], reverse=True)
            sorted_list = [x[1] for x in sorted_list]
            for k in [1, 3, 5]:
                if targets[idx] in sorted_list[:k]:
                    result[f"recall@{k}"] += 1
            result
        for key in result:
            result[key] = result[key] / len(targets)
        return result

    def rating_metric(targets, all_scores):
        predicts = []        
        for idx, target in enumerate(targets):
            scores = all_scores[idx*2: (idx+1)*2]
            predicts.append(np.exp(scores[1])/(np.exp(scores[0])+np.exp(scores[1])))
        return {"AUC": calc_auc(np.array(predicts), np.array(targets))}

    def review_metric(all_scores):
        all_scores = [score for score in all_scores if not np.isnan(score)]
        return {"ppl": -sum(all_scores)/max(1, len(all_scores))}

    def evaluation(model, dataloader, dataset):
        model.eval()
        losses = 0
        if args.global_rank == 0:
            pbar = tqdm(total=len(eval_dataloader), desc="inferencing")
        all_scores = []
        for step, batch in enumerate(dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
                if args.method == "GeneratePrompt":
                    start_idx = args.prefix_length
                else:
                    start_idx = 0
                shift_logits = outputs.logits[..., start_idx:-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                padding_mask = (shift_labels != IGNORE_TOKEN_ID)
                # total_length
                losses = torch.nn.functional.cross_entropy(shift_logits.view(-1, outputs.logits.size(-1)), shift_labels.view(-1), reduction='none')  
                # batch_size * seq_length
                losses = losses.view(outputs.logits.size(0), -1)  
                # batch_size
                individual_losses = (losses * padding_mask).sum(dim=1) / padding_mask.sum(dim=1) 
                scores = -individual_losses
                scores_list = [torch.zeros(outputs.logits.size(0), dtype=scores.dtype).to(device) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(scores_list, scores)
                for scores in zip(*scores_list):
                    for score in scores:
                        all_scores.append(score.float().item())
            if args.global_rank == 0:
                pbar.update(1)
        if "rating" in args.data_names:
            rating_result = rating_metric(dataset.rating_targets, all_scores[:len(dataset.rating_targets)*2])
            all_scores = all_scores[len(dataset.rating_targets)*2:]
        if "list" in args.data_names:
            list_result = list_metric(dataset.list_targets, dataset.candidate_items, all_scores[:len(dataset.list_targets)*10])
            all_scores = all_scores[len(dataset.list_targets)*10:]
        if "review" in args.data_names:
            review_result = review_metric(all_scores)
            
        return rating_result, list_result, review_result

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    def save_model(sub_dir):
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, sub_folder=sub_dir)
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                args.global_rank,
                os.path.join(args.output_dir, sub_dir),
                zero_stage=args.zero_stage)
            
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    rating_result, list_result, review_result = evaluation(model, eval_dataloader, eval_dataset)
    print_rank_0(f"{rating_result}\n{list_result}\n{review_result}", args.global_rank)
    rating_result, list_result, review_result = evaluation(model, test_dataloader, test_dataset)
    print_rank_0(f"{rating_result}\n{list_result}\n{review_result}", args.global_rank)

    best_rating_result = rating_result
    best_list_result = list_result
    best_review_result = review_result
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        if args.global_rank == 0:
            pbar = tqdm(total=len(train_dataloader), desc="training")
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            if args.print_loss:
                print_rank_0(f"Epoch: {epoch}, Step: {step}, loss = {loss}", args.global_rank)
            model.backward(loss)
            model.step()
            if args.global_rank == 0:
                pbar.update(1)
            if step and step % args.save_step == 0:
                rating_result, list_result, review_result = evaluation(model, eval_dataloader, eval_dataset)
                print_rank_0(f"{rating_result}\n{list_result}\n{review_result}", args.global_rank)
                rating_result, list_result, review_result = evaluation(model, test_dataloader, test_dataset)
                print_rank_0(f"{rating_result}\n{list_result}\n{review_result}", args.global_rank)
                model.train()
                if rating_result["AUC"] > best_rating_result["AUC"]:
                    best_rating_result = rating_result
                    print_rank_0(f'saving the final model as best rating model...', args.global_rank)
                    save_model("best_rating")
                if list_result["recall@3"] > best_list_result["recall@3"]:
                    best_list_result = list_result
                    print_rank_0(f'saving the final model as best list model...', args.global_rank)
                    save_model("best_list")
                if review_result["ppl"] < best_review_result["ppl"]:
                    best_review_result = review_result
                    print_rank_0(f'saving the final model as best review model...', args.global_rank)
                    save_model("best_review")

        # Evaluate                
        rating_result, list_result, review_result = evaluation(model, eval_dataloader, eval_dataset)
        print_rank_0(f"{rating_result}\n{list_result}\n{review_result}", args.global_rank)
        rating_result, list_result, review_result = evaluation(model, test_dataloader, test_dataset)
        print_rank_0(f"{rating_result}\n{list_result}\n{review_result}", args.global_rank)
        if rating_result["AUC"] > best_rating_result["AUC"]:
            best_rating_result = rating_result
            print_rank_0(f'saving the final model as best rating model...', args.global_rank)
            save_model("best_rating")
        if list_result["recall@3"] > best_list_result["recall@3"]:
            best_list_result = list_result
            print_rank_0(f'saving the final model as best list model...', args.global_rank)
            save_model("best_list")
        if review_result["ppl"] < best_review_result["ppl"]:
            best_review_result = review_result
            print_rank_0(f'saving the final model as best review model...', args.global_rank)
            save_model("best_review")

if __name__ == "__main__":
    main()