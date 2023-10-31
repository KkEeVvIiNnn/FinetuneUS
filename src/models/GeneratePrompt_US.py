import math

import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

import deepspeed
from deepspeed.compression.helper import recursive_getattr, recursive_setattr

from models.SASRec_review import SASRec_review
from models.SASRec import SASRec
from models.TextUserEncoder import TextUserEncoder

class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling

# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]

# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model

def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model

IGNORE_TOKEN_ID=-100

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, personalized_rep_dim)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, rep_size, prefix_length, hidden_size, num_hidden_layers, num_heads):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(rep_size, prefix_length * hidden_size // num_hidden_layers),
            torch.nn.Tanh(),
            torch.nn.Linear(prefix_length * hidden_size // num_hidden_layers, prefix_length * num_hidden_layers * 2 * hidden_size)
        )
        # self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    def forward(self, personalized_rep):
        past_key_values = self.trans(personalized_rep)
        past_key_values = past_key_values.view(
            -1,
            self.prefix_length,
            self.num_hidden_layers * 2, 
            self.num_heads,
            self.hidden_size // self.num_heads
        )
        # past_key_values = self.dropout(past_key_values)
        # (2*layer) * B * head * len * emb
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

class GeneratePrompt_US(PreTrainedModel):
    def __init__(self, args, llm_config):
        super().__init__(llm_config)
        self.args = args
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_name_or_path,
            config=llm_config,
            cache_dir=args.cache_dir,
        )
        if args.rec_model == "SASRec":
            self.rec_model = SASRec(args.itemnum, args)
        elif args.rec_model == "SASRec_review":
            self.rec_model = SASRec_review(args.itemnum, args)
        elif args.rec_model == "Bert":
            self.rec_model = TextUserEncoder(args.itemnum, args)
        if args.rec_model_load_state_dict:
            self.rec_model.load_state_dict(torch.load(args.rec_model_load_state_dict), strict=False)
        if args.freeze_rec:
            for name, param in self.rec_model.named_parameters():
                param.requires_grad = False
        if args.lora_dim > 0:
            self.llm_model = convert_linear_layer_to_lora(
                self.llm_model,
                args.lora_module_name,
                args.lora_dim
            )
            if args.only_optimize_lora:
                self.llm_model = only_optimize_lora_parameters(self.llm_model)
                self.llm_model = make_model_gradient_checkpointing_compatible(self.llm_model)
        if args.freeze_llm:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        
        # self.prefix_encoder = PrefixEncoder(
        #     self.rec_model.inner_size, 
        #     args.prefix_length,
        #     self.llm_model.config.hidden_size,
        #     self.llm_model.config.num_hidden_layers,
        #     self.llm_model.config.num_attention_heads
        # )
        self.prompt_generator = torch.nn.Sequential(
            torch.nn.Linear(self.rec_model.inner_size, self.llm_model.config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.llm_model.config.hidden_size, args.prefix_length * self.llm_model.config.hidden_size)
        )

    def gradient_checkpointing_enable(self):
        self.llm_model.gradient_checkpointing_enable()

    def gen_personalized_rep(self, his_items, his_ratings, his_reviews, user_description):
        '''
        his_items: (B, max_his_len)
        his_ratings: (B, max_his_len)
        his_reviews: (B, max_his_len, max_review_len)
        '''
        personalized_rep = self.rec_model(his_items, his_ratings, his_reviews, user_description)
        return personalized_rep

    def get_inputs(self, input_ids, attention_mask, labels, personalized_reps):
        B = input_ids.size()[0]
        prefix_prompt_length = self.args.prefix_length
        hidden_size = self.llm_model.config.hidden_size

        token_embeds = self.llm_model.get_input_embeddings()(input_ids)
        prompt_embeds = self.prompt_generator(personalized_reps).reshape(B, prefix_prompt_length, hidden_size)
        inputs_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
        attention_mask = torch.cat([torch.ones(B, prefix_prompt_length).to(attention_mask.device), attention_mask], dim=1)
        if labels is not None:
            labels = torch.cat([torch.full((B, prefix_prompt_length), IGNORE_TOKEN_ID, dtype=torch.long).to(attention_mask.device), labels], dim=1)
        return inputs_embeds, attention_mask, labels

    def forward(self, input_ids, attention_mask, labels, his_items, his_ratings, his_reviews, user_description):
        '''
        input_ids: (B, max_length)
        attention_mask: (B, max_length)
        labels: (B, max_length)
        '''
        B = input_ids.shape[0]
        personalized_rep = self.gen_personalized_rep(his_items, his_ratings, his_reviews, user_description)
        # past_key_values = self.prefix_encoder(personalized_rep)
        # prefix_attention_mask = torch.ones(B, self.args.prefix_length).to(input_ids.device)
        # attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        inputs_embeds, attention_mask, labels = self.get_inputs(input_ids, attention_mask, labels, personalized_rep)

        output = self.llm_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,
            # past_key_values=past_key_values,
        )

        return output

    def generate(self, input_ids, attention_mask, labels, his_items, his_ratings, his_reviews, user_description, generation_config):
        B = input_ids.shape[0]
        personalized_rep = self.gen_personalized_rep(his_items, his_ratings, his_reviews, user_description)
        inputs_embeds, attention_mask, labels = self.get_inputs(input_ids, attention_mask, None, personalized_rep)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            generation_config=generation_config
        )
        return outputs

    def compute_transition_scores(self, **kwargs):
        return self.llm_model.compute_transition_scores(**kwargs)

    def resize_token_embeddings(self, new_num_tokens):
        self.llm_model.resize_token_embeddings(new_num_tokens)
