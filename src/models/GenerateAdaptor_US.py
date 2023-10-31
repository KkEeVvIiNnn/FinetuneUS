import torch
from torch import nn
import torch.nn.functional as F
import math

from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from models.SASRec_review import SASRec_review
from models.SASRec import SASRec
from models.TextUserEncoder import TextUserEncoder
from deepspeed.compression.helper import recursive_getattr, recursive_setattr

IGNORE_TOKEN_ID=-100

class LoRAGenerator(nn.Module):  
    def __init__(self, input_size, lora_dim, rows, columns):  
        super(LoRAGenerator, self).__init__()  
        self.rows = rows
        self.columns = columns
        # Define layers
        self.lora_right_weight = nn.Parameter(torch.zeros(
            input_size,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows*columns))
        self.lora_scaling = 1 / lora_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def forward(self, input):
        # Forward pass  
        input = (input @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling
        input = input.view(-1, self.rows, self.columns)  # Reshape the output tensor to B*H*8  
        return input
    
class Personalized_Adaptor(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 personalized_rep_dim,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        super(Personalized_Adaptor, self).__init__()
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
        self.lora_right_weight_generator = LoRAGenerator(personalized_rep_dim, lora_dim, columns, lora_dim)
        self.lora_left_weight_generator = LoRAGenerator(personalized_rep_dim, lora_dim, lora_dim, rows)
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        # disable the original weight gradient
        self.weight.requires_grad = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def apply_personalized_rep(self, personalized_rep):
        self.personalized_rep = personalized_rep

    def forward(self, input):
        # x = torch.zeros_like(self.get_personalized_rep_func()).to(self.get_personalized_rep_func().device)
        # lora_right_weight = self.lora_right_weight_generator(x)
        # lora_left_weight = self.lora_left_weight_generator(x)
        lora_right_weight = self.lora_right_weight_generator(self.personalized_rep)
        lora_left_weight = self.lora_left_weight_generator(self.personalized_rep)

        value = torch.stack([self.lora_dropout(x)@y for x, y in zip(input, lora_right_weight)])
        value = torch.stack([x@y for x, y in zip(value, lora_left_weight)])
        return F.linear(input, self.weight, self.bias) + value * self.lora_scaling

# convert the linear layer to personalized adaptor
def convert_linear_layer_to_personalized_adaptor(model,
                                 part_module_name,
                                 personalized_rep_dim,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0,
                                 ):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        tmp = Personalized_Adaptor(
            module.weight, personalized_rep_dim, lora_dim, lora_scaling, lora_droppout,
            module.bias)
        tmp = tmp.to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
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

class GenerateAdaptor_US(PreTrainedModel):
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
        if args.freeze_rec:
            for name, param in self.rec_model.named_parameters():
                param.requires_grad = False
        self.llm_model = convert_linear_layer_to_personalized_adaptor(
            self.llm_model,
            args.lora_module_name,
            self.rec_model.inner_size,
            args.lora_dim,
        )
        if args.only_optimize_lora:
            self.llm_model = only_optimize_lora_parameters(self.llm_model)
            self.llm_model = make_model_gradient_checkpointing_compatible(self.llm_model)

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

    def apply_personalized_adaptor(self, personalized_rep):
        for name, module in self.llm_model.named_modules():
            if isinstance(module, Personalized_Adaptor):
                module.apply_personalized_rep(personalized_rep)

    def forward(self, input_ids, attention_mask, labels, his_items, his_ratings, his_reviews, user_description):
        '''
        input_ids: (B, max_length)
        attention_mask: (B, max_length)
        labels: (B, max_length)
        '''
        # print(self.rec_model.rating_emb.weight)
        personalized_rep = self.rec_model(his_items, his_ratings, his_reviews, user_description)
        self.apply_personalized_adaptor(personalized_rep)
        output = self.llm_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def generate(self, input_ids, attention_mask, labels, his_items, his_ratings, his_reviews, user_description, generation_config):
        personalized_rep = self.gen_personalized_rep(his_items, his_ratings, his_reviews, user_description)
        self.apply_personalized_adaptor(personalized_rep)
        outputs = self.llm_model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
        return outputs

    def compute_transition_scores(self, **kwargs):
        return self.llm_model.compute_transition_scores(**kwargs)

    def resize_token_embeddings(self, new_num_tokens):
        self.llm_model.resize_token_embeddings(new_num_tokens)
