# NCCL_DEBUG=INFO
python ./src/infer.py \
    --method LoRA \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --llm_max_length 2048 \
    --cache_dir /home/v-weixu1/.cache \
    --lora_module_name "layers."\
    --lora_dim 8 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 128 \
    --load_state_dict ./output/LoRA/Movies_and_TV/epoch0_step2500/pytorch_model.bin \
    --output_dir ./output/LoRA/Movies_and_TV/epoch0_step2500/ \
    --data_dir ./data \
    --data_names Movies_and_TV_rating

# NCCL_DEBUG=INFO
# python ./src/infer.py \
#     --method LoRA \
#     --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
#     --llm_max_length 2048 \
#     --cache_dir /home/v-weixu1/.cache \
#     --lora_module_name "layers."\
#     --lora_dim 8 \
#     --load_state_dict ./output/LoRA/Movies_and_TV/epoch0_step500/pytorch_model.bin \
#     --per_device_eval_batch_size 1 \
#     --max_new_tokens 128 \
#     --output_dir ./output/LoRA/Movies_and_TV/epoch0_step500/ \
#     --data_dir ./data \
#     --data_names Movies_and_TV_list \
#     --list_type gen

# NCCL_DEBUG=INFO
# python ./src/infer.py \
#     --method LoRA \
#     --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
#     --llm_max_length 2048 \
#     --cache_dir /home/v-weixu1/.cache \
#     --lora_module_name "layers."\
#     --lora_dim 8 \
#     --load_state_dict ./output/LoRA/Movies_and_TV/epoch0_step500/pytorch_model.bin \
#     --per_device_eval_batch_size 1 \
#     --max_new_tokens 128 \
#     --output_dir ./output/LoRA/Movies_and_TV/epoch0_step500/ \
#     --data_dir ./data \
#     --data_names Movies_and_TV_list \
#     --list_type ppl