# NCCL_DEBUG=INFO
python ./src/infer.py \
    --method LoRA \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --llm_max_length 2048 \
    --cache_dir /home/v-weixu1/.cache \
    --lora_module_name "layers."\
    --lora_dim 8 \
    --load_state_dict ./output/LoRA/Books/step_2/pytorch_model.bin \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 128 \
    --output_dir ./output/LoRA/Books/ \
    --data_dir ./data \
    --data_names Books_rating