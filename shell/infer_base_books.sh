# NCCL_DEBUG=INFO
python ./src/infer.py \
    --method Base \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --llm_max_length 2048 \
    --cache_dir /home/v-weixu1/.cache \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 128 \
    --output_dir ./output/Base/Books/ \
    --data_dir ./data \
    --data_names Books_rating