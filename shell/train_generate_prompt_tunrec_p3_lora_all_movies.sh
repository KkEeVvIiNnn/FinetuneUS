# NCCL_DEBUG=INFO 
deepspeed --num_gpus=8 ./src/train.py \
    --method GeneratePrompt \
    --prompt_type all \
    --rec_model SASRec \
    --max_his_len 20 \
    --hidden_units 50 \
    --max_review_len 50 \
    --itemnum 36659 \
    --rec_model_load_state_dict "./baselines/SASRec/outputs/Movies_and_TV-Books/l=2_h=1_d=50_lr=0.001/model.bin" \
    --prefix_length 3 \
    --lora_module_name "layers."\
    --lora_dim 8 \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --llm_max_length 2048 \
    --cache_dir /home/v-weixu1/.cache \
    --data_dir ./data \
    --data_names Movies_and_TV_rating,Movies_and_TV_list,Movies_and_TV_review \
    --max_example_num_per_dataset 100000 \
    --output_dir ./output/GeneratePrompt_tr_p3_lora/Movies_and_TV \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --weight_decay 0 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 50 \
    --save_step 500 \
    --seed 1234 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --print_loss \
    --deepspeed > GP_tr_p3_lora_all_Movies_and_TV.log