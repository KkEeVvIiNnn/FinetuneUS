# NCCL_DEBUG=INFO 
deepspeed ./src/train.py \
    --method GenerateAdaptor \
    --prompt_type task \
    --rec_model SASRec_review \
    --max_his_len 20 \
    --hidden_units 50 \
    --max_review_len 50 \
    --itemnum 36659 \
    --rec_model_load_state_dict "./baselines/SASRec_review/outputs/Books-Movies_and_TV/l=2_h=1_d=50_lr=0.001/model.bin" \
    --lora_module_name "layers."\
    --lora_dim 8 \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --llm_max_length 700 \
    --cache_dir /home/jingyao/.cache \
    --data_dir ./data \
    --data_names Books_rating,Books_list,Books_review \
    --max_example_num_per_dataset 100000 \
    --output_dir ./output/GenerateAdaptor_tunerec_1024/Books \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --weight_decay 0 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 50 \
    --save_step 1000 \
    --seed 1234 \
    --zero_stage 3 \
    --print_loss \
    --deepspeed > GA_tr_nn_1024_Books.log