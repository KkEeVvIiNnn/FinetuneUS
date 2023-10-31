# NCCL_DEBUG=INFO 
python ./src/infer.py \
    --method GeneratePrompt \
    --prefix_length 3 \
    --max_his_len 20 \
    --hidden_units 50 \
    --max_review_len 50 \
    --itemnum 36659 \
    --rec_model_load_state_dict "./baselines/SASRec_review/outputs/Movies_and_TV-Books/l=2_h=1_d=50_lr=0.001/model.bin" \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --cache_dir /home/v-weixu1/.cache \
    --llm_max_length 2048 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 128 \
    --data_dir ./data \
    --data_names Movies_and_TV_rating \
    --load_state_dict ./output/GeneratePrompt_p3_tr/Movies_and_TV/epoch0_step2000/pytorch_model.bin \
    --output_dir ./output/GeneratePrompt_p3_tr/Movies_and_TV/epoch0_step2000/
# NCCL_DEBUG=INFO 
python ./src/infer.py \
    --method GeneratePrompt \
    --prefix_length 3 \
    --max_his_len 20 \
    --hidden_units 50 \
    --max_review_len 50 \
    --itemnum 36659 \
    --rec_model_load_state_dict "./baselines/SASRec_review/outputs/Movies_and_TV-Books/l=2_h=1_d=50_lr=0.001/model.bin" \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --cache_dir /home/v-weixu1/.cache \
    --llm_max_length 2048 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 128 \
    --data_dir ./data \
    --data_names Movies_and_TV_rating \
    --load_state_dict ./output/GeneratePrompt_p3_tr/Movies_and_TV/epoch0_step3000/pytorch_model.bin \
    --output_dir ./output/GeneratePrompt_p3_tr/Movies_and_TV/epoch0_step3000/
# NCCL_DEBUG=INFO 
python ./src/infer.py \
    --method GeneratePrompt \
    --prefix_length 3 \
    --max_his_len 20 \
    --hidden_units 50 \
    --max_review_len 50 \
    --itemnum 36659 \
    --rec_model_load_state_dict "./baselines/SASRec_review/outputs/Movies_and_TV-Books/l=2_h=1_d=50_lr=0.001/model.bin" \
    --llm_model_name_or_path lmsys/vicuna-7b-v1.3 \
    --cache_dir /home/v-weixu1/.cache \
    --llm_max_length 2048 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 128 \
    --data_dir ./data \
    --data_names Movies_and_TV_rating \
    --load_state_dict ./output/GeneratePrompt_p3_tr/Movies_and_TV/epoch0_step9000/pytorch_model.bin \
    --output_dir ./output/GeneratePrompt_p3_tr/Movies_and_TV/epoch0_step9000/