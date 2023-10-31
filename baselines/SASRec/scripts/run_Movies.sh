python main.py \
    --device=cuda \
    --dataset=Movies_and_TV-Books \
    --do_train \
    --do_test \
    --epochs 10 \
    --max_his_len 20 \
    --max_review_len 50 \
    --lr 0.001 \
    --dropout_rate 0.5 \
    --hidden_units 50 \
    --num_blocks 2 \
    --num_heads 1

# python main.py \
#     --device=cuda \
#     --dataset=Movies_and_TV-Books \
#     --do_test \
#     --max_his_len 20 \
#     --max_review_len 50 \
#     --hidden_units 50 \
#     --num_blocks 2 \
#     --num_heads 1 \
#     --load_state_dict "outputs/Movies_and_TV-Books/l=2_h=1_d=50_lr=0.001/model_epoch_0.bin"