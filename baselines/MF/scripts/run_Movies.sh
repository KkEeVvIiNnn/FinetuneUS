# for lr in {0.001,0.0001,0.0003}; do
#     for hidden_units in {50,100}; do
#         python main.py \
#             --device=cuda \
#             --dataset=Movies_and_TV-Books \
#             --do_train \
#             --do_test \
#             --epochs 10 \
#             --lr $lr \
#             --hidden_units $hidden_units
#     done
# done

python main.py \
    --device=cuda \
    --dataset=Movies_and_TV-Books \
    --do_train \
    --do_test \
    --epochs 10 \
    --lr 0.001 \
    --hidden_units 50