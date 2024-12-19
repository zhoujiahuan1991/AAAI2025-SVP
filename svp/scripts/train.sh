#! /bin/bash
conda activate <env_name>
cd path/to/code/svp;

declare -a array_datasets=(CUB200 CIFAR CIFAR10 DTD DOGS GTSRB NABIRDS SVHN FLOWERS FOOD)
declare -a array_lr=(6e-3 5e-3 5e-3 6.5e-3 5e-4 1e-2 1e-2 1.5e-2 1.5e-2 5e-3)
declare -a array_weight_decay=(1e-4 1e-5 1e-4 1e-4 1e-8 1e-6 0 1e-3 1e-4 1e-4)
declare -a array_shared_layers=(8 12 8 4 12 6 8 12 8 8)
declare -a array_mixup=(0.8 0.8 0.8 0.8 0.8 0.8 0.8 0 0 0.8)
declare -a array_cutmix=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 1.0)
declare -a array_seed=(0 0 0 3407 0 0 0 0 0 0)
declare -a array_batch_size=(96 96 96 96 64 96 96 96 96 96)
declare -a array_input_relu=(False True True False False False False False True False)
echo ${array_input_relu[${seed}]}
for seed in 0 1 2 3 4 5 6 7 8 9; do
    python main.py \
        --data-set ${array_datasets[${seed}]} \
        --transfer_type prompt \
        --prompt_type addv4 \
        --shared_layers ${array_shared_layers[${seed}]} \
        --prompt_add_gen mlp384*64 \
        --model vim_small \
        --batch-size 96 \
        --lr ${array_lr[${seed}]} \
        --min-lr 1e-5 \
        --warmup-lr 1e-5 \
        --weight-decay ${array_weight_decay[${seed}]} \
        --num_workers 5 \
        --epochs 100 \
        --no_amp \
        --mixup ${array_mixup[${seed}]} \
        --cutmix ${array_cutmix[${seed}]} \
        --seed ${array_seed[${seed}]} \
        --input_relu ${array_input_relu[${seed}]} \
        --output_dir ../logs \
        --finetune ../checkpoint/vim-small-midclstok/vim_s_midclstok_80p5acc.pth \
        --data-path path/to/your/data/root
done
