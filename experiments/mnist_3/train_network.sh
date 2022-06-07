#!/bin/bash

program=train_network

# Choose the dataset
dataset=NR-NR
# dataset=R-R
# dataset=NR-R

mkdir $program
cd $program
mkdir $dataset
cd ..
dir=./$program/$dataset

num_epochs=256
batch_size=20
learning_rate=0.001
seed=123456789
num_layers=3
hidden_dim=32
max_l=11
diag_cg=1
device=cuda

name=${program}.dataset.${dataset}.num_epochs.${num_epochs}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.num_layers.${num_layers}.hidden_dim.${hidden_dim}.max_l.${max_l}.diag_cg.${diag_cg}
python3 $program.py --dataset=$dataset.gz --dir=$dir --name=$name --num_epochs=$num_epochs --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --num_layers=$num_layers --hidden_dim=$hidden_dim --max_l=$max_l --diag_cg=$diag_cg --device=$device
