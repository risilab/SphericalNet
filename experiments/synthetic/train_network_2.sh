#!/bin/bash

program=train_network_2
dir=./$program/
mkdir $dir

num_epochs=128
batch_size=100
learning_rate=0.001
seed=123456789
num_layers=3
hidden_dim=32
max_l=3
diag_cg=1
device=cuda

name=${program}.device.${device}.num_epochs.${num_epochs}.batch_size.${batch_size}.learning_rate.${learning_rate}.seed.${seed}.num_layers.${num_layers}.hidden_dim.${hidden_dim}.max_l.${max_l}.diag_cg.${diag_cg}
python3 $program.py --dir=$dir --name=$name --num_epochs=$num_epochs --batch_size=$batch_size --learning_rate=$learning_rate --seed=$seed --num_layers=$num_layers --hidden_dim=$hidden_dim --max_l=${max_l} --diag_cg=${diag_cg} --device=$device
