#!/bin/bash

#  Setting GPU card
export CUDA_VISIBLE_DEVICES=0 

#  Define variables
NAME="XXX"       # output of model name
TRAIN_FILE="mace_train_final.xyz" 
VALID_FILE="mace_valid_final.xyz"
TEST_FILE="mace_test_final.xyz"

# 3. Run the training
# MACE will divide the data set for 90% training, 5% validation, 5% test
# --E0s="average": let the model to estimate the isolate atom energies automatically
# Another option: --E0s="{30: -48394.4754, 8: -2043.1727, 1: -13.5392}" need to have eV as unit

# For real research, You need to at least change max_num_epochs=800 and start_swa=400
mace_run_train \
    --name="$NAME" \
    --train_file="$TRAIN_FILE" \
    --valid_file="$VALID_FILE" \
    --test_file="$TEST_FILE" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=64 \
    --max_L=2 \
    --correlation=3 \
    --r_max=6.0 \
    --forces_weight=1000 \
    --energy_weight=10 \
    --batch_size=16 \
    --valid_batch_size=30 \
    --max_num_epochs=1000 \
    --start_swa=500 \
    --scheduler_patience=15 \
    --patience=30 \
    --eval_interval=4 \
    --ema \
    --swa \
    --error_table="PerAtomRMSE" \
    --default_dtype="float64" \
    --device=cuda \
    --seed=123 \
    --energy_key="energy" \
    --forces_key="forces" \
    --save_cpu

