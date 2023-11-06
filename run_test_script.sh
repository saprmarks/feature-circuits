#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59

#source $HOME/miniconda3/bin/activate
#conda activate fp

LR=$1
MAX_CONTEXTS=40

python test_script.py \
	--lr $LR \
	--steps 100000 \
	--contexts_per_step $MAX_CONTEXTS \
	--resample_steps 25000 \
	--save_steps 10000 \
	--in_bsz $MAX_CONTEXTS