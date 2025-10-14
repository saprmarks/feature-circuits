#!/bin/bash
#SBATCH --job-name=circuits008
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=2
#SBATCH --time=0-1:00:00
#SBATCH --output=/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/logs/slurm-%A_%a.err
#SBATCH --mem=16GB
#SBATCH --array=0-1587%5

conda activate features
python /om2/user/ericjm/dictionary-circuits/dense_clustering/experiments/exp008/circuits/eval_circuit_all.py $SLURM_ARRAY_TASK_ID
