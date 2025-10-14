#!/bin/bash
#SBATCH --job-name=circuits008
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=2
#SBATCH --time=0-0:15:00
#SBATCH --output=/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/logs/slurm-%A_%a.out
#SBATCH --error=/om/user/ericjm/results/dictionary-circuits/dense_clustering/exp008/circuits/logs/slurm-%A_%a.err
#SBATCH --mem=16GB
#SBATCH --array=4,5,10,14,21,26,28,32,34,38,42,49,54,56,61,68,70,72,77,86,105,107,111,112,116,118,119,121,126,127,129,133,20,64,74,134,135,136,137,147,148,153,158,159,160,173,174,182,183,187,194

conda activate features
python /om2/user/ericjm/dictionary-circuits/dense_clustering/experiments/exp008/circuits/eval_circuit.py $SLURM_ARRAY_TASK_ID
