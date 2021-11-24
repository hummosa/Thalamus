#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH -t 00:50:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mem=10G
#SBATCH --job-name=clean
#-------------------------------------------------------------------------------
## Run model 
python refactor_cognitive_serialtasks.py  