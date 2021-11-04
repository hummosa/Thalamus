#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mem=10G
#SBATCH --job-name=Cog_obs
#-------------------------------------------------------------------------------
## Run model 
python cognitive_serialtasks.py  