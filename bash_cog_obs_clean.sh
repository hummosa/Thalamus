#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH -t 00:50:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mem=10G
#SBATCH -o ./slurm/%j.out
#SBATCH -p halassa
#SBATCH --job-name=clean
#-------------------------------------------------------------------------------
## Run model 
python higher_cognition.py  
