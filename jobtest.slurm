#!/bin/bash
#SBATCH --job-name=training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=%x.out
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate
cd /scratch/ps4521

python DLSQ4.py
