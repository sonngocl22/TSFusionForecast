#!/usr/bin/env bash
#SBATCH --job-name=tft1
#SBATCH --output=logs/tft1-%j.log
#SBATCH --error=logs/tft1-%j.err
#SBATCH --mail-user=le004@uni-hildesheim.de
#SBATCH --partition=NGPU,GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-120

srun python /home/le004/master_thesis/git/TSFusionForecast/code/TFT_m4_cluster_tune.py --job-index 1 --total-jobs 1
