#!/usr/bin/env bash
#SBATCH --job-name=tft
#SBATCH --output=logs/tft%j.log
#SBATCH --error=logs/tft%j.err
#SBATCH --mail-user=le004@uni-hildesheim.de
#SBATCH --partition=NGPU,GPU,STUD
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-120

srun python /home/le004/master_thesis/git/TSFusionForecast/code/TFT_m4_cluster_tune.py --job-index 1 --total-jobs 1
