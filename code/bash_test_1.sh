#!/usr/bin/env bash
#SBATCH --job-name=tft$i
#SBATCH --output=logs/tft$i-%j.log
#SBATCH --error=logs/tft$i-%j.err
#SBATCH --mail-user=le004@uni-hildesheim.de
#SBATCH --partition=NGPU,GPU,STUD
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-120

srun python /home/le004/master_thesis/git/TSFusionForecast/code/TFT_m4_tune_train.py --job-index 1 --total-jobs 1
