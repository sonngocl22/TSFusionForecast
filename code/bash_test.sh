#!/usr/bin/env bash
#SBATCH --job-name=test
#SBATCH --output=logs/test%j.log
#SBATCH --error=logs/test%j.err
#SBATCH --mail-user=le004@uni-hildesheim.de
#SBATCH --partition=NGPU,GPU,STUD
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-120

echo "Start test"
srun python /home/le004/master_thesis/git/TSFusionForecast/code/TFT_m4_tune_train.py
echo "End test"
