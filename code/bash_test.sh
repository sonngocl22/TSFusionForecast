#!/usr/bin/env bash
#SBATCH --job-name=tft_train_tuned
#SBATCH --output=logs/train_tuned%j.log
#SBATCH --error=logs/train_tuned%j.err
#SBATCH --mail-user=le004@uni-hildesheim.de
#SBATCH --partition=NGPU,GPU,STUD
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-120

echo "Start test"
srun python /home/le004/master_thesis/git/TSFusionForecast/code/TFT_m4_tuned_test.py
echo "End test"
