#!/usr/bin/env bash
#SBATCH --job-name=job-generator
#SBATCH --output=logs/job-generator_%j.log
#SBATCH --partition=NGPU,GPU,STUD,TEST

#Echo bash info
echo "Executing job generator"
echo ""  #Echo new line

# Define the number of jobs
NUM_JOBS=10

# Loop to create and submit jobs
for i in $(seq 1 $NUM_JOBS); do

    JOB_FILE="job_$i.sh"

    # Create a SLURM job file
    cat <<EOF > $JOB_FILE
#!/usr/bin/env bash
#SBATCH --job-name=tft$i
#SBATCH --output=logs/tft$i-%j.log
#SBATCH --error=logs/tft$i-%j.err
#SBATCH --mail-user=le004@uni-hildesheim.de
#SBATCH --partition=NGPU,GPU
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-120

python /home/le004/master_thesis/git/TSFusionForecast/code/TFT_m4_cluster_tune.py --job-index $i --total-jobs $NUM_JOBS
EOF

    # Submit the job
    sbatch $JOB_FILE

    # Delete the job file
    rm $JOB_FILE

done

echo "Submitted $NUM_JOBS jobs."
