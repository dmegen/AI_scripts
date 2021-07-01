#!/bin/tcsh

#SBATCH --job-name=v6p1_1_predict
#SBATCH --account=p48500047
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=100G
#SBATCH --partition=dav
#SBATCH --output=predict-unet_model_v6p1_1.log_job_%j

set output_root_dir = /glade/work/hardt/models
set model_run_name  = unet_v6p1
set script = predict-unet_model_v6p1_1

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load cuda/10.1
ncar_pylib /glade/work/hardt/20191211_20200420

mkdir -p $output_root_dir/$model_run_name
cp ./$script.py $output_root_dir/$model_run_name/$script-$SLURM_JOB_ID.py

### Run program
srun ./$script.py

### Print job info to log
scontrol show job $SLURM_JOB_ID
mv ./*.log_job_$SLURM_JOB_ID $output_root_dir/$model_run_name/.
