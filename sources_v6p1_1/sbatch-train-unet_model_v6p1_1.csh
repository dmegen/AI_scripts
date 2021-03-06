#!/bin/tcsh

#SBATCH --job-name=v6p1_1
#SBATCH --account=p48500047
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --constraint=v100
#SBATCH --mem=300G
#SBATCH --partition=dav
#SBATCH --exclusive
#SBATCH --output=train-unet_model_v6p1_1.log_job_%j

setenv  NEPTUNE_API_TOKEN "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOTFlYTI0ZDAtMzk3Zi00NzU0LWEzNWUtMzY2ZDcxZTA4OTg2In0="

#set CUDA_VISIBLE_DEVICES = 1 
set TF_FORCE_GPU_ALLOW_GROWTH=true

set output_root_dir = /glade/work/hardt/models
set model_run_name  = unet_v6p1
set script = train-unet_model_v6p1_1
set model = unet_model_v6p1_1

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load cuda/10.1
ncar_pylib /glade/work/hardt/20191211_20200420

mkdir -p $output_root_dir/$model_run_name
cp ./$script.py $output_root_dir/$model_run_name/$script-$SLURM_JOB_ID.py
cp ./$model.py $output_root_dir/$model_run_name/$model-$SLURM_JOB_ID.py

### Run program
srun ./$script.py

### Print job info to log
scontrol show job $SLURM_JOB_ID
mv ./*.log_job_$SLURM_JOB_ID $output_root_dir/$model_run_name/.
