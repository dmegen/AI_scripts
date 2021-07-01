#!/bin/tcsh

### Job Name
#PBS -N PBS_job_train-unet

### Charging account
#PBS -A p48500047

### Request one chunk of resources with 1 GPU and 300 GB of memory
#PBS -l select=1:ngpus=1:mem=300GB
#PBS -l gpu_type=v100

### Allow job to run up to 12 hours
#PBS -l walltime=04:00:00

### Route the job to the casper queue
#PBS -q casper

setenv  NEPTUNE_API_TOKEN "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOTFlYTI0ZDAtMzk3Zi00NzU0LWEzNWUtMzY2ZDcxZTA4OTg2In0="

#set CUDA_VISIBLE_DEVICES = 1 
set TF_FORCE_GPU_ALLOW_GROWTH=true

set output_root_dir = /glade/work/hardt/models
set model_run_name  = unet_v6p0
set script = train-unet_model_v6p0_3
set model = unet_model_v6p0_3

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load cuda/10.1
source /glade/u/apps/opt/ncar_pylib/ncar_pylib.csh /glade/work/hardt/20191211_20200420

mkdir -p $output_root_dir/$model_run_name
cp ./$script.py $output_root_dir/$model_run_name/$script-$PBS_JOBID.py
cp ./$model.py $output_root_dir/$model_run_name/$model-$PBS_JOBID.py

### Run program
python ./$script.py

### Print job info to log
qstat -f $PBS_JOBID
mv ./PBS_job_train-unet.o* $output_root_dir/$model_run_name/${script}.$PBS_JOBID.log
