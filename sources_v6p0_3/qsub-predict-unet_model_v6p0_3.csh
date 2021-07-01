#!/bin/tcsh

### Job Name
#PBS -N PBS_job_predict-unet

### Charging account
#PBS -A p48500047

### Request one chunk of resources with 1 GPU and 300 GB of memory
#PBS -l select=1:ncpus=1:mem=100GB

### Allow job to run up to 12 hours
#PBS -l walltime=03:00:00

### Route the job to the casper queue
#PBS -q casper

set output_root_dir = /glade/work/hardt/models
set model_run_name  = unet_v6p0
set script = predict-unet_model_v6p0_3

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load cuda/10.1
source /glade/u/apps/opt/ncar_pylib/ncar_pylib.csh /glade/work/hardt/20191211_20200420

mkdir -p $output_root_dir/$model_run_name
cp ./$script.py $output_root_dir/$model_run_name/$script-$PBS_JOBID.py

### Run program
python ./$script.py

### Print job info to log
qstat -f $PBS_JOBID
mv ./PBS_job_predict-unet.o* $output_root_dir/$model_run_name/${SCRIPT}.$PBS_JOBID.log
