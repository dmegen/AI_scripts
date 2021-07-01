#!/bin/tcsh

### Job Name
#PBS -N PBS_job_v2_processing

### Charging account
#PBS -A p48500047

### Request one chunk of resources with 1 GPU and 300 GB of memory
#PBS -l select=1:ncpus=1:mem=300GB

### Allow job to run up to 12 hours
#PBS -l walltime=12:00:00

### Route the job to the casper queue
#PBS -q casper

set script = model2-v2_ref_W_processing

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load cuda/10.1
source /glade/u/apps/opt/ncar_pylib/ncar_pylib.csh /glade/work/hardt/20191211_20200420

### Run program
python ./$script.py

### Print job info to log
qstat -f $PBS_JOBID
