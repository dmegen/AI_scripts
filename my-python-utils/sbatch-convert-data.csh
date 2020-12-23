#!/bin/tcsh

#SBATCH --job-name=convert_data
#SBATCH --account=p48500047
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=400G
#SBATCH --partition=dav
#SBATCH --output=data_to_AGL_heights_3d.log_job_%j

set script = data_to_AGL_heights_3d

setenv TMPDIR /glade/scratch/$USER/temp
mkdir -p $TMPDIR

module load cuda/10.1
ncar_pylib /glade/work/hardt/20191211_20200420

### Run program
srun ./$script.py

### Print job info to log
scontrol show job $SLURM_JOB_ID
