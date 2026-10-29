#!/bin/bash

### TC2 Job Script ###

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1

### Specify Memory allocate to this job ###
##SBATCH --mem=1G

### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=1

### Specify number of node to compute ###
#SBATCH --nodes=1

### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
#SBATCH --nodelist=TC2N03

### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ###
#SBATCH --time=6:00:00

### Specify name for the job, filename format for output and error ###
#SBATCH --job-name=PEFT
#SBATCH --output=PEFT.out
#SBATCH --error=PEFT.err

### Your script for computation ###
module load anaconda
eval "$(conda shell.bash hook)"
conda activate PEFT
python run_tasks.py --config task.json

