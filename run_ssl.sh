#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node=1    
#SBATCH --hint=nomultithread   
#SBATCH --time=10:00:00        
#SBATCH --output=train_%x.out  
#SBATCH --error=train_%x.out 

config_file='gigassl/configs/simclr_mil_scm.yaml'
echo $SLURM_JOB_NAME
mkdir -p LOGS/${SLURM_JOB_NAME}
mkdir -p CHECKPOINTS/${SLURM_JOB_NAME}
python gigassl/main_sharedaug.py -c ${config_file} --log_dir LOGS/${SLURM_JOB_NAME}/ --ckpt_dir CHECKPOINTS/${SLURM_JOB_NAME}/ --scale --rot --flip
