#!/bin/bash
#SBATCH --job-name=infer 
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --ntasks-per-node=1    
#SBATCH --hint=nomultithread   
#SBATCH --time=03:00:00        
#SBATCH --output=infer_%x.out  
#SBATCH --error=infer_%x.out   

model='/path/to/the/gigassl/model.pth'
data='/path/to/the/WSI/to/encode'
ensemble=20
o='/path/to/the/output/directory'

python gigassl/inference.py --model $model --data $data --ensemble $ensemble -o $o
