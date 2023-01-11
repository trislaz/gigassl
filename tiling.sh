#!/bin/bash
#SBATCH --job-name=augmented 
#SBATCH --array=0-4%5
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1             
#SBATCH --ntasks-per-node=1    
#SBATCH --hint=nomultithread   
#SBATCH --time=03:00:00        
#SBATCH --output=tiling_%x.out  
#SBATCH --error=tiling_%x.out   

tiler_type='SharedAugTiler'
path_wsi="./test_dataset/slides"
ext='svs'
path_outputs="./test_dataset/encoded"
model_path="/path/of/the/model/if/using/ssl/for/encoding.pth" 
level=1
size=224
tiler="imagenet" 
normalizer="macenko"
Naug=50
Nt=256
NWSI=10 # Number of WSI per job. set the array so that NWSI * Njobs = total number of WSI
num_worker=8 #Set it as the number of cpus per task

python build_dataset/main_tiling.py --path_wsi "$path_wsi" --ext $ext --level "$level" --tiler "$tiler" --size "$size" --model_path "$model_path" --path_outputs "$path_outputs" --normalizer "$normalizer" --Naug $Naug --Nt $Nt --NWSI $NWSI --tiler_type $tiler_type
