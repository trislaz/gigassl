from tilers import SharedAugTiler, NormalTiler
import pandas as pd
import os
from glob import glob
import torch
from argparse import ArgumentParser
import numpy as np

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--path_wsi', type=str, default='/home/username/data/WSIs')
    parser.add_argument('--ext', type=str, default='svs')
    parser.add_argument('--normalizer', type=str, default=None, help='Normalizer to apply to each image before encoding.')
    parser.add_argument('--path_outputs', type=str, default='/home/username/data/WSIs')
    parser.add_argument('--model_path', type=str, default='/home/username/data/WSIs')
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--mask_tolerance', type=int, default=0.1)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--tiler', type=str, default='imagenet')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--Naug', type=int, default=25)
    parser.add_argument('--Nt', type=int, default=25)
    parser.add_argument('--NWSI', type=int, default=100, help='Number of WSIs per job')
    parser.add_argument('--tiler_type', type=str, default='NormalTiler', help='Type of tiler to use. Options are "SharedAugTiler" and "NormalTiler"')
    return parser.parse_args()

args = get_arguments()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
job_id=int(os.environ["SLURM_ARRAY_TASK_ID"])
slides = sorted(glob(os.path.join(args.path_wsi, f'*.{args.ext}')))
for ind in range(args.NWSI):
    path_wsi = slides[job_id*args.NWSI+ind]
    # Tile one WSI
    it = eval(args.tiler_type)(path_wsi=path_wsi,level=args.level,
            path_outputs=args.path_outputs, size=args.size, device=args.device,
            tiler=args.tiler, model_path=args.model_path, normalizer=args.normalizer,
            mask_tolerance=args.mask_tolerance, Naug=args.Naug, Nt=args.Nt)
    it.tile_image()


