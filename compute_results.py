from glob import glob
from linear_evaluation import main
import os
import argparse

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--fixsplit', action='store_true', help='Fix the splits')
args = parser.parse_args()

CSV = glob('tables/*.csv')
dataset = 'TCGA_encoded_t5e1000/'

for csv in CSV:
    task = csv.split('/')[-1].split('.')[0]
    if args.fixsplit:
        splitdirs = glob(f'splitdirs/{task}/*')
    else:
        splitdirs = [None]
    for splitdir in splitdirs:
        main(csv=csv, dataset=dataset, splitdir=splitdir)
