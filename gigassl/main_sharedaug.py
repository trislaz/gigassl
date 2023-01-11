"""
Minor modifications from https://github.com/PatrickHua/SimSiam
"""
import os
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from numpy import math
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from arguments import get_args
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from optimizers import get_optimizer, LR_Scheduler
from datetime import datetime

from simclr import SimCLRMIL
from networks import FullSparseConvMIL
from train_dataset import EmbWSISharedAug

def main(device, args):
    augmentation ={ 
        'sampling': True,
        'vflip': args.flip,
        'hflip': args.flip,
        'rotations': args.rot,
        'vscale': args.scale,
        'hscale': args.scale}
    print(augmentation)

    train_dataset = EmbWSISharedAug(
            path=args.dataset.wsi, 
            ntiles=args.dataset.nb_tiles, 
            Naug=args.dataset.Naug,
            transform=augmentation)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = SimCLRMIL(FullSparseConvMIL, args.model).to(device)
    model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True 
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')

    avg_loss = 0.
    avg_output_std = 0.

    for epoch in global_progress:
        model.train()
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=False)
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            images1, images2 = ([x.to(device, non_blocking=True) for x in images1], [x.to(device, non_blocking=True) for x in images2])
            model.zero_grad()
            data_dict = model.forward(images1, images2)
            loss = data_dict['loss'].mean() # ddp
            output_std = data_dict['output_std']
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            
            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)
            w = 0.9
            avg_loss = w * avg_loss + (1 - w) * data_dict['loss']
            avg_output_std = w * avg_output_std + (1 - w) * output_std.item()
        
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
        collapse_level = max(0., 1 - math.sqrt(512) * avg_output_std)
        print(f'collapse level = {collapse_level}')

        # Save checkpoint
        if epoch % 100 == 0:
            model_path = os.path.join(args.ckpt_dir, f"{args.name}_e{epoch}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
            torch.save({
                'epoch': epoch+1,
                'state_dict': model.module.state_dict(),
                'config': args,
            }, model_path)

            print(f"Model saved to {model_path}")
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                f.write(f'{model_path}')

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')
