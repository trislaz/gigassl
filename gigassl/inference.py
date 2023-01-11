from networks import FullSparseConvMIL
from arguments import get_args, Namespace
from argparse import ArgumentParser
from glob import glob
import os
import numpy as np
import yaml
import torch
import re 

def get_embeddings(model, embs, ntiles=50, rep_ensemble=20):
    Repre = []
    average = []
    hooker = type('hooker', (), {'item':None})()
    def hook(m, i, o):
        hooker.item = i[0].cpu()
    handle = model.net.linear_classifier.register_forward_hook(hook) 
    IDs = [os.path.basename(x).replace('_embeddings.npy', '') for x in glob(os.path.join(embs, 'tiles', '*_embeddings.npy'))]
    for i in IDs:
        im = torch.Tensor(np.load(os.path.join(embs, 'tiles', i+'_embeddings.npy')))
        xy = torch.Tensor(np.load(os.path.join(embs, 'coordinates', i+'_xy.npy')))/4
        repre=[]
        for rep in range(rep_ensemble):
            sample = torch.randint(0, im.shape[0], (ntiles,))
            im_s = im[sample, :].unsqueeze(0).cuda()
            xy_s = torch.Tensor(xy)[sample, :].unsqueeze(0).cuda()
            _ = model((im_s, xy_s))
            repre.append(hooker.item)
        repre = np.vstack([x[0].cpu().numpy() for x in repre])
        Repre.append(repre.mean(0))
    Repre = np.vstack(Repre)
    return Repre, IDs

def load_pretrained_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    config = ckpt['config']
    config.model.freeze_pooling = 1
    model = FullSparseConvMIL(config.model)
    state_dict = ckpt['state_dict']
    state_dict = {k.replace('backbone.', ''):w for k,w in state_dict.items() if k.startswith('backbone') and not 'classifier' in k}
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'classifier' in name:
            continue
        assert (param == state_dict[name]).all().item(), 'Weights not loaded properly'
    print('Loaded the weigths properly.')
    return model, config.dataset.nb_tiles

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--data', type=str, help='path to the data folder')
    parser.add_argument('--ensemble', type=int, default=20, help='number of ensemble')
    parser.add_argument('-o', type=str, default='./WSI_encodings', help='output folder')
    args = parser.parse_args()

    model, ntiles = load_pretrained_model(args.model)
    model.cuda()
    model.eval()
    Repre, ids = get_embeddings(model, args.data, args.ensemble, ntiles)
    outdir = os.path.join(args.o, os.path.basename(args.model).split('.')[0])
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'embeddings.npy'), Repre)
    np.save(os.path.join(outdir, 'ids.npy'), ids)

if __name__ == '__main__':
    main()
