import openslide
import numpy as np
import os
from torchvision.transforms import ToTensor
from torchvision import transforms
from staintools import StainNormalizer
from PIL import Image
from torchvision.models import resnet18, resnet50
import torch
from utils import make_auto_mask, patch_sampling, get_image
from torch.utils.data import DataLoader, RandomSampler, Dataset
from kornia.augmentation import (
        ColorJitter, RandomResizedCrop, RandomGrayscale, RandomRotation, 
        RandomGaussianBlur, RandomGaussianNoise
        )

def load_moco_model(moco_weights_path, model_name='resnet50'):
    """
    Loads a resnet with moco pretrained weights.
    Args:
        moco_weights_path (str): Path to moco weights.
        model_name (str): Name of the model.
    Returns:
        model (torch.nn.Module): Model with moco weights.
    """

    model = eval(model_name)(pretrained=False)
    checkpoint = torch.load(moco_weights_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        if 'fc' in name:
            continue
        assert (param == state_dict[name]).all().item(), 'Weights not loaded properly'
    print('Loaded the weigths properly.')
    return model

class SharedAugTiler:
    """
    Tiles a WSI and encodes its tiles with a pretrained model.
    Special case with shared augmentations:
    - Naug batches of Nt tiles are sampled from the WSI.
    - Each batch is augmented with the same augmentations.
    - The model is applied to each batch.
    - The features are saved under /path_outputs/level_{level}/{tiler}/tiles/{name_wsi}/{aug_id}.npy
    - The coordinates of the tiles are saved under /path_outputs/level_{level}/{tiler}/coordinates/{name_wsi}/{aug_id}.npy
    """
    def __init__(self, path_wsi, level, path_outputs, size, device, tiler, model_path=None, normalizer=None, mask_tolerance=0.5, Naug=50, Nt=256, num_workers=5):
        self.Naug = Naug
        self.Nt = Nt
        self.num_workers = num_workers
        self.level = level 
        self.device = device
        self.size = (size, size)
        self.path_wsi = path_wsi 
        self.model_path = model_path 
        self.tiler = tiler
        self.normalize = normalizer is not None
        self.normalizer = self._get_normalizer(normalizer)
        self.name_wsi, self.ext_wsi = os.path.splitext(os.path.basename(self.path_wsi))
        self.outpath = self._set_out_path(os.path.join(path_outputs, f'level_{level}', f'{tiler}'), self.name_wsi)
        self.slide = openslide.open_slide(self.path_wsi)
        self.mask_tolerance = mask_tolerance
        self.mask_function = lambda x: make_auto_mask(x, mask_level=-1)

    def _set_out_path(self, path_outputs, name_wsi):
        outpath = {}
        outpath['tiles'] = os.path.join(path_outputs, 'tiles', name_wsi)
        outpath['coordinates'] = os.path.join(path_outputs, 'coordinates', name_wsi)
        outpath['example'] = os.path.join(path_outputs, 'examples')
        for key in outpath.keys():
            if not os.path.exists(outpath[key]):
                os.makedirs(outpath[key])
        return outpath

    def _get_transforms(self, aug=True, imagenet=True, shared=True):
        if aug:
            t = [
                RandomResizedCrop((224, 224), (0.5, 1), p=1, same_on_batch=shared),
                ColorJitter(0.4, 0.4, 0.4, 0.2, p=0.8, same_on_batch=shared),
                RandomGrayscale(p=0.2, same_on_batch=shared),
                RandomGaussianBlur([5,5], [0.1, 1], p=0.5, same_on_batch=shared),
                RandomGaussianNoise(0, 0.1, p=0, same_on_batch=shared)
                    ]
        else:
            t = [torch.nn.Identity()]
        if imagenet:
            t.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        else: 
            t.append(transforms.Normalize((0.6387467, 0.51136744, 0.6061169), (0.31200314, 0.3260718, 0.30386254)))
        return transforms.Compose(t)

    def _get_normalizer(self, normalizer):
        """
        Returns a StainNormalizer object.
        """
        if normalizer is None:
            return None
        else:
            target = np.array(Image.open('build_dataset/image_macenko_norm.png'))[:,:,:3]
            normalizer = StainNormalizer(normalizer)
            normalizer.fit(target)
            print('Normalizing with ', normalizer)
            return normalizer

    def tile_image(self):
        """tile_image.
        Main function of the class. Tiles the WSI and writes the outputs.
        WSI of origin is specified when initializing TileImage.
        """
        sample = patch_sampling(slide=self.slide, mask_level=-1, mask_function=self.mask_function, 
            analyse_level=self.level, patch_size=self.size, mask_tolerance = self.mask_tolerance)
        param_tiles = sample['params']
        print(len(param_tiles))
        model = self._get_model(self.tiler)
        model.eval()
        model.to(self.device)
        preprocess = self._get_transforms(aug=True, imagenet=True)
        self._forward_pass_WSI(model, param_tiles, preprocess, self.Naug, self.Nt)

    def _forward_pass_WSI(self, model, param_tiles, preprocess, Naug, Nt):
        """
        Forward pass of the WSI.

        :param model: torch.nn.Module, model to use for the forward pass. This implementation works with resnet18.
        :param param_tiles: list, output of the patch_sampling.
        :param preprocess: kornia.augmentation.AugmentationSequential, preprocessing to apply to the tiles.
        :param Naug: int, number of augmentations per WSI.
        :param Nt: int, number of tiles per augmentation.
        """
        hook = [0]
        def hook_l3(m, i, o):
            hook[0] = o
        model.layer3.register_forward_hook(hook_l3)
        model.eval()
        model.to(self.device)
        data = WSI_dataset(param_tiles, self.slide, self.normalizer)
        data_loader = DataLoader(data,
                batch_size=Nt,
                num_workers=self.num_workers, 
                sampler=RandomSampler(data, replacement=True, num_samples=Naug*Nt)) 
        for aug, (batch, paras) in enumerate(data_loader):
            paras = np.array(paras)
            batch = batch.to(self.device)
            batch = preprocess(batch)
            with torch.no_grad():
                batch = model(batch).squeeze().cpu().numpy()
            embeddings = torch.mean(hook[0], dim=(2, 3)).squeeze().cpu().numpy()
            np.save(os.path.join(self.outpath['tiles'], f'{aug}.npy'), embeddings)
            np.save(os.path.join(self.outpath['coordinates'], f'{aug}.npy'), paras)

    def _get_model(self, tiler):
        """
        Returns a torch.nn.Module object.
        """
        if tiler == 'imagenet':
            return resnet18(pretrained=True)
        elif tiler == 'moco':
            return load_moco_model(self.model_path, model_name='resnet18')
        else:
            raise ValueError('Tiler not implemented')
            
class NormalTiler(SharedAugTiler):
    def __init__(self, path_wsi, level, path_outputs, size, device, tiler, model_path=None, normalizer=None, mask_tolerance=0.5, Naug=50, Nt=256, num_workers=5):
        path_outputs = path_outputs+'_normal'
        super().__init__(path_wsi, level, path_outputs, size, device, tiler, model_path, normalizer, mask_tolerance, Naug, Nt, num_workers)
        self.aug = self._get_transforms(aug=False, imagenet=True)

    def _forward_pass_WSI(self, model, param_tiles, preprocess, Naug, Nt):
        """
        Forward pass of the WSI.

        :param model: torch.nn.Module, model to use for the forward pass. This implementation works with resnet18.
        :param param_tiles: list, output of the patch_sampling.
        :param preprocess: kornia.augmentation.AugmentationSequential, preprocessing to apply to the tiles.
        :param Naug: int, number of augmentations per WSI.
        :param Nt: int, number of tiles per augmentation.
        """
        hook = [0]
        def hook_l3(m, i, o):
            hook[0] = o
        model.layer3.register_forward_hook(hook_l3)
        model.eval()
        model.to(self.device)
        data = WSI_dataset(param_tiles, self.slide, self.normalizer)
        data_loader = DataLoader(data,
                batch_size=Nt,
                num_workers=self.num_workers) 
        E = np.zeros((len(data), 256))
        P = np.zeros((len(data), 2))
        for aug, (batch, paras) in enumerate(data_loader):
            paras = np.array(paras)
            batch = batch.to(self.device)
            batch = preprocess(batch)
            with torch.no_grad():
                batch = model(batch).squeeze().cpu().numpy()
            embeddings = torch.mean(hook[0], dim=(2, 3)).squeeze().cpu().numpy()
            E[aug*Nt:(aug+1)*Nt] = embeddings
            P[aug*Nt:(aug+1)*Nt] = paras
        np.save(os.path.join(self.outpath['tiles'], f'{self.name_wsi}_embeddings.npy'), E)
        np.save(os.path.join(self.outpath['coordinates'], f'{self.name_wsi}_xy.npy'), P)

    def _set_out_path(self, path_outputs, name_wsi):
        outpath = {}
        outpath['tiles'] = os.path.join(path_outputs, 'tiles')
        outpath['coordinates'] = os.path.join(path_outputs, 'coordinates')
        outpath['example'] = os.path.join(path_outputs, 'examples')
        for key in outpath.keys():
            if not os.path.exists(outpath[key]):
                os.makedirs(outpath[key])
        return outpath

class WSI_dataset(Dataset):
    def __init__(self, params, slide, name_wsi, normalizer=None):
        self.params = params
        self.normalizer = normalizer
        self.slide = slide
        self.name_wsi = name_wsi

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx]
        image = get_image(slide=self.slide, para=params, numpy=True)
        if self.normalizer:
            image = self.normalizer.transform(image)
        image = ToTensor()(image)
        return image, np.array(params[:2])

