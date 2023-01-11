from torch.utils.data import (Dataset, DataLoader, SubsetRandomSampler)
import pickle
from numpy import math
from sklearn.preprocessing import LabelEncoder
from joblib import load
import pandas as pd
from PIL import Image
from glob import glob
import numpy as np
import torch
from torchvision import transforms
from functools import reduce
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import os

class TwoCropsTransform:
    """
    Applies a base transform two times to the input.
    Used to create input for the self-supervised learning algorithm.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class LocationsAugmentation(torch.nn.Module):
    """
    Slide level transformation using as input embeddings vectors and location vectors.
    """
    def __init__(self, sampling=None, vflip=False, hflip=False, rotations=False, vscale=False, hscale=False, crop=False):
        """
        :param sampling: function, function that takes a matrix and a location vector and returns sampled matrices.
        :param vflip: bool, whether to flip vertically.
        :param hflip: bool, whether to flip horizontally.
        :param rotations: bool, whether to rotate.
        :param vscale: bool, whether to scale vertically.
        :param hscale: bool, whether to scale horizontally.
        :param crop: bool, whether to crop.
        """
        super().__init__()
        self.vflip = vflip
        self.hflip = hflip
        self.rotations = rotations
        self.vscale = vscale
        self.hscale = hscale
        self.crop = crop
        self.sampling = sampling

    def data_augment_tiles_locations_once(self, tiles_embeddings, tiles_locations):
        """
        :param tiles_embeddings: torch.Tensor , shape (N, F), where N is the number of tiles and F is the number of features.
        :param tiles_locations: torch.Tensor , shape (N, 2), where N is the number of tiles and 2 is the number of coordinates.
        :return: np.ndarray, shape (N, F), where N is the number of tiles and F is the number of features.
        """
        if callable(self.sampling):
            tiles_embeddings, tiles_locations = self.sampling(tiles_embeddings, tiles_locations)
        elif type(self.sampling) is int:
            indices =  torch.randint(0, tiles_locations.shape[0], (self.sampling, ))
            tiles_embeddings = tiles_embeddings[indices, :]
            tiles_locations = tiles_locations[indices, :]
        tiles_locations, tiles_embeddings = torch.Tensor(tiles_locations), torch.Tensor(tiles_embeddings)

        n_tiles, location_shape = tiles_locations.shape
        assert location_shape == 2, tiles_locations.shape
        device = tiles_locations.device

        tiles_embeddings = tiles_embeddings.clone()
        tiles_locations = tiles_locations.clone()
#        print('embeddings :', tiles_embeddings.shape,"locations : ", tiles_locations.shape)

        transform_matrix = torch.eye(2)
        # Random rotations
        if self.rotations:
            theta = (torch.rand((1,)) * 360 - 180).item()
            rot_matrix = torch.tensor([[math.cos(theta), -math.sin(theta)],
                                       [math.sin(theta), math.cos(theta)]])
            transform_matrix = rot_matrix

        # Random flips
        if self.vflip or self.hflip:
            flip_h = np.array([-1., 1.])[torch.randint(0,2,(1,))] if self.hflip else 1.
            flip_v = np.array([-1., 1.])[torch.randint(0,2,(1,))] if self.vflip else 1.
            flip_matrix = torch.tensor([[flip_h, 0.],
                                        [0., flip_v]]).float()
            transform_matrix = torch.mm(transform_matrix, flip_matrix)

        # Random resizes per axis
        if self.vscale or self.hscale:
            size_factor_h = 0.6 * torch.rand((1,)).item() + 0.7 if self.hscale else 1.
            size_factor_v = 0.6 * torch.rand((1,)).item() + 0.7 if self.vscale else 1.
            resize_matrix = torch.tensor([[size_factor_h, 0.],
                                          [0., size_factor_v]])
            transform_matrix = torch.mm(transform_matrix, resize_matrix)

        # First random translates ids, then apply matrix
        effective_sizes = torch.max(tiles_locations, dim=0)[0] - torch.min(tiles_locations, dim=0)[0]
        random_indexes = [torch.randint(0, max(int(size), 1), (1,)).item() for size in effective_sizes]
        translation_matrix = torch.tensor(random_indexes)
        tiles_locations -= translation_matrix.to(device)
        # Applies transformation
        tiles_locations = torch.mm(tiles_locations.float(), transform_matrix.to(device)).long()
        return tiles_embeddings, tiles_locations

    def forward(self, tiles):
        tiles_embeddings, tiles_locations = tiles
        return self.data_augment_tiles_locations_once(tiles_embeddings, tiles_locations)

class EmbWSISharedAug(Dataset):
    def __init__(self, path, ntiles, Naug, transform=None):
        super(EmbWSISharedAug, self).__init__()
        self.path = path
        self.ntiles = ntiles
        self.Naug = Naug
        self.files = glob(os.path.join(self.path,  'tiles', '*'))
        self.info = self._get_info()
        if transform is not None:
            if transform['sampling']:
                transform['sampling'] = self.sample_inside_wsi
            self.transform = TwoCropsTransform(LocationsAugmentation(**transform))

    def __len__(self):
        return len(self.files)

    def _get_info(self):
        """Assures that the info idx corresponds to the right file.
        """
        info = []
        for f in self.files:
            name = os.path.basename(f).replace('_embedded.npy', '')
            i = os.path.join(self.path, 'coordinates', name)
            assert os.path.exists(i), f"no info found for {f} at {i}"
            info.append(i)
        return info

    def __getitem__(self, idx):
        file_root = self.files[idx]
        info_root = self.info[idx]
        if self.transform:
            mat, info = self.transform((file_root, info_root))
        return (mat, info), file_root

    def sample_inside_wsi(self, mat, info):
        """Adapted to the new sharedaug format.
        Each slide has its own folder with the embedded matrices corresponding to each augmentation labeled 1-Naug.npy.
        params:
        - mat: str, path to the mat folder
        - info: str, path to the info folder
        """
        aug = torch.randint(self.Naug, (1,))[0]
        mat = np.load(os.path.join(mat, '{}.npy'.format(aug)))
        info = np.load(os.path.join(info, '{}.npy'.format(aug)))
        selection = torch.randint(0, mat.shape[0], (self.ntiles,))
        mat = mat[selection, :]
        info = info[selection, :]
        return mat, info

    def _get_sharedaug_params(self):
        return 50, 256
