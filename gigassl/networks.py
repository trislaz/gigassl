import contextlib
import math
import os.path
import random
import warnings
from typing import Union, Type

import torch
import torch.nn as nn
import torch.nn.functional
import torchvision.models.resnet as resnet_factory

import sparseconvnet
from sparseconvnet import SparseConvNetTensor

class LinearWithMIL(nn.Module):
    def __init__(self, mil_model, linear_classifier, freeze_pooling_model):
        super().__init__()
        self.mil_model = mil_model
        self.linear_classifier = linear_classifier

        self.freeze_pooling_model = freeze_pooling_model

    def forward(self, tile_embeddings, tiles_original_locations, *args, **kwargs):
        with torch.no_grad() if self.freeze_pooling_model else contextlib.suppress():
            wsi_embedding = self.mil_model(tile_embeddings, tiles_original_locations, *args, **kwargs)
        return self.linear_classifier(wsi_embedding)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        self.linear_classifier.train(mode)
        self.mil_model.train(mode and not self.freeze_pooling_model)
        return self

class TileEmbedder(nn.Module):
    def __init__(self, model, frozen):
        super().__init__()
        self.model = model
        self.frozen = frozen

    def forward(self, tiles, tiles_original_locations):
        """
        Computes concurrent and independent tile embedding with the tile embedder.
        :param tiles:
            for not patient_classification:
                tensor of tiles of expected shape (B_batch, B_tiles, channels, width, height) with B_batch equal
                to the number of considered WSI, and B_tiles equal to the number of tiles per considered WSI
            for patient_classification:
                B_batch size list((B_wsi, B_tiles, channels, width, height))
        :return:
            for not patient_classification:
                a tensor of tiles embeddings of shape (B_batch, B_tiles, tile_latent_size)
            for patient_classification:
                B_batch size list((B_wsi, wsi_latent_size))
        """
        with torch.no_grad() if self.frozen else contextlib.suppress():
            assert isinstance(tiles, (list, tuple))
            assert isinstance(tiles_original_locations, (list, tuple))
            assert len(tiles_original_locations) == len(tiles)
            # not patient_classification:
            #        Flatten all tiles across all WSI:
            #        (n_wsis, varying_n_tiles, channels, width, height) ->
            #        (n_wsis*varying_n_tiles, channels, width, height)
            if not self.patient_classification:
                tiles = torch.vstack(tiles)
                return self.model(tiles)

            # patient_classification:
            #        B_batch size list((B_wsi, B_tiles, channels, width, height)) +
            #        tile embedder forward for each element
            return [self.model(patient_wsis, wsis_args)
                    for patient_wsis, wsis_args in zip(tiles, tiles_original_locations)]

class SparseConvMIL(nn.Module):
    def __init__(self, sparse_cnn: nn.Module, sparse_map_downsample: int, deterministic: bool):
        super().__init__()
        self.sparse_pooling = sparse_cnn
        self.sparse_adaptive_pooling = SparseAdaptiveAvgPool(1)
        self.sparse_map_downsample = sparse_map_downsample

        self.deterministic = deterministic

    @staticmethod
    def post_process_tiles_locations(tiles_locations):
        """
        Reformat the tiles locations into the proper expected format: the sparse-input CNN library sparseconvnet
            expects locations in the format
            [[tile1_loc_x, tile1_loc_y, batch_index_of_tile1],
             [tile2_loc_x, tile2_loc_y, batch_index_of_tile2],
             ...
             [tileN_loc_x, tileN_loc_y, batch_index_of_tileN]]
        :param tiles_locations: locations of sampled tiles with shape (B, n_tiles, 2) with B batch size, n_tiles the
            number of tiles per batch index and the other dimension for both coordinates_x and coordinates_y
        :return: a reformatted tensor of tiles locations with shape (n_tiles, 3)
        """
        device = tiles_locations[0].device
        # reshaped_tiles_locations = tiles_locations.view(tiles_locations.shape[0] * tiles_locations.shape[1], -1)
        reshaped_tiles_locations = torch.vstack(tiles_locations)
        repeated_batch_indexes = torch.tensor([[b] for b, tl in enumerate(tiles_locations)
                                               for _ in range(len(tl))]).to(device)
        return torch.cat((reshaped_tiles_locations, repeated_batch_indexes), dim=1)

    @staticmethod
    def __correct_size_for_convs(size):
        """ Changes the size such that two 3-width 2-stride convs result in a size correct for sparseconvnet.
            Size should be such that size//2 is odd, and (size//2)//2 is also odd. """
        # size of <5 implies that after 2 3/2 conv, there is no output
        if size < 5:
            size = 5
        if size % 2 == 0:
            size += 1
        if size // 2 % 2 == 0:
            size += 2
        return size

    def forward_sparse_cnn(self, tile_embeddings, tiles_locations):
        # Instantiates an empty sparse map container for sparseconvnet. Spatial_size is set to the maximum of tiles
        # locations for both axis; mode=4 implies that two embeddings at the same location are averaged elementwise
        spatial_size_x = int(torch.max(tiles_locations[:, 0])) + 1
        spatial_size_y = int(torch.max(tiles_locations[:, 1])) + 1

        spatial_size_x = self.__correct_size_for_convs(spatial_size_x)
        spatial_size_y = self.__correct_size_for_convs(spatial_size_y)
        input_layer = sparseconvnet.InputLayer(dimension=2, spatial_size=(spatial_size_x, spatial_size_y), mode=4)

        # Assign each tile embedding to their corresponding post-processed tile location
        sparse_map = input_layer([tiles_locations, tile_embeddings])

        wsi_embedding = self.sparse_pooling(sparse_map)
        wsi_embedding = self.sparse_adaptive_pooling(wsi_embedding)
        wsi_embedding = torch.flatten(wsi_embedding, start_dim=1)
        return wsi_embedding

    def forward(self, tile_embeddings, tiles_original_locations, *args, **kwargs):
        tiles_locations = [t.clone() for t in tiles_original_locations]
        tiles_locations = [t / self.sparse_map_downsample for t in tiles_locations]

        # Offsets tiles to the leftmost and topmost
        tiles_locations = [tiles_locations - torch.min(tiles_locations, dim=0, keepdim=True)[0]
                           for tiles_locations in tiles_locations]
        tiles_locations = self.post_process_tiles_locations(tiles_locations)
        tile_embeddings = tile_embeddings.view(-1, tile_embeddings.shape[-1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wsi_embedding = self.forward_sparse_cnn(tile_embeddings, tiles_locations)

        return wsi_embedding


    """AverageMIL."""

class AttentionMIL(nn.Module):
    def __init__(self, n_input_channels, attention_inner_neurons, gated_mechanism=False):
        super().__init__()

        self.gated_mechanism = gated_mechanism
        if not gated_mechanism:
            # Tanh-activated MLP
            self.attention_module = nn.Sequential(
                nn.Linear(n_input_channels, attention_inner_neurons, False),
                nn.Tanh(),
                nn.Linear(attention_inner_neurons, 1, False)
            )
        else:
            # Concatenation of tanh-activated MLP and sigmoid-activated MLP
            self.attention_module = nn.ModuleList([
                nn.Sequential(nn.Linear(n_input_channels, attention_inner_neurons), nn.Tanh()),
                nn.Sequential(nn.Linear(n_input_channels, attention_inner_neurons), nn.Sigmoid()),
                nn.Linear(attention_inner_neurons, 1, False)
            ])

    def forward(self, tiles_embeddings, _, *args, **kwargs):
        res = []
        for slide_tiles_embeddings in tiles_embeddings:
            slide_tiles_embeddings = slide_tiles_embeddings.unsqueeze(0)
            # Compute attention weights
            if not self.gated_mechanism:
                attention_scores = self.attention_module(slide_tiles_embeddings)
            else:
                sigm, tanh, combiner = self.attention_module
                attention_scores = combiner(sigm(slide_tiles_embeddings) * tanh(slide_tiles_embeddings))
            attention_scores = attention_scores.transpose(2, 1)
            attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

            res.append(torch.matmul(attention_weights, slide_tiles_embeddings).squeeze(-2))

        return torch.vstack(res)

class AverageMIL(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(tiles_embeddings, _, *args, **kwargs):
        return torch.vstack([torch.mean(t, dim=0) for t in tiles_embeddings])

class MaxAverageMIL(nn.Module):
    def __init__(self):
        super().__init__()

class SparseAdaptiveAvgPool(nn.AdaptiveAvgPool1d):
    """
    Custom pooling layer that transforms a (c, w, h) input sparse tensor into a (c,) output sparse tensor
    """

    def __init__(self, output_size):
        super().__init__(output_size)

    def forward(self, sparse_tensor_input):
        input_features = sparse_tensor_input.features
        input_locations = sparse_tensor_input.get_spatial_locations()

        res = []
        for batch_idx in torch.unique(input_locations[..., 2]):
            pooled = super().forward(input_features[input_locations[..., 2] == batch_idx].transpose(0, 1).unsqueeze(0))
            res.append(pooled)

        return torch.cat(res, dim=0)

def get_classifier(input_n_neurons: int, inner_n_neurons: int, n_classes: int):
    """
    Instantiates a ReLU-activated 1-hidden layer MLP.
    :param input_n_neurons: vector size of input data (should be WSI embedding)
    :param inner_n_neurons: number of inner neurons
    :param n_classes: number of output classes
    :return: a Sequential model
    """
    if inner_n_neurons is not None:
        return nn.Sequential(
            nn.Linear(input_n_neurons, inner_n_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(inner_n_neurons, n_classes, False),
        )
    return nn.Linear(input_n_neurons, n_classes, False)

def get_resnet_model(resnet_architecture: str, pretrained: Union[bool, str]):
    """
    Instantiates a ResNet architecture without the finale FC layer.
    :param resnet_architecture: the desired ResNet architecture (e.g. ResNet34 or Wide_Resnet50_2)
    :param pretrained: True to load an architecture pretrained on Imagenet, otherwise standard initialization
    :return: (a Sequential model, number of output channels from the returned model)
    """
    if isinstance(pretrained, str):
        imagenet_pretrained = pretrained.lower() == 'imagenet'
    else:
        imagenet_pretrained = pretrained
    assert resnet_architecture.lower() in resnet_factory.__all__
    resnet_model = getattr(resnet_factory, resnet_architecture.lower())(imagenet_pretrained, progress=True)

    if isinstance(pretrained, str) and pretrained.lower() == 'histo':
        ozanciga_state_dict = get_state_dict_ozanciga_histo_pretrained_resnet()
        try:
            resnet_model.load_state_dict(ozanciga_state_dict)
        except RuntimeError as e:
            if e == 'Error(s) in loading state_dict for ResNet:' \
                    '	Missing key(s) in state_dict: "fc.weight", "fc.bias".''':
                resnet_model.load_state_dict(ozanciga_state_dict, strict=False)
    n_output_channels = resnet_model.fc.in_features
    resnet_model.fc = nn.Sequential()
    return resnet_model, n_output_channels

def get_state_dict_ozanciga_histo_pretrained_resnet():
    model_link = 'https://github.com/ozanciga/self-supervised-histopathology/releases/download/' \
                 'tenpercent/tenpercent_resnet18.ckpt'
    output_folder = os.path.join('/tmp', 'downloaded_models')
    output_path = os.path.join(output_folder, 'ozanciga_tenpercent_resnet18.ckpt')
    if not os.path.exists(output_path):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        torch.hub.download_url_to_file(model_link, output_path)

    model = torch.load(output_path)
    return {key.replace('model.resnet.', ''): value for key, value in model['state_dict'].items()
            if 'fc.1.' not in key and 'fc.3.' not in key}

def sparse_conv3x3(in_planes: int, planes: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding"""
    if stride == 1:
        return sparseconvnet.SubmanifoldConvolution(2, in_planes, planes, 3, False)
    return sparseconvnet.Convolution(2, in_planes, planes, 3, stride, False)

class SparseBasicBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        norm = lambda n: sparseconvnet.BatchNormalization(n, eps=1e-5, momentum=.9)

        self.conv1 = sparse_conv3x3(in_planes, planes, stride)
        self.norm1 = norm(planes)
        self.relu1 = sparseconvnet.ReLU()
        self.conv2 = sparse_conv3x3(planes, planes, 1)
        self.norm2 = norm(planes)
        self.relu2 = sparseconvnet.ReLU()

        # downsample
        self.downsample = stride > 1
        self.expand = self.downsample and in_planes != planes
        if self.downsample:
            self.avg_pool = sparseconvnet.AveragePooling(2, 3, 2)
            if self.expand:
                self.convd = sparseconvnet.SubmanifoldConvolution(2, in_planes, planes, 1, False)
            self.normd = norm(planes)

    def load_weights_from(self, dense_block):
        self.load_conv_layer_weights_from(self.conv1, dense_block.get_submodule('conv1'))
        self.load_conv_layer_weights_from(self.conv2, dense_block.get_submodule('conv2'))
        if self.expand:
            self.load_conv_layer_weights_from(self.convd, dense_block.get_submodule('downsample').get_submodule('0'))

        self.load_bn_layer_weights(self.norm1, dense_block.get_submodule('bn1'))
        self.load_bn_layer_weights(self.norm2, dense_block.get_submodule('bn2'))
        if self.downsample:
            self.load_bn_layer_weights(self.normd, dense_block.get_submodule('downsample').get_submodule('1'))

    @staticmethod
    def load_conv_layer_weights_from(sparse_block_layer, dense_block_layer):
        layer_weights = dense_block_layer.weight
        layer_weights = layer_weights.permute(2, 3, 1, 0)
        layer_weights = layer_weights.view(layer_weights.shape[0] * layer_weights.shape[1], 1, *layer_weights.shape[2:])
        assert sparse_block_layer.weight.shape == layer_weights.shape, \
            (sparse_block_layer.weight.shape, layer_weights.shape)
        sparse_block_layer.weight = torch.nn.Parameter(layer_weights, True)

    @staticmethod
    def load_bn_layer_weights(sparse_bn_block_layer, dense_bn_block_layer):
        assert sparse_bn_block_layer.weight.shape == dense_bn_block_layer.weight.shape
        assert sparse_bn_block_layer.bias.shape == dense_bn_block_layer.bias.shape
        assert sparse_bn_block_layer.running_mean.shape == dense_bn_block_layer.running_mean.shape
        assert sparse_bn_block_layer.running_var.shape == dense_bn_block_layer.running_var.shape
        sparse_bn_block_layer.weight = dense_bn_block_layer.weight
        sparse_bn_block_layer.bias = dense_bn_block_layer.bias
        sparse_bn_block_layer.running_mean = dense_bn_block_layer.running_mean
        sparse_bn_block_layer.running_var = dense_bn_block_layer.running_var

    def forward(self, x):
        a = self.conv1(x)
        a = self.norm1(a)
        a = self.relu1(a)

        a = self.conv2(a)
        a = self.norm2(a)

        if self.downsample:
            identity_signal = self.avg_pool(x)
            if self.expand:
                identity_signal = self.convd(identity_signal)
            identity_signal = self.normd(identity_signal)
        else:
            identity_signal = x

        # sum both convoluted signal and residual signal
        output = SparseConvNetTensor()
        output.metadata = a.metadata
        output.spatial_size = a.spatial_size
        output.features = a.features + identity_signal.features
        output = self.relu2(output)
        return output

class SparseBottleneck(torch.nn.Module):
    expansion: int = 4

    def __init__(self, in_planes, planes, stride):
        super().__init__()
        norm = lambda n: sparseconvnet.BatchNormalization(n, eps=1e-5, momentum=.9)

        self.conv1 = sparseconvnet.SubmanifoldConvolution(2, in_planes, planes, 1, False)
        self.norm1 = norm(planes)
        self.relu1 = sparseconvnet.ReLU()
        self.conv2 = sparse_conv3x3(planes, planes, 1)
        self.norm2 = norm(planes)
        self.relu2 = sparseconvnet.ReLU()
        self.conv3 = sparseconvnet.SubmanifoldConvolution(2, planes, planes * self.expansion, 1, False)
        self.norm3 = norm(planes * self.expansion)
        self.relu3 = sparseconvnet.ReLU()

        # downsample
        self.downsample = stride > 1
        if self.downsample:
            self.avg_pool1 = sparseconvnet.AveragePooling(2, 3, 2)
            self.avg_pool2 = sparseconvnet.AveragePooling(2, 3, 2)
            self.convd = sparseconvnet.SubmanifoldConvolution(2, in_planes, planes * self.expansion, 1, False)
            self.normd = norm(planes * self.expansion)

    def load_weights_from(self, dense_block):
        SparseBasicBlock.load_conv_layer_weights_from(self.conv1, dense_block.get_submodule('conv1'))
        SparseBasicBlock.load_conv_layer_weights_from(self.conv2, dense_block.get_submodule('conv2'))
        SparseBasicBlock.load_conv_layer_weights_from(self.conv3, dense_block.get_submodule('conv3'))
        if self.downsample:
            SparseBasicBlock.load_conv_layer_weights_from(self.convd,
                                                          dense_block.get_submodule('downsample').get_submodule('0'))

        SparseBasicBlock.load_bn_layer_weights(self.norm1, dense_block.get_submodule('bn1'))
        SparseBasicBlock.load_bn_layer_weights(self.norm2, dense_block.get_submodule('bn2'))
        SparseBasicBlock.load_bn_layer_weights(self.norm3, dense_block.get_submodule('bn3'))
        if self.downsample:
            SparseBasicBlock.load_bn_layer_weights(self.normd,
                                                   dense_block.get_submodule('downsample').get_submodule('1'))

    def forward(self, x):
        a = self.conv1(x)
        a = self.norm1(a)
        a = self.relu1(a)

        if self.downsample:
            a = self.avg_pool1(a)
        a = self.conv2(a)
        a = self.norm2(a)
        a = self.relu2(a)

        a = self.conv3(a)
        a = self.norm3(a)

        if self.downsample:
            identity_signal = self.avg_pool2(x)
            identity_signal = self.convd(identity_signal)
            identity_signal = self.normd(identity_signal)
        else:
            identity_signal = x

        # sum both convoluted signal and residual signal
        output = SparseConvNetTensor()
        output.metadata = a.metadata
        output.spatial_size = a.spatial_size
        output.features = a.features + identity_signal.features
        output = self.relu3(output)
        return output

def cut_resnet_dense_sparse(resnet_model, cut_block, overwrite_dense_sparse_shape_differences=False):
    dense_subresnet = []
    to_sparse_subresnet = []
    for c, child in enumerate(resnet_model.children()):
        # unsparsifiable initial children from id 0 to 3
        if c < 3 + cut_block:
            dense_subresnet.append(child)
        else:
            to_sparse_subresnet.append(child)

    # sparsify remaining blocks except last one which is AdaptiveAvgPool2d
    sparse_subresnet = []
    n_input_channels = 64 * 2 ** max(1, cut_block - 2)
    for t, to_sparse_block in enumerate(to_sparse_subresnet[:-2]):
        current_block_id = 4 - (len(to_sparse_subresnet) - 2 - t - 1)
        resnet_block = resnet_model.get_submodule(f'layer{current_block_id}')

        # Build sparse inner block one by one by loading resnet weights
        sparse_inner_blocks = []
        n_inner_resnet_blocks = len(list(resnet_block.children()))
        for i in range(n_inner_resnet_blocks):
            associated_resnet_inner_block = resnet_block.get_submodule(str(i))
            if isinstance(associated_resnet_inner_block, resnet_factory.BasicBlock):
                type_sparse_block = SparseBasicBlock
                in_planes = n_input_channels if i == 0 else 2 * n_input_channels
                planes = 2 * n_input_channels
            elif isinstance(associated_resnet_inner_block, resnet_factory.Bottleneck):
                type_sparse_block = SparseBottleneck
                in_planes = 4 * n_input_channels if i == 0 else 8 * n_input_channels
                planes = 2 * n_input_channels
            else:
                raise ValueError(f'Unexpected block encountered: {associated_resnet_inner_block.__classname__}')

            stride = 2 if i == 0 else 1
            sparse_inner_block = type_sparse_block(in_planes, planes, stride)
            sparse_inner_block.load_weights_from(associated_resnet_inner_block)
            sparse_inner_blocks.append(sparse_inner_block)

        if overwrite_dense_sparse_shape_differences:
            layer_weights = torch.clone(sparse_inner_blocks[0].conv1.weight)
            sparse_inner_blocks[0].conv1 = sparseconvnet.SubmanifoldConvolution(2, 256, 512, 1, False)
            sparse_inner_blocks[0].conv1.weight = torch.nn.Parameter(layer_weights[..., :256, :], True)

            layer_weights = torch.clone(sparse_inner_blocks[0].convd.weight)
            sparse_inner_blocks[0].convd = sparseconvnet.SubmanifoldConvolution(2, 256, 2048, 1, False)
            sparse_inner_blocks[0].convd.weight = torch.nn.Parameter(layer_weights[..., :256, :], True)

        sparse_block = sparseconvnet.Sequential(*sparse_inner_blocks)
        sparse_subresnet.append(sparse_block)

        n_input_channels *= 2

    dense_subresnet = torch.nn.Sequential(*dense_subresnet)
    sparse_subresnet = torch.nn.Sequential(*sparse_subresnet)
    return dense_subresnet, sparse_subresnet

class FullSparseConvMIL(nn.Module):
    def __init__(self, args):
        super(FullSparseConvMIL, self).__init__()
        self.args = args
        self.model_path = args.model_path
        resnet_model, _ = get_resnet_model('resnet18', 'imagenet')
        _, sparse_subresnet = cut_resnet_dense_sparse(resnet_model, 4)
        pooling_model = SparseConvMIL(sparse_subresnet, args.downmap, False) #downsample 500 before
        n_classes = args.num_class
        linear_classifier = torch.nn.Linear(512, n_classes, False)
        freeze_pooling_model = args.freeze_pooling
        whole_model = LinearWithMIL(pooling_model, linear_classifier, freeze_pooling_model)
        self.net = whole_model
    def forward(self, x):
        if isinstance(x, dict):
            x = self.net(x['x'], x['loc'])
        else:
            x = self.net(*x)
        return x

