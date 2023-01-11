"""
Code mostly written by Peter Naylor.
see https://github.com/PeterJackNaylor/useful_wsi
"""
import numpy as np
import itertools
import openslide
import os
from skimage._shared.utils import warn
from skimage.color import rgb2gray
from skimage.morphology import square, closing, opening
from skimage.filters import threshold_otsu

def get_image(slide, para, numpy=True):
    """
    Returns cropped image given a set of parameters.
    You can feed a string or an openslide image.
    Args:
        slide : String or openslide object from which we extract.
        para : List of 5 integers corresponding to: [x, y, size_x_level, size_y_level, level]
        numpy : Boolean, by default True, wether or not to convert the output to numpy array instead
                of PIL image.
    Returns:
        A tile (or crop) from slide corresponding to para. It can be a numpy array
        or a PIL image.

    """
    if isinstance(para, dict):
        slide = openslide.open_slide(slide) if isinstance(slide, str) else slide
        slide = slide.read_region((para['x'], para['y']), para['level'], (para['xsize'], para['ysize']))
        if numpy:
            slide = np.array(slide)[:, :, 0:3]
    else:
        if len(para) != 5:
            raise NameError("Not enough parameters...")
        slide = openslide.open_slide(slide) if isinstance(slide, str) else slide
        slide = slide.read_region((para[0], para[1]), para[4], (para[2], para[3]))
        if numpy:
            slide = np.array(slide)[:, :, 0:3]
    return slide
 
def make_auto_mask(slide, mask_level):
    """make_auto_mask. Create a binary mask from a downsampled version
    of a WSI. Uses the Otsu algorithm and a morphological opening.

    :param slide: WSI. Accepted extension *.tiff, *.svs, *ndpi.
    :param mask_level: level of the pyramidal WSI used to create the mask.
    :return mask: ndarray. Binary mask of the WSI. Dimensions are the one of the 
    dowsampled image.
    """
    if mask_level < 0:
        mask_level = len(slide.level_dimensions) + mask_level
    slide = openslide.open_slide(slide) if isinstance(slide, str) else slide
    im = slide.read_region((0,0),mask_level, slide.level_dimensions[mask_level]) 
    im = np.array(im)[:,:,:3]
    im_gray = rgb2gray(im)
    im_gray = clear_border(im_gray, prop=30)
    size = im_gray.shape
    im_gray = im_gray.flatten()
    pixels_int = im_gray[np.logical_and(im_gray > 0.02, im_gray < 0.98)]
    t = threshold_otsu(pixels_int)
    mask = opening(closing(np.logical_and(im_gray<t, im_gray>0.02).reshape(size), selem=square(2)), selem=square(2))
    final_mask = mask
    return mask

def clear_border(mask, prop):
    r, c = mask.shape
    pr, pc = r//prop, c//prop
    mask[:pr, :] = 0
    mask[r-pr:, :] = 0
    mask[:,:pc] = 0
    mask[:,c-pc:] = 0
    return mask

def get_x_y(slide, point_l, level, integer=True):
    """
    Code @PeterNaylor from useful_wsi.
    Given a point point_l = (x_l, y_l) at a certain level. This function
    will return the coordinates associated to level 0 of this point point_0 = (x_0, y_0).
    Args:
        slide : Openslide object from which we extract.
        point_l : A tuple, or tuple like object of size 2 with integers.
        level : Integer, level of the associated point.
        integer : Boolean, by default True. Wether or not to round
                  the output.
    Returns:
        A tuple corresponding to the converted coordinates, point_0.
    """
    x_l, y_l = point_l
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])
  
    x_0 = x_l * size_x_0 / size_x_l
    y_0 = y_l * size_y_0 / size_y_l
    if integer:
        point_0 = (int(x_0), int(y_0))
    else:
        point_0 = (x_0, y_0)
    return point_0

def grid_blob(slide, point_start, point_end, patch_size,
              analyse_level):
    """
    Forms a uniform grid starting from the top left point point_start
    and finishes at point point_end of size patch_size at level analyse_level
    for the given slide.
    Args:
        slide : String or open_slide object. 
        point_start : Tuple like object of integers of size 2.
        point_end : Tuple like object of integers of size 2.
        patch_size : Tuple like object of integers of size 2.
        analse_level : Integer. Level resolution to use for extracting the tiles.
    Returns:
        List of coordinates of grid.
    """
    if analyse_level == 0:
        patch_size_0 = patch_size
    else:
        patch_size_0 = get_size(slide, patch_size, analyse_level, 0)
    size_x, size_y = patch_size_0
    list_x = range(point_start[0], point_end[0], size_x)
    list_y = range(point_start[1], point_end[1], size_y)
    return list(itertools.product(list_x, list_y))

def get_size(slide, size_from, level_from, level_to, integer=True):
    """
    Given a size (size_from) at a certain level (level_from), this function will return
    a new size (size_to) but at a different level (level_to).
    Args:
        slide : Openslide object from which we extract.
        size_from : A tuple, or tuple like object of size 2 with integers.
        level_from : Integer, initial level.
        level_to : Integer, final level.
        integer : Boolean, by default True. Wether or not to round
                  the output.
        Returns:
            A tuple, or tuple like object of size 2 with integers corresponding 
            to the new size at level level_to. Or size_to.
    """
    size_x, size_y = size_from
    downsamples = slide.level_downsamples
    scal = float(downsamples[level_from]) / downsamples[level_to]
    if integer:
        func_round = round
    else:
        func_round = lambda x: x
    size_x_new = func_round(float(size_x) * scal)
    size_y_new = func_round(float(size_y) * scal)
    size_to = size_x_new, size_y_new
    return size_to

def get_x_y_from_0(slide, point_0, level, integer=True):
    """
    Given a point point_0 = (x0, y0) at level 0, this function will return 
    the coordinates associated to the level 'level' of this point point_l = (x_l, y_l).
    Inverse function of get_x_y
    Args:
        slide : Openslide object from which we extract.
        point_0 : A tuple, or tuple like object of size 2 with integers.
        level : Integer, level to convert to.  
        integer : Boolean, by default True. Wether or not to round
                  the output.
    Returns:
        A tuple corresponding to the converted coordinates, point_l.
    """
    x_0, y_0 = point_0
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])
  
    x_l = x_0 * size_x_l / size_x_0
    y_l = y_0 * size_y_l / size_y_0
    if integer:
        point_l = (round(x_l), round(y_l))
    else:
        point_l = (x_l, y_l)
    return point_l

def check_patch(slide, mask, coord_grid_0, mask_level, 
                patch_size, analyse_level,
                mask_tolerance=0.5,
                margin=0):
    """
    Filters a list of possible coordinates with a set of filtering parameters.

    Args:
        slide : String or open_slide object. The slide from which you wish to sample.
        mask : Binary numpy array, where positive pixels correspond to tissue area and
               negative pixels to background areas in the slide.
        coord_grid_0 : List of list of two elements where each (sub) list can be described as 
                       possible coordinates for a possible tile at analyse_level.
        mask_level : Integer or None. Level to which apply mask_function to the rgb 
                     image of the slide at that resolution. mask_function(slide[mask_level])
                     will return the binary image corresponding to the tissue.
        patch_size : Tuple of integers or None. If none the default tile size will (512 + margin, 512 + margin).
        analyse_level : Integer. Level resolution to use for extracting the tiles.
        list_func : None or list of functions to apply to the tiles. Useful to filter the tiles
                    that are part of the tissue mask. Very useful if the tissue mask is bad and 
                    samples many white background tiles, in this case it is interesting to add 
                    a function to eliminate tiles that are too white, like the function white_percentage.
        mask_tolerance : Float between 0 and 1. A tile will be accepted if pixel_in_mask / total_pixel > value.
                         So if mask_tolerance = 1, only tiles that are completly within the mask are accepted.
        margin : Integer. By default set to 0, number of pixels at resolution 0 to add
                 to patch_size on each side. (different to overlapping as this is at resolution 0)
    Returns:
        List of parameters where each parameter is a list of 5 elements
        [x, y, size_x_level, size_y_level, level]
    """
    slide_png = np.array(slide.read_region((0,0), mask_level, slide.level_dimensions[mask_level]))[:,:,:3]
    assert slide_png.shape[0:2] == mask.shape[0:2], "Raise value, mask not of the right shape {}".format(mask.shape[0:2])
    shape_mask = np.array(mask.shape[0:2])
    parameters = []
    patch_size_l = get_size(slide, patch_size, analyse_level, mask_level)
    radius = np.array([max(el // 2, 1) for el in patch_size_l])
    for coord_0 in coord_grid_0:
        coord_l = get_x_y_from_0(slide, coord_0, mask_level)
        point_cent_l = [coord_l + radius, shape_mask - 1 - radius] 
        point_cent_l = np.array(point_cent_l).min(axis=0)
        if mask_percentage(mask, point_cent_l, radius, mask_tolerance): 
            still_add = True
            if ((coord_l + radius) != point_cent_l).any():
                still_add = False
            if still_add:
                sub_param = [coord_0[1] - margin, coord_0[0] - margin, \
                             patch_size[0] + 2 * margin, patch_size[1] + 2 * margin, \
                             analyse_level]
                parameters.append(sub_param)
    return parameters

def mask_percentage(mask, point, radius, mask_tolerance=0.5):
    """
    Given a binary image and a point and a radius -> sub_img
    Computes a score to know how much of the  sub_img is covered
    by tissue region. Given a tolerance threshold this will return
    a boolean.
    tolerance is mask_tolerance
    tolerance of 1 means that the entire image is in the mask area.
    tolerance of 0.1 means that the image has to overlap at least at 10%
              with the mask.
    Args:
        mask : Binary numpy array, where positive pixels correspond to tissue area and
               negative pixels to background areas in the slide.
        point : A tuple like object of size 2 
                with integers.
        radius : None (default) or a tuple, or tuple like 
                 object of size 2 with integers.
        tolerance : A float between 0 and 1. By default 0.5.
    Returns:
        A boolean. If True, keep, else discard
    """
    sub_mask = pj_slice(mask, point - radius, point + radius + 1)
    score = sub_mask.sum() / (sub_mask.shape[0] * sub_mask.shape[1])
    accepted = score > mask_tolerance
    return accepted

def pj_slice(array_np, point_0, point_1=None):
    """
    Allows to slice numpy array's given one point or 
    two points.
    Args:
        array_np : Numpy array to slice
        point_0 : A tuple, or tuple like object of size 2 
                  with integers.
        point_1 : None (default) or a tuple, or tuple like 
                  object of size 2 with integers.
    Returns:
        If point_1 is None, returns array_np evaluated in point_0,
        else returns a slice of array_np between point_0 and point_1.
    """
    x_0, y_0 = check_borders_correct(array_np, point_0)
    if point_1 is None:
        result = array_np[x_0, y_0]
    else:
        x_1, y_1 = check_borders_correct(array_np, point_1)
        if x_0 > x_1:
            warnings.warn("Invalid x_axis slicing, \
                point_0: {} and point_1: {}".format(point_0, point_1))
        if y_0 > y_1:
            warnings.warn("Invalid y_axis slicing, \
                point_0: {} and point_1: {}".format(point_0, point_1))
        result = array_np[x_0:x_1, y_0:y_1]
    return result

def check_borders_correct(array_np, point):
    shape = array_np.shape
    if point[0] < 0 or point[1] < 0 or point[0] > shape[0] or point[1] > shape[1]:
        x, y = point
        x = max(0, x)
        y = max(0, y)
        x = min(shape[0], x)
        y = min(shape[1], y)
        warnings.warn("Invalid point: {}, corrected to {}".format(point, (x, y)))
        point = (x, y)
    return point

def patch_sampling(slide, seed=None, mask_level=None,
                   mask_function=None, analyse_level=0, patch_size=None, overlapping=0,
                   list_func=None, mask_tolerance=0.5, 
                   n_samples=10, with_replacement=False):
    """
    Code @PeterNaylor from useful_wsi.
    Returns a list of tiles from slide given a mask generating method
    and a sampling method
    Args:
        slide : String or open_slide object. The slide from which you wish to sample.
        seed : Integer or None. Seed value to use for setting numpy randomness.
        mask_level : Integer or None. Level to which apply mask_function to the rgb 
                     image of the slide at that resolution. mask_function(slide[mask_level])
                     will return the binary image corresponding to the tissue.
        mask_function : Function that returns a binary image of same size as input. 
                        Mask_function is applied in order to determine the tissue areas on 
                        the slide.
        analyse_level : Integer. Level resolution to use for extracting the tiles.
        patch_size : Tuple of integers or None. If none the default tile size will (512, 512).
        overlapping : Integer. By default set to 0, number of pixels at analyse level to add
                      to patch_size on each side.
        list_func : None or list of functions to apply to the tiles. Useful to filter the tiles
                    that are part of the tissue mask. Very useful if the tissue mask is bad and 
                    samples many white background tiles, in this case it is interesting to add 
                    a function to eliminate tiles that are too white, like the function white_percentage.
        mask_tolerance : Float between 0 and 1. A tile will be accepted if pixel_in_mask / total_pixel > value.
                         So if mask_tolerance = 1, only tiles that are completly within the mask are accepted.
        n_samples : Integer, default to 10, number of tiles to extract from the slide with the 
                    sampling method "random_sampling".
        with_replacement : Bool, default to False. Wether or not you can sample with replacement in the case
                           of random sampling.

    Returns:
        List of parameters where each parameter is a list of 5 elements
        [x, y, size_x_level, size_y_level, level]
    """
    np.random.seed(seed)
    slide = openslide.open_slide(slide) if isinstance(slide, str) else slide
    if patch_size is None:
        patch_size = (256, 256)
    if list_func is None:
        list_func = list()
    if mask_level is None:
        mask_level = slide.level_count - 1
    wsi_mask = mask_function(slide)
    min_row, min_col, max_row, max_col = 0, 0, *wsi_mask.shape
    point_start_l = min_row, min_col
    point_end_l = max_row, max_col
    point_start_0 = get_x_y(slide, point_start_l, mask_level)
    point_end_0 = get_x_y(slide, point_end_l, mask_level)
    grid_coord = grid_blob(slide, point_start_0, point_end_0, patch_size,
                           analyse_level)

    margin_mask_level = get_size(slide, (overlapping, 0),
                                 0, analyse_level)[0]
    parameter = check_patch(slide, wsi_mask, grid_coord,
                            mask_level, patch_size, analyse_level,
                            mask_tolerance=mask_tolerance,
                            margin=margin_mask_level)
    return_list = parameter
    return {'params':return_list,'mask':wsi_mask}
