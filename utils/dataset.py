import math, warnings
import glob
import os
from os import listdir
from os.path import isfile, join

import cv2
import skimage.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from tqdm import tqdm

import staintools, spams

def read_image(image_path, resize_ratio=1):

    if not(isinstance(image_path, str)):
        # if tensor with byte string
        image_path = image_path.numpy().decode('utf-8')

    image_level_1 = skimage.io.MultiImage(
        image_path)[1]  # Modify to use higher
    # resolution images

    if resize_ratio != 1:
        new_w = int(image_level_1.shape[1] * resize_ratio)
        new_h = int(image_level_1.shape[0] * resize_ratio)
        image_level_1 = cv2.resize(
            image_level_1, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image_level_1


def _mask_tissue(image, kernel_size=(7, 7), gray_threshold=220):
    """Masks tissue in image. Uses gray-scaled image, as well as
    dilation kernels and 'gap filling'
    """
    # Define elliptic kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Convert rgb to gray scale for easier masking
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Now mask the gray-scaled image (capturing tissue in biopsy)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    # Use dilation and findContours to fill in gaps/holes in masked tissue
    mask = cv2.dilate(mask, kernel, iterations=1)
    contour = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        try:
            cv2.drawContours(mask, [cnt], 0, 1, -1)
        except:
            a = 0
    return mask


def _pad_image(image, pad_len, pad_val):
    """Pads inputted image, accepts both
    2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
    elif image.ndim == 3:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
    return None


def _transpose_image(image):
    """Inputs an image and transposes it, accepts
    both 2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.transpose(image, (1, 0)).copy()
    elif image.ndim == 3:
        return np.transpose(image, (1, 0, 2)).copy()
    return None


def _get_tissue_parts_indices(tissue, min_consec_info):
    """If there are multiple tissue parts in 'tissue', 'tissue' will be
    split. Each tissue part will be taken care of separately (later on),
    and if the tissue part is less than min_consec_info, it's considered
    to small and won't be returned.
    """
    split_points = np.where(np.diff(tissue) != 1)[0] + 1
    tissue_parts = np.split(tissue, split_points)
    return [
        tp for tp in tissue_parts if len(tp) >= min_consec_info
    ]


def _get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
    This tissue part is divided into N subparts and returned.
    Argument min_decimal_keep basically decides if we should reduce the
    N subparts to N-1 subparts, due to overflow.
    """
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end - start) / patch_size
    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts * patch_size) - (end - start)
    shift = excess // 2

    return [
        i * patch_size + start - shift
        for i in range(num_subparts)
    ]


def _eval_and_append_xy_coords(coords,
                               image,
                               mask,
                               patch_size,
                               x, y,
                               min_patch_info,
                               transposed,
                               precompute):
    """Based on computed x and y coordinates of patch:
    slices out patch from original image, flattens it,
    preprocesses it, and finally evaluates its mask.
    If patch contains more info than min_patch_info,
    the patch coordinates are kept, along with a value
    'val1' that estimates how much information there
    is in the patch. Smaller 'val1' assumes more info.
    """
    patch_1d = (
        image[y: y + patch_size, x:x + patch_size, :]
        .mean(axis=2)
        .reshape(-1)
    )
    idx_tissue = np.where(patch_1d <= 210)[0]
    idx_black = np.where(patch_1d < 5)[0]
    idx_background = np.where(patch_1d > 210)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = 210
        patch_1d[idx_background] = 210
        val1 = int(patch_1d.mean())
        val2 = mask[y:y + patch_size, x:x + patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([
                        coords, [[val1, x - patch_size, y - patch_size]]
                    ])
                else:
                    coords = np.concatenate([
                        coords, [[val1, y - patch_size, x - patch_size]]
                    ])
            else:
                coords = np.concatenate([
                    coords, [[val1, y, x]]
                ])

    return coords


def compute_coords(image,
                   patch_size=256,
                   precompute=False,
                   min_patch_info=0.35,
                   min_axis_info=0.35,
                   min_consec_axis_info=0.35,
                   min_decimal_keep=0.7):
    """
    Input:
        image : 3-d np.ndarray
        patch_size : size of patches/tiles, will be of
            size (patch_size x patch_size x 3)
        precompute : If True, only coordinates will be returned,
            these coordinates match the inputted 'original' image.
            If False, both an image and coordinates will be returned,
            the coordinates does not match the inputted image but the
            image that it is returned with.
        min_patch_info : Minimum required information in patch
            (see '_eval_and_append_xy_coords')
        min_axis_info : Minimum fraction of on-bits in x/y dimension to be
            considered enough information. For x, this would be fraction of
            on-bits in x-dimension of a y:y+patch_size slice. For y, this would
            be the fraction of on-bits for the whole image in y-dimension
        min_consec_axis_info : Minimum consecutive x/y on-bits
            (see '_get_tissue_parts_indices')
        min_decimal_keep : Threshold for decimal point for removing "excessive" patch
            (see '_get_tissue_subparts_coords')

    Output:
        image [only if precompute is False] : similar to input image, but fits
            to the computed coordinates
        coords : the coordinates that will be used to compute the patches later on
    """

    if not isinstance(image, np.ndarray):
        # if image is a Tensor
        image = image.numpy()

    # masked tissue will be used to compute the coordinates
    mask = _mask_tissue(image)

    # initialize coordinate accumulator
    coords = np.zeros([0, 3], dtype=int)

    # pad image and mask to make sure no tissue is potentially missed out
    image = _pad_image(image, patch_size, 'maximum')
    mask = _pad_image(mask, patch_size, 'minimum')

    y_sum = mask.sum(axis=1)
    x_sum = mask.sum(axis=0)
    # if on bits in x_sum is greater than in y_sum, the tissue is
    # likely aligned horizontally. The algorithm works better if
    # the image is aligned vertically, thus the image will be transposed
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        image = _transpose_image(image)
        mask = _transpose_image(mask)
        y_sum, _ = x_sum, y_sum
        transposed = True
    else:
        transposed = False

    # where y_sum is more than the minimum number of on-bits
    y_tissue = np.where(y_sum >= (patch_size * min_axis_info))[0]

    if len(y_tissue) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute:
            return [(0, 0, 0)]
        else:
            return image, [(0, 0, 0)]

    y_tissue_parts_indices = _get_tissue_parts_indices(
        y_tissue, patch_size * min_consec_axis_info)

    if len(y_tissue_parts_indices) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute:
            return [(0, 0, 0)]
        else:
            return image, [(0, 0, 0)]

    # loop over the tissues in y-dimension
    for yidx in y_tissue_parts_indices:
        y_tissue_subparts_coords = _get_tissue_subparts_coords(
            yidx, patch_size, min_decimal_keep)

        for y in y_tissue_subparts_coords:
            # in y_slice, where x_slice_sum is more than the minimum number of
            # on-bits
            x_slice_sum = mask[y:y + patch_size, :].sum(axis=0)
            x_tissue = np.where(x_slice_sum >= (patch_size * min_axis_info))[0]

            x_tissue_parts_indices = _get_tissue_parts_indices(
                x_tissue, patch_size * min_consec_axis_info)

            # loop over tissues in x-dimension (inside y_slice
            # 'y:y+patch_size')
            for xidx in x_tissue_parts_indices:
                x_tissue_subparts_coords = _get_tissue_subparts_coords(
                    xidx, patch_size, min_decimal_keep)

                for x in x_tissue_subparts_coords:
                    coords = _eval_and_append_xy_coords(
                        coords, image, mask, patch_size, x, y,
                        min_patch_info, transposed, precompute
                    )

    if len(coords) < 1:
        warnings.warn("Not enough tissue in image (x-dim)", RuntimeWarning)
        if precompute:
            return [(0, 0, 0)]
        else:
            return image, [(0, 0, 0)]

    if precompute:
        return coords
    else:
        return image, coords


def get_normalizer(reference):
    """Return stain normalizer given reference image"""
    # Read data
    target = staintools.read_image(reference)

    # Standardize brightness (This step is optional but can improve the tissue mask calculation)
    target = staintools.LuminosityStandardizer.standardize(target)

    # Stain normalize
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target)

    return normalizer


def convert_RGB_to_OD(I):
    mask = (I == 0)
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)


def get_concentrations(I, stain_matrix, regularizer=0.01):
    """
    Estimate concentration matrix given an image and stain matrix.
    :param I:
    :param stain_matrix:
    :param regularizer:
    :return:
    """
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T


def he_stain_pipeline(image, normalizer):
    image_norm = normalizer.transform(staintools.LuminosityStandardizer.standardize(image))
    
    stain_matrix = staintools.MacenkoStainExtractor.get_stain_matrix(image_norm)
    
    image_con = get_concentrations(
                                image_norm,
                                stain_matrix,
                                regularizer=0.01
                                ).reshape(
                                    (image.shape[0],
                                    image.shape[1],
                                    2))
    return image_con, image_norm


if __name__ == '__main__':

    input_path = '/work/07457/rochan_a/frontera/data/org/train_images/'
    mask_path = '/work/07457/rochan_a/frontera/data/org/train_label_masks/'
    train_list = pd.read_csv('/work/07457/rochan_a/frontera/data/org/train.csv')
    output_path = '/home1/07457/rochan_a/mvrl-wsi-pathology/experiments/utils/'

    patch_size = 1000

    onlyfiles = [f.split('_')[0]
                 for f in listdir(mask_path) if isfile(join(mask_path, f))]
    print(onlyfiles)

    unique = pd.unique(train_list['gleason_score'])
    for gl_sc in unique:
        p = train_list[train_list['gleason_score'] == gl_sc].sample(n=20)#(n=10)

        if not os.path.exists(output_path + str(gl_sc)):
            os.makedirs(output_path + str(gl_sc))

        for idx, val in tqdm(p.iterrows()):
            f_name = val['image_id']
            if f_name not in onlyfiles:
                print('Mask Missing!')
                break

            image_ = input_path + f_name + '.tiff'
            image = read_image(image_, 1)

            mask_ = mask_path + f_name + '_mask.tiff'
            mask = tiff.imread(mask_)

            mask = cv2.resize(
                mask,
                (image.shape[1],
                 image.shape[0]),
                interpolation=cv2.INTER_AREA)

            coords = compute_coords(image,
                                    patch_size=patch_size,
                                    precompute=True,
                                    min_patch_info=0.2,
                                    min_axis_info=0.2,
                                    min_consec_axis_info=0.2,
                                    min_decimal_keep=0.7)

            # sort coords (high info -> low info)
            coords = sorted(coords, key=lambda x: x[0], reverse=False)
            for iidx, (v, y, x) in enumerate(coords):
                x = 0 if x < 0 else x
                y = 0 if y < 0 else y
                start_point = (x, y)
                X = image.shape[1] if x + \
                    patch_size > image.shape[1] else x + patch_size
                Y = image.shape[0] if y + \
                    patch_size > image.shape[0] else y + patch_size
                end_point = (X, Y)
                try:
                    cv2.imwrite(output_path + str(gl_sc) + '/' + f_name + '_p_' + str(iidx) + '.tiff',
                                image[start_point[1]:end_point[1], start_point[0]:end_point[0]])
                    cv2.imwrite(output_path + str(gl_sc) + '/' + f_name + '_mask_p_' + str(iidx) + '.tiff',
                                mask[start_point[1]:end_point[1], start_point[0]:end_point[0]])
                except BaseException:
                    print(start_point, end_point)

    for folder in unique:
        mask_path = '{}/'.format(folder)

        files = []
        for f in listdir(mask_path):
            if isfile(join(mask_path, f)) and len(f.split('_')) == 3:
                files.append(f.split('.')[0])

        train = files[:int(len(files) * 0.8)]
        test = files[int(len(files) * 0.8):]

        with open(join(mask_path, 'train.txt'), 'w') as f:
            for item in train:
                f.write("%s\n" % item)

        with open(join(mask_path, 'test.txt'), 'w') as f:
            for item in test:
                f.write("%s\n" % item)
