from os import listdir
from os.path import isfile, join

import pickle
import cv2
import numpy as np
import scipy.io as sio
import staintools

import torch
from torch.utils import data
import torchvision.transforms as tvt

import tifffile as tiff

def load_data(input_path, image_id, index):
    """Read file at index

    Params
    ------
        input_path: Path to directory
        image_id: ID of image to read
        index: Patch index (Remove later)

    Returns
    -------
        Image
    """
    image_path = join(input_path, image_id + ".tiff")
    image = tiff.imread(image_path)

    return image


def stain_normalizer(ref_image):
    """Create stain normalizer given reference image

    Params
    ------
        ref_image: Path to reference image

    Returns
    -------
        Stain normalizer
    """
    # Read data
    target = staintools.read_image(ref_image)

    # Standardize brightness (This step is optional but can improve the tissue mask calculation)
    target = staintools.LuminosityStandardizer.standardize(target)

    # Stain normalize
    normalizer = staintools.StainNormalizer(method='vahadane')
    return normalizer


def get_file_list(input_path):
    """Get list of images in folder

    Params
    ------
        input_path: Path to directory

    Returns
    -------
        Dictionary with Image ID as key and patch count as value (Remove later)
    """

    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    ids = {}
    for files in onlyfiles:
        if files.split('_')[0] not in ids.keys():
            ids[files.split('_')[0]] = 0
        else:
            ids[files.split('_')[0]] += 1

    return ids

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
