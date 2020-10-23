import numpy as np
from os.path import *
from imageio import imread
from .flow_utils import read_flow

def read_gen(filename: str):
    """
    Read image or optical flow from given filename
    Args:
        filename: file name, str
    Return:
        image or optical flow: np.ndarray
    """
    ext = splitext(filename)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(filename)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(filename)
    elif ext == '.flo':
        return read_flow(filename).astype(np.float32)
    else:
        raise ValueError(f'.{ext} is not supported')
    