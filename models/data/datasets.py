import torch
from torch.utils.data import Dataset

import os
from os.path import *
import numpy as np

from glob import glob
from .utils import frame_utils

from imageio import imread

class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, range_=(0, 1), old_range=(0, 255)):
        """
        :param range_: Value range to which images should be rescaled
        :param old_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = range_[0]
        self.max = range_[1]
        self._data_min = old_range[0]
        self._data_max = old_range[1]

    def __call__(self, images):

        images = images - self._data_min  # normalize to (0, data_max-data_min)
        images /= (self._data_max - self._data_min)  # normalize to (0, 1)
        images *= (self.max - self.min)  # norm to (0, target_max-target_min)
        images += self.min  # normalize to (target_min, target_max)
        
        return images
    
class StaticRandomCrop:
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop:
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]
    
class MpiSintel(Dataset):
    def __init__(self, is_cropped=False, is_rescaled=False, crop_size=None, root='', dstype='clean', replicates=1):
        self.is_cropped = is_cropped
        self.is_rescaled = is_rescaled
        if self.is_cropped:
            if not crop_size:
                raise ValueError('crop_size should be given.')
            self.crop_size = crop_size
        self.replicates = replicates
        
        flow_root = join(root, 'flow')
        image_root = join(root, dstype)
        
        file_list = sorted(glob(join(flow_root, '*/*.flo')))
        
        self.flow_list = []
        self.image_list = []
        
        for file in file_list:
            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])
            
            img1 = join(image_root, fprefix + '{:04d}'.format(fnum+0) + '.png')
            img2 = join(image_root, fprefix + '{:04d}'.format(fnum+1) + '.png')
            
            assert isfile(img1), 'Cannot find file: {}'.format(img1)
            assert isfile(img2), 'Cannot find file: {}'.format(img2)
            assert isfile(file), 'Cannot find file: {}'.format(file)
            
            self.image_list.append([img1, img2])
            self.flow_list.append(file)
        assert (len(self.image_list) == len(self.flow_list)), 'number of image pairs shoule be equal to number of flows'
        
        self.size = len(self.image_list)
        self.render_size = list(frame_utils.read_gen(self.image_list[0][0]).shape[:2])
        
        if (self.render_size[0] % 64) or (self.render_size[1] % 64):
            self.render_size[0] = ((self.render_size[0]) // 64) * 64
            self.render_size[1] = ((self.render_size[1]) // 64) * 64


    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]
        else:
            index = index % self.size

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            flow = frame_utils.read_gen(self.flow_list[index])

            images = [img1, img2]
            image_size = img1.shape[:2]

            if self.is_cropped:
                cropper = StaticRandomCrop(image_size, self.crop_size)
            else:
                cropper = StaticCenterCrop(image_size, self.render_size)

            images = list(map(cropper, images))
            flow = cropper(flow)

            # rescale image range from [0, 255] to [0, 1]
            if self.is_rescaled:
                rescale = RescaleTransform()
                images = list(map(rescale, images))

            images = np.array(images).transpose(3, 0, 1, 2)
            flow = flow.transpose(2, 0, 1)

            images = torch.from_numpy(images.astype(np.float32))
            flow = torch.from_numpy(flow.astype(np.float32))

            return images, flow
    
    def __len__(self):
        return self.size + self.replicates
    
class MpiSintelClean(MpiSintel):
    def __init__(self, is_cropped=False, is_rescaled=False, crop_size=None, root='', replicates=1):
        super().__init__(is_cropped=is_cropped, is_rescaled=is_rescaled, crop_size=crop_size, root=root, dstype='clean', replicates=replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, is_cropped=False, is_rescaled=False, crop_size=None, root='', replicates=1):
        super().__init__(is_cropped=is_cropped, is_rescaled=is_rescaled, crop_size=crop_size, root=root, dstype='final', replicates=replicates)
    