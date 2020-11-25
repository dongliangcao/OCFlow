import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
from os.path import *
import numpy as np

from glob import glob
from .utils import frame_utils

from imageio import imread
from models.data.utils.flow_utils import resize_flow

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
    def __init__(self, transform=transforms.ToTensor(), root='', dstype='clean', replicates=1, image_size = 'None', stack_imgs=True):
#         self.is_cropped = is_cropped
#         self.is_rescaled = is_rescaled
#         if self.is_cropped:
#             if not crop_size:
#                 raise ValueError('crop_size should be given.')
#             self.crop_size = crop_size
        self.transform = transform
        self.replicates = replicates
        self.image_size =image_size
        self.stack_imgs = stack_imgs
        
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
        #print(self.image_list)

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

            image_size = img1.shape[:2]

            cropper = StaticCenterCrop(image_size, self.render_size)
            img1 = cropper(img1)
            img2 = cropper(img2)
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            if self.image_size:
                resize = transforms.Resize(self.image_size)
                img1 = resize(img1)
                img2 = resize(img2)
            if self.stack_imgs:
                images = torch.stack((img1, img2))
            else:
                images = torch.cat((img1, img2))

            flow = frame_utils.read_gen(self.flow_list[index]).astype(np.float32)
            flow = cropper(flow)
            if self.image_size: 
                flow = resize_flow(flow, self.image_size[0], self.image_size[1])
            flow = flow.transpose(2,0,1)
            flow = torch.from_numpy(flow)
            
            
            return images, flow #size [2,C,H,W]
    
    def __len__(self):
        return self.size * self.replicates
    
class MpiSintelClean(MpiSintel):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size =None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='clean', replicates=replicates, image_size = image_size, stack_imgs=stack_imgs)

class MpiSintelFinal(MpiSintel):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size = None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='final', replicates=replicates, image_size = image_size, stack_imgs= stack_imgs)
    
class FlyingChairs(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size =None, stack_imgs=True):
#         self.is_cropped = is_cropped
#         self.is_rescaled = is_rescaled
#         if self.is_cropped:
#             if not crop_size:
#                 raise ValueError('crop_size should be given.')
#             self.crop_size = crop_size
        self.transform = transform
        self.replicates = replicates

        images = sorted( glob( join(root, '*.ppm') ) )

        self.flow_list = sorted( glob( join(root, '*.flo') ) )

        assert (len(images)//2 == len(self.flow_list)) 
        
        self.image_list = []
        for i in range(len(self.flow_list)):
            im1 = images[2*i]
            im2 = images[2*i + 1]
            self.image_list += [ [ im1, im2 ] ]

        assert len(self.image_list) == len(self.flow_list), 'The number of image pairs should be equal to the number of optical flow'

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

            image_size = img1.shape[:2]

            cropper = StaticCenterCrop(image_size, self.render_size)
            img1 = cropper(img1)
            img2 = cropper(img2)
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            if self.image_size:
                resize = transforms.Resize(self.image_size)
                img1 = resize(img1)
                img2 = resize(img2)
            if self.stack_imgs:
                images = torch.stack((img1, img2))
            else:
                images = torch.cat((img1, img2))

            flow = frame_utils.read_gen(self.flow_list[index]).astype(np.float32)
            flow = cropper(flow)
            if self.image_size: 
                flow = resize_flow(flow, self.image_size[0], self.image_size[1])
            flow = flow.transpose(2,0,1)
            flow = torch.from_numpy(flow)

            return images, flow

    def __len__(self):
        return self.size * self.replicates 
    
class ImagesFromFolder(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root='', iext='png', replicates=1, stack_imgs=True, image_size = None):
        self.transform = transform
        self.replicates = replicates
        self.stack_imgs = stack_imgs
        images = sorted(glob(join(root, '*.' + iext)))
        self.image_list = []
        for i in range(len(images)-1):
            im1 = images[i]
            im2 = images[i+1]
            self.image_list.append([ im1, im2 ])

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

            image_size = img1.shape[:2]
            cropper = StaticCenterCrop(image_size, self.render_size)
            img1 = cropper(img1)
            img2 = cropper(img2)
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            if self.image_size:
                resize = transforms.Resize(self.image_size)
                img1 = resize(img1)
                img2 = resize(img2)
            if self.stack_imgs:
                images = torch.stack((img1, img2))
            else:
                images = torch.cat((img1, img2))
            
            return images

    def __len__(self):
        return self.size * self.replicates

class ImgFlowOccFromFolder(Dataset):
    def __init__(self, transform=transforms.ToTensor(), image_size=None, root='', iext='png', replicates=1, stack_imgs=True):
        self.transform = transform
        self.image_size = image_size
        self.replicates = replicates
        self.stack_imgs = stack_imgs
        
        first_images = sorted(glob(join(root, 'img_1', '*.' + iext)))
        second_images = sorted(glob(join(root, 'img_2', '*.' + iext)))
        self.flow_list = sorted(glob(join(root, 'flow', '*.flo')))
        self.occlusion_list = sorted(glob(join(root, 'occlusion', '*.' + iext)))
        assert len(first_images) == len(second_images) and len(first_images) == len(self.flow_list) and len(first_images) == len(self.occlusion_list), 'The number of image pairs should be equal to the number of optical flows and occlusion maps'
        self.image_list = list(zip(first_images, second_images))
        
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

            image_size = img1.shape[:2]

            cropper = StaticCenterCrop(image_size, self.render_size)
            img1 = cropper(img1)
            img2 = cropper(img2)
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            if self.image_size:
                resize = transforms.Resize(self.image_size)
                img1 = resize(img1)
                img2 = resize(img2)
            if self.stack_imgs:
                images = torch.stack((img1, img2))
            else:
                images = torch.cat((img1, img2))
            
            flow = frame_utils.read_gen(self.flow_list[index]).astype(np.float32)
            flow = cropper(flow)
            if self.image_size: 
                flow = resize_flow(flow, self.image_size[0], self.image_size[1])
            flow = flow.transpose(2,0,1)
            flow = torch.from_numpy(flow)

            occlusion = frame_utils.read_gen(self.occlusion_list[index]).astype(np.float32)
            occlusion = cropper(occlusion)
            occlusion = occlusion.transpose(2, 0, 1)
            occlusion = torch.from_numpy(occlusion)
            if self.image_size:
                occlusion = resize(occlusion)
            return images, flow, occlusion
        
    def __len__(self):
        return self.size * self.replicates