import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
from os.path import *
import numpy as np

import cv2

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
        self.h1 = np.random.randint(0, h - self.th)
        self.w1 = np.random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop:
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]
    
class StaticRandomOcclusion:
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = np.random.randint(0, h - self.th)
        self.w1 = np.random.randint(0, w - self.tw)
        
    def __call__(self, img):
        occlusion_map = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)
        occlusion_map[:, self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw)] = 1.0 # 1: occluded pixels
        img[:, self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw)] = 0.0
        return img, occlusion_map
    
class FreeFormRandomOcclusion:
    def __init__(self, max_vertices=4, max_brush_width=3, max_len=30, max_angle=np.pi):
        self.max_v = max_vertices
        self.mbw = max_brush_width
        self.mlen = max_len
        self.mangle = max_angle
        
    def __call__(self, img):
        occlusion_map = np.zeros(shape=(img.shape[1], img.shape[2]))
        num_v = 6 + np.random.randint(self.max_v)

        for i in range(num_v):
            start_x = np.random.randint(img.shape[2])
            start_y = np.random.randint(img.shape[1])
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(self.mangle)
                if i % 2 == 0:
                    angle = 2 * np.pi - angle
                length = 10 + np.random.randint(self.mlen)
                brush_w = 5 + np.random.randint(self.mbw)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(occlusion_map, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
                
        occlusion_map = occlusion_map.reshape((1,)+occlusion_map.shape).astype(np.float32)
        occlusion_map = torch.from_numpy(occlusion_map)
        img = torch.where(occlusion_map == 0.0, img, torch.zeros_like(img))
        return img, occlusion_map
    
class MpiSintel(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root='', dstype='clean', replicates=1, image_size=None, stack_imgs=True):
        self.transform  = transform
        self.replicates = replicates
        self.image_size = image_size
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
            
            
            return images, flow
    
    def __len__(self):
        return self.size * self.replicates
    
class MpiSintelClean(MpiSintel):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size =None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='clean', replicates=replicates, image_size = image_size, stack_imgs=stack_imgs)

class MpiSintelFinal(MpiSintel):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size = None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='final', replicates=replicates, image_size = image_size, stack_imgs= stack_imgs)

class MpiSintelOcc(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root='', dstype='clean', replicates=1, image_size=None, stack_imgs=True):
        self.transform = transform
        self.replicates = replicates
        self.image_size =image_size
        self.stack_imgs = stack_imgs
        
        occ_root = join(root, 'occlusions')
        image_root = join(root, dstype)
        
        file_list = sorted(glob(join(occ_root, '*/*.png')))
        self.occ_list = []
        self.image_list = []
        
        for file in file_list:
            fbase = file[len(occ_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])
            
            img1 = join(image_root, fprefix + '{:04d}'.format(fnum+0) + '.png')
            img2 = join(image_root, fprefix + '{:04d}'.format(fnum+1) + '.png')
            
            assert isfile(img1), 'Cannot find file: {}'.format(img1)
            assert isfile(img2), 'Cannot find file: {}'.format(img2)
            assert isfile(file), 'Cannot find file: {}'.format(file)
            
            self.image_list.append([img1, img2])
            self.occ_list.append(file)
        assert (len(self.image_list) == len(self.occ_list)), 'number of image pairs shoule be equal to number of occlusions'
        
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

            occ = frame_utils.read_gen(self.occ_list[index]).astype(np.float32)
            occ = cropper(occ)
            toTensor = transforms.ToTensor()
            occ = toTensor(occ)
            if self.image_size: 
                resize = transforms.Resize(self.image_size)
                occ = resize(occ)
            occ[occ > 0.5] = 1.0
            occ[occ != 1.0] = 0.0

            return images, occ
    
    def __len__(self):
        return self.size * self.replicates   
    
class MpiSintelCleanOcc(MpiSintelOcc):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size=None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='clean', replicates=replicates, image_size=image_size, stack_imgs=stack_imgs)

class MpiSintelFinalOcc(MpiSintelOcc):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size=None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='final', replicates=replicates, image_size=image_size, stack_imgs= stack_imgs)

class MpiSintelFlowOcc(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root='', dstype='clean', replicates=1, image_size=None, stack_imgs=True):
        self.transform  = transform
        self.replicates = replicates
        self.image_size = image_size
        self.stack_imgs = stack_imgs
        
        flow_root  = join(root, 'flow')
        occ_root   = join(root, 'occlusions')
        image_root = join(root, dstype)
        
        flow_file_list = sorted(glob(join(flow_root, '*/*.flo')))
        occ_file_list = sorted(glob(join(occ_root, '*/*.png')))
        assert len(occ_file_list) == len(flow_file_list), 'number of occlusion maps should be equal to number of flows'
        self.flow_list  = []
        self.occ_list   = []
        self.image_list = []
        
        for flow_file, occ_file in zip(flow_file_list, occ_file_list):
            fbase = flow_file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])
            
            img1 = join(image_root, fprefix + '{:04d}'.format(fnum+0) + '.png')
            img2 = join(image_root, fprefix + '{:04d}'.format(fnum+1) + '.png')
            
            assert isfile(img1), 'Cannot find file: {}'.format(img1)
            assert isfile(img2), 'Cannot find file: {}'.format(img2)
            assert isfile(flow_file), 'Cannot find file: {}'.format(flow_file)
            assert isfile(occ_file), 'Cannot find file: {}'.format(occ_file)
            
            self.image_list.append([img1, img2])
            self.flow_list.append(flow_file)
            self.occ_list.append(occ_file)
            
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
            # read image pair
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

            # read optical flow
            flow = frame_utils.read_gen(self.flow_list[index]).astype(np.float32)
            flow = cropper(flow)
            if self.image_size: 
                flow = resize_flow(flow, self.image_size[0], self.image_size[1])
            flow = flow.transpose(2,0,1)
            flow = torch.from_numpy(flow)
            
            occ = frame_utils.read_gen(self.occ_list[index]).astype(np.float32)
            occ = cropper(occ)
            toTensor = transforms.ToTensor()
            occ = toTensor(occ)
            if self.image_size: 
                resize = transforms.Resize(self.image_size)
                occ = resize(occ)
            occ[occ > 0.5] = 1.0
            occ[occ != 1.0] = 0.0
            return images, flow, occ
    
    def __len__(self):
        return self.size * self.replicates

class MpiSintelCleanFlowOcc(MpiSintelFlowOcc):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size=None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='clean', replicates=replicates, image_size=image_size, stack_imgs=stack_imgs)

class MpiSintelFinalFlowOcc(MpiSintelFlowOcc):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size=None, stack_imgs=True):
        super().__init__(transform=transform, root=root, dstype='final', replicates=replicates, image_size=image_size, stack_imgs= stack_imgs)
                
class MpiSintelInpainting(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root='', dstype='clean', replicates=1, image_size=None, occlusion_ratio=0.5, static_occ=True):
        self.transform = transform
        self.replicates = replicates
        self.image_size =image_size
        self.occlusion_ratio = occlusion_ratio
        self.static_occ = static_occ
        image_root = join(root, dstype)
        
        self.image_list = sorted(glob(join(image_root, '*/*.png')))
        
        self.size = len(self.image_list)

        self.render_size = list(frame_utils.read_gen(self.image_list[0]).shape[:2])
        
        if (self.render_size[0] % 64) or (self.render_size[1] % 64):
            self.render_size[0] = ((self.render_size[0]) // 64) * 64
            self.render_size[1] = ((self.render_size[1]) // 64) * 64


    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]
        else:
            index = index % self.size

            img = frame_utils.read_gen(self.image_list[index])

            
            h, w = img.shape[:2]
            cropper = StaticCenterCrop((h, w), self.render_size)
            img = cropper(img)
            
            complete_img = img.copy()
            
            
            if self.transform:
                img = self.transform(img)
                complete_img = self.transform(complete_img)
            
            if self.image_size:
                resize = transforms.Resize(self.image_size)
                img = resize(img)
                complete_img = resize(complete_img)
            
            if self.static_occ:
                h, w = img.shape[1], img.shape[2]
                th, tw = int(self.occlusion_ratio * h), int(self.occlusion_ratio * w)
                occ = StaticRandomOcclusion((h, w), (th, tw))
            else:
                h, w = img.shape[1], img.shape[2]
                max_brush_width = int(0.05 * h)
                max_len = int(0.5 * h)
                occ = FreeFormRandomOcclusion(max_brush_width=max_brush_width, max_len=max_len)
            img, occlusion_map = occ(img)
            
            return img, complete_img, occlusion_map
    
    def __len__(self):
        return self.size * self.replicates
    
class MpiSintelCleanInpainting(MpiSintelInpainting):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size=None, occlusion_ratio=0.5, static_occ=False):
        super().__init__(transform=transform, root=root, dstype='clean', replicates=replicates, image_size=image_size, occlusion_ratio=occlusion_ratio, static_occ=static_occ)

class MpiSintelFinalInpainting(MpiSintelInpainting):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size=None, occlusion_ratio=0.5, static_occ=False):
        super().__init__(transform=transform, root=root, dstype='clean', replicates=replicates, image_size=image_size, occlusion_ratio=occlusion_ratio, static_occ=static_occ)

class FlyingChairs(Dataset):
    def __init__(self, transform=transforms.ToTensor(), root='', replicates=1, image_size =None, stack_imgs=True):
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
        self.occ_list = sorted(glob(join(root, 'occlusion', '*.' + iext)))
        assert len(first_images) == len(second_images) and len(first_images) == len(self.flow_list) and len(first_images) == len(self.occ_list), 'The number of image pairs should be equal to the number of optical flows and occlusion maps'
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
            # read image pair
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

            # read optical flow
            flow = frame_utils.read_gen(self.flow_list[index]).astype(np.float32)
            flow = cropper(flow)
            if self.image_size: 
                flow = resize_flow(flow, self.image_size[0], self.image_size[1])
            flow = flow.transpose(2,0,1)
            flow = torch.from_numpy(flow)
            
            occ = frame_utils.read_gen(self.occ_list[index]).astype(np.float32)
            occ = cropper(occ)
            toTensor = transforms.ToTensor()
            occ = toTensor(occ)
            if self.image_size: 
                resize = transforms.Resize(self.image_size)
                occ = resize(occ)
            occ[occ > 0.5] = 1.0
            occ[occ != 1.0] = 0.0
            return images, flow, occ
        
    def __len__(self):
        return self.size * self.replicates