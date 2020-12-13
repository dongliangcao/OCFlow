import pytorch_lightning as pl
from torchvision import transforms
from models.data.datasets import ImgFlowOccFromFolder, MpiSintelClean, MpiSintelFinal, MpiSintelCleanOcc, MpiSintelFinalOcc, MpiSintelCleanFlowOcc, MpiSintelFinalFlowOcc, MpiSintelCleanInpainting, MpiSintelFinalInpainting
from torch.utils.data import DataLoader
from math import ceil
import torch

class DatasetModule(pl.LightningDataModule): 
    def __init__(self, root='', image_size=None, batch_size=32, dataset_name='MpiSintelClean', num_workers=6, overfit=False, occlusion_ratio=0.3, static_occ=False):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.overfit = overfit
        self.occlusion_ratio = occlusion_ratio
        self.static_occ = static_occ
    def prepare_data(self):
        self.datasets = dict()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if self.dataset_name == 'ImgFlowOcc': 
            dataset = ImgFlowOccFromFolder(root=self.root, transform=transform, image_size=self.image_size, stack_imgs=False)
        elif self.dataset_name == 'MpiSintelClean': 
            dataset = MpiSintelClean(root=self.root, transform=transform, image_size=self.image_size, stack_imgs=False)
        elif self.dataset_name =='MpiSintelFinal': 
            dataset = MpiSintelFinal(root=self.root, transform=transform, image_size=self.image_size, stack_imgs=False)
        elif self.dataset_name == 'MpiSintelCleanOcc':
            dataset = MpiSintelCleanOcc(root=self.root, transform=transform, image_size=self.image_size, stack_imgs=False)
        elif self.dataset_name == 'MpiSintelFinalOcc':
            dataset = MpiSintelFinalOcc(root=self.root, transform=transform, image_size=self.image_size, stack_imgs=False)
        elif self.dataset_name == 'MpiSintelCleanFlowOcc':
            dataset = MpiSintelCleanFlowOcc(root=self.root, transform=transform, image_size=self.image_size, stack_imgs=False)
        elif self.dataset_name == 'MpiSintelFinalFlowOcc':
            dataset = MpiSintelFinalFlowOcc(root=self.root, transform=transform, image_size=self.image_size, stack_imgs=False)
        elif self.dataset_name == 'MpiSintelCleanInpainting':
            dataset = MpiSintelCleanInpainting(root=self.root, transform=transform, image_size=self.image_size, occlusion_ratio=self.occlusion_ratio, static_occ=self.static_occ)
        elif self.dataset_name == 'MpiSintelFinalInpainting':
            dataset = MpiSintelFinalInpainting(root=self.root, transform=transform, image_size=self.image_size, occlusion_ratio=self.occlusion_ratio, static_occ=self.static_occ)
        else:
            raise ValueError('Unsupported dataset type: {}'.format(self.dataset_name))
        if not self.overfit:
            len_trainset = ceil(0.8 * len(dataset))
            len_valset = ceil(0.1 * len(dataset))
            train_dset, val_dset, test_dset = torch.utils.data.random_split(dataset, [len_trainset, len_valset, len(dataset) - len_trainset - len_valset], generator=torch.Generator().manual_seed(42))
        else:
            train_dset = val_dset = test_dset = dataset


        self.datasets['train'] = train_dset
        self.datasets['val'] = val_dset
        self.datasets['test'] = test_dset
        
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.datasets['val'], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.datasets['test'], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True) 

