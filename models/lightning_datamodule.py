import pytorch_lightning as pl
from torchvision import transforms
from models.data.datasets import ImgFlowOccFromFolder, MpiSintelClean, MpiSintelFinal, MpiSintelCleanOcc, MpiSintelFinalOcc, MpiSintelCleanFlowOcc, MpiSintelFinalFlowOcc
from torch.utils.data import DataLoader
from math import ceil
import torch
torch.manual_seed(0)
class DatasetModule(pl.LightningDataModule): 
    def __init__(self, root='', image_size=None, batch_size=32, dataset_name='MpiSintelClean', num_workers=6):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.num_workers = num_workers
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
        else:
            raise ValueError('Unsupported dataset type: {}'.format(self.dataset_name))
        len_trainset = ceil(0.8 * len(dataset))
        len_valset = ceil(0.1 * len(dataset))
        train_dset, val_dset, test_dset = torch.utils.data.random_split(dataset, [len_trainset, len_valset, len(dataset) - len_trainset - len_valset])


        self.datasets['train'] = train_dset
        self.datasets['val'] = val_dset
        self.datasets['test'] = test_dset
        
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.datasets['val'], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.datasets['test'], shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True) 
