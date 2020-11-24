import pytorch_lightning as pl
from torchvision import transforms
from models.data.datasets import ImgFlowOccFromFolder, MpiSintelClean, MpiSintelFinal
from torch.utils.data import DataLoader, random_split
from math import ceil
class DatasetModule(pl.LightningDataModule): 
    def __init__(self, root = '', image_size = (512, 1024), batch_size = 32, dataset_name ='MpiSintelClean'):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
    def prepare_data(self):
        self.datasets = dict()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if self.dataset_name == 'ImgFlowOcc': 
            dataset = ImgFlowOccFromFolder(root=self.root, transform=transform, resize=transforms.Resize(self.image_size), stack_imgs=False)
        elif self.dataset_name == 'MpiSintelClean': 
            dataset = MpiSintelClean(root = self.root, transform = transform, resize = transforms.Resize(self.image_size), stack_imgs=False)
        elif self.dataset_name =='MpiSintelFinal': 
            dataset = MpiSintelFinal(root = self.root, transform = transform, resize = transforms.Resize(self.image_size), stack_imgs=False)
        train_dset, val_dset, test_dset = random_split(dataset, [ceil(len(dataset)*0.8), ceil(len(dataset)*0.1), len(dataset) - ceil(len(dataset)*0.8) - ceil(len(dataset)*0.1)])

        self.datasets['train'] = train_dset
        self.datasets['val'] = val_dset
        self.datasets['test'] = test_dset
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], shuffle=False, batch_size=self.batch_size, num_workers=6)
    
    def val_dataloader(self):
        return DataLoader(self.datasets['val'], shuffle=False, batch_size=self.batch_size, num_workers=6)
    
    def test_dataloader(self):
        return DataLoader(self.datasets['test'], shuffle=False, batch_size=self.batch_size, num_workers=6) 

