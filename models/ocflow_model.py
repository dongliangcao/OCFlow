import torch
import torch.nn as nn
from models.data.datasets import MpiSintelClean, MpiSintelFinal, FlyingChairs
from models.networks.ocflownet import OCFlowNet
from torch.utils.data import DataLoader
import torch.optim as optim
class OCFlowModel(): 
    def __init__(self, dataset_name = 'MpiSintelClean', root = '', batch_size =1, num_epoch = 10, gammas = [1,1], print_every =1):
        super(OCFlowModel,self).__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size =batch_size
        self.num_epoch = num_epoch
        self.gammas = gammas
        self.print_every = print_every

    def create_dataset(self): 
        if self.dataset_name == 'MpiSintelClean': 
            dataset = MpiSintelClean(root =self.root)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        elif self.dataset_name == 'MpiSintelFinal': 
            dataset = MpiSintelFinal(root =self.root)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        elif self.dataset_name == 'FlyingChairs': 
            dataset = FlyingChairs(root =self.root)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return dataset, dataloader

    def charbonnier_loss(self, diff, mask, q=0.5):
        diff = torch.pow((diff**2+0.001**2), q)
        diff = diff*mask
        diff_sum = torch.sum(diff)
        loss_mean = diff_sum / (torch.sum(mask)+ 1e-6) 
        return loss_mean

    def train(self): 
        self.dataset, self.dataloader = self.create_dataset()
        ocflow_net = OCFlowNet()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        ocflow_net.to(device)
        optimizer = optim.Adam(ocflow_net.parameters(), lr = 0.001)
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.num_epoch): 
                running_loss = 0.0
                for i, (image_batch, flow_batch) in enumerate(self.dataloader): 
                    I1 = image_batch[:, 0, :, :, :].to(device)
                    I2 = image_batch[:, 1, :, :, :].to(device)
                    image_batch = image_batch.to(device)
                    optimizer.zero_grad()

                    print(I1.shape)
                    print(I2.shape)
                    O_s, O_h, Ic1, Iw1 = ocflow_net(image_batch)
                    photometric_loss = self.charbonnier_loss(Iw1-I1, O_s)
                    reconstruction_loss = torch.sum(((I1-Ic1)*(1-O_h))**2)
                    total_loss = self.gammas[0]*photometric_loss + self.gammas[1]*reconstruction_loss
                    total_loss.backward()
                    optimizer.step()

                    running_loss += total_loss.item()
                    if (i+1) % self.print_every == 0:    # print every 200 mini-batches
                        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                        running_loss = 0.0

    def test(self):    
        pass

