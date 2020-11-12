import models.ocflow_model
from models.data.utils.flow_utils import read_flow
from models.networks.image_inpainting_net import SceneCompletionNet
from models.networks.warping_layer import Warping
from imageio import imread
import os
import torch
import torch.optim as optim
import numpy as np

class OverfitInpainting(): 
    def __init__(self, num_epoch, print_every): 
        self.num_epoch = num_epoch
        self.print_every = print_every
    def train(self): 
        completion_net = SceneCompletionNet()
        warping = Warping()

        dir_name = os.path.dirname(os.getcwd())
        dir_flow = os.path.join(dir_name, 'sample_data/optical_flow.flo')
        dir_mask = os.path.join(dir_name, 'sample_data/occlusion_mask.png')
        dir_im1 = os.path.join(dir_name, 'sample_data/image1.png')
        dir_im2 = os.path.join(dir_name, 'sample_data/image2.png')

        flow = torch.from_numpy(read_flow(dir_flow)) # [H,W,2]
        occlusion_mask = torch.from_numpy(np.array(imread(dir_mask))) #[H,W]
        img1 = torch.from_numpy(np.array(imread(dir_im1))) # [H,W,3]
        img2 = torch.from_numpy(np.array(imread(dir_im2)))

        flow = flow.unsqueeze(0).permute(0,3,1,2) #[1,2,H,W]
        occlusion_mask = occlusion_mask.view(1,1,occlusion_mask.size()[0], occlusion_mask.size()[1]) #[1,1,H,W]
        img1 = img1.unsqueeze(0).permute(0,3,1,2)
        img2 = img2.unsqueeze(0).permute(0,3,1,2)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        completion_net.to(device)
        optimizer = optim.Adam(completion_net.parameters(), lr = 0.001)
        img1 = img1.to(device)
        img2 = img2.to(device)
        flow = flow.to(device)
        occlusion_mask = occlusion_mask.to(device)

        running_loss = 0.0
        for epoch in range(self.num_epoch): 
            optimizer.zero_grad()
            Iw1= warping(img2, flow)
            Io1 = Iw1*occlusion_mask
            Ic1 = completion_net(Io1)

            loss = torch.sum(torch.abs(img1-Ic1)*(1-occlusion_mask)) / (torch.sum(1-occlusion_mask) + 1e-12) 
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (epoch+1) % self.print_every == 0:    # print every 200 mini-batches
                print('epoch %d th, average loss: %.3f' % ( epoch+1, running_loss / (epoch+1)))
                running_loss = 0
def main(): 
    model = OverfitInpainting(num_epoch = 10, print_every =1)
    model.train()
if __name__ == '__main__':
    main()
    