"""network to estimate optical flow and occlusion mask"""
import torch
import torch.nn as nn
import torch.nn.functional as F
        
class CostVolumeLayer(nn.Module):
    """
    Calculate the cost volume between the warped feature and the reference feature 
    """
    def __init__(self, search_range=4):
        super(CostVolumeLayer, self).__init__()
        self.window = search_range
    def forward(self,x, warped): 
        """
        Args: 
        x: input feature, torch.Tensor [B, C, H, W]
        warped: warped feature, torch.Tensor[B,C,H,W]
        Returns: 
        stacked: cost volume tensor, torch.Tensor [B, (search_range*2+1)**2, H, W] 
        """
        total = []
        keys = []

        row_shifted = [warped]

        for i in range(self.window+1):
            if i != 0:
                row_shifted = [F.pad(row_shifted[0], (0,0,0,1)), F.pad(row_shifted[1], (0,0,1,0))]

                row_shifted = [row_shifted[0][:, :, 1:, :], row_shifted[1][:, :, :-1, :]]

            for side in range(len(row_shifted)):
                total.append(torch.mean(row_shifted[side] * x, dim = 1))
                keys.append([i * (-1) ** side, 0])
                col_previous = [row_shifted[side], row_shifted[side]]
                for j in range(1, self.window+1):
                    col_shifted = [F.pad(col_previous[0], (0,1)), F.pad(col_previous[1], (1,0))]

                    col_shifted = [col_shifted[0][:, :, :, 1:], col_shifted[1][:, :, :, :-1]]

                    for col_side in range(len(col_shifted)):
                        total.append(torch.mean(col_shifted[col_side] * x, dim=1))
                        keys.append([i * (-1) ** side, j * (-1) ** col_side])
                    col_previous = col_shifted

            if i == 0:
                row_shifted *= 2

        total = [t for t, _ in sorted(zip(total, keys), key=lambda pair: pair[1])]
        stacked = torch.stack(total, dim =1)

        return stacked / ((2.0*self.window+1)**2.0)



