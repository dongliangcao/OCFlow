import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_cost_volume(features1, features2, max_displacement=4):
    """Compute the cost volume between features1 and features2.

    Displace features2 up to max_displacement in any direction and compute the
    per pixel cost of features1 and the displaced features2.

    Args:
    features1: Tensor of shape [b, c, h, w]
    features2: Tensor of shape [b, c, h, w]
    max_displacement: int, maximum displacement for cost volume computation.

    Returns:
    tf.tensor of shape [b, (2 * max_displacement + 1) ** 2, h, w] of costs for
    all displacements.
    """
    _, _, height, width = features1.size()
#     if max_displacement <= 0 or max_displacement >= height:
#         raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement
    num_shifts = 2 * max_disp + 1

    # Pad features2 and shift it while keeping features1 fixed 
    # to compute the cost volume through correlation.

    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = F.pad(features2, (max_disp, max_disp, max_disp, max_disp))
    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            corr = torch.mean(features1 * features2_padded[:, :, i:(height+i), j:(width+j)], dim=1, keepdim=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, dim=1)
    return cost_volume


