import torch
import torch.nn as nn
import torch.nn.functional as F

import collections

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

def normalize_features(feature_list, normalize=True, center=True, moments_across_channels=True,
                        moments_across_images=True):
    """Normalizes feature tensors (e.g., before computing the cost volume).

    Args:
        feature_list: list of Tensors, each with dimensions [b, c, h, w]
        normalize: bool flag, divide features by their standard deviation
        center: bool flag, subtract feature mean
        moments_across_channels: bool flag, compute mean and std across channels
        moments_across_images: bool flag, compute mean and std across images

    Returns:
        list, normalized feature_list
    """

    # Compute feature statistics

    statistics = collections.defaultdict(list)
    dim = (1, 2, 3) if moments_across_channels else (2, 3)
    for feature_image in feature_list:
        variance, mean = torch.var_mean(feature_image, dim=dim, keepdim=True, unbiased=False)
        statistics['mean'].append(mean)
        statistics['var'].append(variance)

    if moments_across_images:
        statistics['mean'] = [torch.mean(torch.stack(statistics['mean']))] * len(feature_list)
        statistics['var'] = [torch.mean(torch.stack(statistics['var']))] * len(feature_list)

    statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]
    
    # Center and normalize features
    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, statistics['mean'])
        ] 

    if normalize:
        feature_list = [
            f / std for f, std in zip(feature_list, statistics['std'])
        ]
    return feature_list
