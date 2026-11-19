"""Convert binary polyp training masks to class IDs."""

import torch


def polyp_convert(mask: torch.Tensor):
    # mask: b, 1, h, w; 0 = white background, 1 = black polyp lesion
    return mask.squeeze(1).long()


converter = {"polyp": polyp_convert}
