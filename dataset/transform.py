# Weak augmentation from the original polyp training pipeline.
# Original attribution: https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/medical/dataset/transform.py

import numpy as np
from scipy import ndimage


def random_rot_flip(img, mask, k=None, axis=None):
    if k is None:
        k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    mask = np.rot90(mask, k)
    if axis is None:
        axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return img, mask, k, axis


def random_rotate(img, mask, angle=None):
    if angle is None:
        angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return img, mask, angle
