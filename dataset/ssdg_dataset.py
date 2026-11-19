"""Source-domain polyp data loading for DG and semi-supervised DG training."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from .polyp import Polyp
from .transform import random_rot_flip, random_rotate


class SSDGDataset(Dataset):
    """Load the four source domains, excluding the held-out domain entirely.

    Samples contain an augmented image, its source-domain index, and a binary
    mask. Domain indices enumerate source domains in their original order.
    """

    def __init__(self, name: str, root: str | Path,
                 target_domain: int | list[int], mode: str,
                 n_domains: int, image_size: int):
        if name.lower() != "polyp":
            raise ValueError("this release supports only polyp")
        if mode != "train":
            raise ValueError("only training data loading is included")
        if n_domains != 5:
            raise ValueError("polyp requires five domains")
        targets = [target_domain] if isinstance(target_domain, int) else target_domain
        if len(targets) != 1 or targets[0] not in range(n_domains):
            raise ValueError("target_domain must identify one domain in [0, 4]")
        self.name = name.lower()
        self.root = root
        self.target_domain = target_domain
        self.mode = mode
        self.image_size = image_size
        self.datasets = [Polyp(root, domain, "train")
                         for domain in range(n_domains) if domain not in targets]

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    @property
    def lengths(self) -> list[int]:
        return [len(dataset) for dataset in self.datasets]

    def __getitem__(self, index):
        cnt = 0
        for domain_id, dataset in enumerate(self.datasets):
            if index < cnt + len(dataset):
                break
            cnt += len(dataset)
        else:
            raise IndexError(f"out of range: {index}")

        img, mask, _ = dataset[index - cnt]
        img = img.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)
        weak_fn = None
        if random.random() > 0.5:
            weak_fn = random_rot_flip
        elif random.random() > 0.5:
            weak_fn = random_rotate
        if weak_fn:
            img, mask, *_ = weak_fn(img, mask)

        img = zoom(img, (self.image_size / img.shape[0],
                         self.image_size / img.shape[1], 1), order=0)
        mask = zoom(mask, (self.image_size / mask.shape[0],
                           self.image_size / mask.shape[1], 1), order=0)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
        return img, domain_id, mask

    def split_ulb_lb(self, fraction: float | int):
        """Split each source domain using the original seeded split rules.

        A float denotes the labeled fraction; an integer denotes a count per
        source domain. Returned indices follow the original logging format.
        """
        if type(fraction) is int:
            if fraction <= 0:
                raise ValueError("labeled sample count must be positive")
        elif type(fraction) is float:
            if not 0.0 < fraction <= 1.0:
                raise ValueError("labeled fraction must be in (0, 1]")
        else:
            raise ValueError("expected a float fraction or integer count")

        lb_dataset, ulb_dataset = deepcopy(self), deepcopy(self)
        lb_dataset.mode, ulb_dataset.mode = "train_l", "train_u"
        ulbs, lbs, indices = [], [], []
        for dataset in self.datasets:
            if type(fraction) is int:
                (lb, ulb), lb_indices = dataset.random_split_k(fraction)
                ulb_indices = lb_indices
            else:
                (lb, ulb), (_, ulb_indices) = dataset.random_split(fraction)
            ulbs.append(ulb)
            lbs.append(lb)
            indices.append(ulb_indices)
        ulb_dataset.datasets, lb_dataset.datasets = ulbs, lbs
        return ulb_dataset, lb_dataset, indices

    def __repr__(self) -> str:
        return "\n".join(str(dataset) for dataset in self.datasets)
