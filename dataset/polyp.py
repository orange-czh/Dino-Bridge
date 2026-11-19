from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .base_dataset import BaseDataset, Stage


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


class Polyp(BaseDataset):
    """Five-domain 2D binary polyp segmentation dataset.

    Images and masks are paired by filename stem.  Masks are returned as one
    foreground channel: black is polyp and white is background.  A threshold
    is used instead of exact equality because JPEG masks may contain small
    compression-induced gray values near binary boundaries.
    """

    def __init__(self, root: str | Path, domain: int, stage: Stage):
        super().__init__(
            data_dir=Path(root) / f"Domain{domain + 1}" / "train",
            domain=domain,
            stage=stage,
            image_filter=Polyp._polyp_image_filter,
            label_filter=Polyp._polyp_label_filter,
        )

    def load_image_label(
        self,
        image_path: Path,
        label_path: Path,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        label = np.asarray(Image.open(label_path).convert("L"))

        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        # One foreground channel: black lesion -> class 1; white background
        # -> class 0.  Thresholding is robust to JPEG gray boundary artifacts.
        label = (label < 128).astype(np.uint8)[None, ...]
        return image, label

    @staticmethod
    def _polyp_image_filter(path: Path) -> str | None:
        if (path.parent.name == "image" and not path.name.startswith(".")
                and path.suffix.lower() in _IMAGE_SUFFIXES):
            return path.stem
        return None

    @staticmethod
    def _polyp_label_filter(path: Path) -> str | None:
        if (path.parent.name == "mask" and not path.name.startswith(".")
                and path.suffix.lower() in _IMAGE_SUFFIXES):
            return path.stem
        return None
