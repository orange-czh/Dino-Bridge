"""Model factory with portable, user-supplied pretrained weights."""

from __future__ import annotations

from pathlib import Path


def factory(name: str, in_channels: int, out_classes: int,
            pretrained_path: str | None = None):
    root = Path(__file__).resolve().parent.parent
    if name in {"dino", "dinob"}:
        from .dinounet_training import DinoUNet

        variants = {
            "dino": ("dinounet_s", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
            "dinob": ("dinounet_b", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        }
        model_name, filename = variants[name]
        weights = Path(pretrained_path).expanduser() if pretrained_path else root / "pretrained" / filename
        if not weights.is_file():
            raise FileNotFoundError(
                f"DINOv3 weights not found: {weights}. Supply --pretrained or see README.md."
            )
        return DinoUNet(
            network_config=None,
            input_channels=in_channels,
            num_classes=out_classes,
            dinov3_pretrained_path=str(weights.resolve()),
            dinov3_model_name=model_name,
        )

    if name in {"unet", "unet32", "unet8", "unet_nd", "unet32_nd", "unet8_nd"}:
        from .unet import UNet

        num_filters = 8 if name.startswith("unet8") else 32 if name.startswith("unet32") else 16
        return UNet(in_channels, out_classes, num_filters=num_filters,
                    use_dropout=not name.endswith("_nd"))

    if name == "deeplab":
        import torch
        from .deeplabv3p import DeepLabV3Plus

        weights = Path(pretrained_path).expanduser() if pretrained_path else root / "pretrained/resnet50_v1c.pth"
        if not weights.is_file():
            raise FileNotFoundError(f"ResNet weights not found: {weights}. Supply --pretrained.")
        model = DeepLabV3Plus(in_channels, out_classes)
        model_dict = model.backbone.state_dict()
        state_dict = torch.load(weights, map_location="cpu")
        for key, value in state_dict.items():
            if key in model_dict and model_dict[key].size() == value.size():
                model_dict[key] = value
        model.backbone.load_state_dict(model_dict)
        return model

    raise ValueError(f"unsupported model: {name}")
