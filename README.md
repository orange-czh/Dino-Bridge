# DINO-Bridge

Official code for DINO-Bridge.

## Environment

Use Python 3.10 and a CUDA-capable NVIDIA GPU. The dependency versions below
match the original local environment: PyTorch 2.2.1 with CUDA 12.1 and
torchvision 0.17.1. Install a matching CUDA toolkit and a C++ compiler to build
the deformable-attention extension.

```bash
conda create -n dino-bridge python=3.10 -y
conda activate dino-bridge
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./model/dinounet/network_architecture/ops
```

## Pretrained weights

Obtain the DINOv3 weights [dinov3_vits16_pretrain_lvd1689m-08c60483.pth] through DINOv3 and follow its license terms. Supply a local file using `--pretrained`, or put it in `pretrained/` under the repository root.


## Train DINO-Bridge

Run from the repository root. A single held-out-domain run:

```bash
CUDA_VISIBLE_DEVICES=0 python train_ssdg.py \
  --config configs/dataset/xxx.yaml \
  --shared-config configs/shared.yaml \
  --train-config configs/adamw.yaml \
  --model dino --domain 0 --ratio 0.3 --seed 520 \
  --data-root /path/to/xxx \
  --pretrained /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --save-path outputs/xxx/domain0
```

Run all held-out domains serially on one GPU:

```bash
bash train.sh serial train_ssdg.py xxx dino 0 0.3 adamw \
  --data-root /path/to/xxx --pretrained /path/to/weights.pth
```

Or run them concurrently on five GPUs:

```bash
bash train.sh parallel train_ssdg.py xxx dino 0,1,2,3,4 0.3 adamw \
  --data-root /path/to/xxx --pretrained /path/to/weights.pth
```

## Acknowledgments

Thanks a lot for the valuable work of [DINOv3](https://arxiv.org/pdf/2508.10104) and [DINO-Unet](https://github.com/yifangao112/DinoUNet).