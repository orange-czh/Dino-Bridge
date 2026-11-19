"""Supervised domain-generalization baseline for polyp training."""

import logging
import os
import pprint

import torch
import yaml
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from dataset.ssdg_dataset import SSDGDataset
from model import factory
from utils import AverageMeter, DiceLoss, fix_seed, init_log
from utils.env import get_module_version
from utils.mask_convert import converter
from utils.parse_args import parse_args
from utils.sampler import MultiDomainSampler

def main():
    args = parse_args()

    fix_seed(args.seed)

    cfg = {}
    for path in [args.config, args.shared_config, args.train_config]:
        with open(path) as config_file:
            cfg.update(yaml.safe_load(config_file))
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if cfg["dataset"] != "polyp" or cfg["n_domains"] != 5:
        raise ValueError("this release supports only the five-domain polyp dataset")
    if not 0 <= args.domain < 5:
        raise ValueError("--domain must be in [0, 4]")
    if (type(args.ratio) is float and not 0.0 < args.ratio <= 1.0) or args.ratio <= 0:
        raise ValueError("--ratio must be a fraction in (0, 1] or a positive count")
    if cfg["epochs"] <= 0 or cfg["iters"] <= 0:
        raise ValueError("epochs and iters must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    if args.model in {"dino", "dinob"}:
        from model.dinounet.network_architecture.ms_deform_attn import MSDA
        if MSDA is None:
            raise RuntimeError("build the deformable-attention CUDA extension before training; see README.md")

    torch.set_num_threads(cfg["num_threads"])

    convert = converter[cfg["dataset"]]
    trainset = SSDGDataset(name=cfg["dataset"],
                           root=cfg["data_root"],
                           target_domain=args.domain,
                           mode="train",
                           n_domains=cfg["n_domains"],
                           image_size=cfg["image_size"])
    _, trainset_l, indices = trainset.split_ulb_lb(args.ratio)

    if any(length == 0 for length in trainset_l.lengths):
        raise ValueError("every source domain needs at least one labeled sample")

    logger = init_log("global", logging.INFO)
    logger.propagate = 0  # type: ignore

    logger.info("labeled: \n{}".format(trainset_l))

    env = get_module_version([
        "numpy",
        "PIL",
        "scipy",
        "torch",
        "torchvision",
    ])
    env = {"env": env}
    all_args = {**cfg, **vars(args), **env}
    logger.info("cfg: \n{}\n".format(pprint.pformat(all_args)))

    os.makedirs(args.save_path, exist_ok=True)

    with open(args.save_path + "/split", "w") as f:
        f.write(str(indices))

    model = factory(args.model, cfg["n_channels"], cfg["n_classes"], args.pretrained)
    model.cuda()

    if cfg["optimizer"] == "sgd":
        optimizer = SGD(model.parameters(), cfg["lr"], momentum=0.9)
    elif cfg["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), cfg["lr"])
    elif cfg["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), cfg["lr"])
    else:
        raise NotImplementedError

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg["n_classes"])
    dice_args = dict(softmax="softmax", onehot=True)

    sampler = MultiDomainSampler(trainset_l.lengths, balanced=cfg["balanced"])
    trainloader = DataLoader(trainset_l,
                             batch_size=cfg["batch_size"],
                             pin_memory=True,
                             num_workers=cfg["num_workers"],
                             drop_last=True,
                             sampler=sampler)
    trainloader = iter(trainloader)

    n_iters = cfg["iters"]
    total_iters = n_iters * cfg["epochs"]
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]

        if epoch >= cfg["epochs"] - 1:
            logger.info("************ Skip trained checkpoint at epoch %i\n" %
                        epoch)
            exit()

        # reset learning rate
        current_iters = (epoch + 1) * n_iters
        lr = cfg["lr"] * (1 - current_iters / total_iters)**0.9
        optimizer.param_groups[0]["lr"] = lr

        logger.info("************ Load from checkpoint at epoch %i\n" % epoch)

    writer = SummaryWriter(args.save_path)

    for epoch in range(epoch + 1, cfg["epochs"]):
        logger.info(
            "===========> Epoch: {:}, LR: {:.5f}".
            format(epoch, optimizer.param_groups[0]["lr"]))

        model.train()
        total_loss = AverageMeter()

        for i in range(n_iters):
            # make sure batch size is the same
            image, _, mask = next(trainloader)
            image, mask = image.cuda(), mask.cuda()
            mask = convert(mask)

            pred = model(image)
            loss = (criterion_ce(pred, mask) +
                    criterion_dice(pred, mask, **dice_args)) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            current_iters = epoch * n_iters + i
            lr = cfg["lr"] * (1 - current_iters / total_iters)**0.9
            optimizer.param_groups[0]["lr"] = lr

            if i % max(1, n_iters // 8) == 0:
                logger.info("Iters: {:}, Total loss: {:.3f}".format(
                    i, total_loss.avg))
                writer.add_scalar("train/loss_all", total_loss.avg,
                                  current_iters)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
        if epoch == cfg["epochs"] - 1:
            torch.save(checkpoint, os.path.join(args.save_path, "last.pth"))

    writer.close()



if __name__ == "__main__":
    main()
