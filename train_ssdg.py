"""Train DINO-Bridge on source domains for binary polyp segmentation.

CSRV, RCTC, the EMA teacher, and the original training defaults are retained.
The held-out domain is excluded from loading and optimization.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pprint
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn.functional as F
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
from utils.parse_args import float_or_int
from utils.sampler import MultiDomainSampler


def parse_ssdg_args(args: Sequence[str] | None = None):
    """Parse baseline arguments plus DINO-Bridge hyperparameters.

    The defaults make the command emitted by ``train.sh`` directly usable;
    optional arguments can still be appended after the training config name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--shared-config", default="configs/shared.yaml")
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--model", choices=["dino", "dinob"], required=True)
    parser.add_argument("--pretrained", help="path to DINOv3 pretrained weights")
    parser.add_argument("--data-root", help="override the polyp data root")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--domain", type=int, required=True)
    parser.add_argument("--ratio", type=float_or_int, required=True)

    # EMA teacher and RCTC.
    parser.add_argument("--decay", type=float, default=0.99,
                        help="EMA decay for the teacher")
    parser.add_argument("--lambda-u-max", type=float, default=1.0,
                        help="maximum RCTC loss weight")
    parser.add_argument("--rampup-iters", type=int, default=None,
                        help="linear RCTC ramp-up; default is five epochs")

    # CSRV relation measurement and feedback controller.  The paper text does
    # not prescribe these numerical values, so they are explicit and tunable.
    parser.add_argument("--relation-temperature", type=float, default=0.05)
    parser.add_argument("--low-frequency-ratio", type=float, default=0.05,
                        help="half-width ratio of the centered Fourier mask")
    parser.add_argument("--rho-init", type=float, default=0.3)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--rho-step", type=float, default=0.1,
                        help="eta_rho in the CSRV controller")
    parser.add_argument("--rho-beta", type=float, default=0.9,
                        help="temporal smoothing coefficient")
    parser.add_argument("--rho-target", type=float, default=0.9,
                        help="desired relation-stability operating point")

    # Each iteration contains both labeled and unlabeled data.  Half of the
    # baseline batch size for each stream keeps the largest concatenated
    # forward at the original configured batch size.
    parser.add_argument("--labeled-batch-size", type=int, default=None)
    parser.add_argument("--unlabeled-batch-size", type=int, default=None)
    return parser.parse_args(args)


def validate_method_args(args) -> None:
    if not 0.0 <= args.decay < 1.0:
        raise ValueError("--decay must be in [0, 1)")
    if args.lambda_u_max < 0.0:
        raise ValueError("--lambda-u-max must be non-negative")
    if args.rampup_iters is not None and args.rampup_iters < 0:
        raise ValueError("--rampup-iters must be non-negative")
    if args.relation_temperature <= 0.0:
        raise ValueError("--relation-temperature must be positive")
    if not 0.0 < args.low_frequency_ratio <= 0.5:
        raise ValueError("--low-frequency-ratio must be in (0, 0.5]")
    if not 0.0 <= args.rho_init <= args.rho_max <= 1.0:
        raise ValueError("expected 0 <= rho-init <= rho-max <= 1")
    if args.rho_step < 0.0:
        raise ValueError("--rho-step must be non-negative")
    if not 0.0 <= args.rho_beta < 1.0:
        raise ValueError("--rho-beta must be in [0, 1)")
    if not 0.0 <= args.rho_target <= 1.0:
        raise ValueError("--rho-target must be in [0, 1]")


def load_config(args) -> dict:
    with open(args.config, "r") as f:
        cfg: dict = yaml.safe_load(f)
    with open(args.shared_config, "r") as f:
        cfg.update(yaml.safe_load(f))
    with open(args.train_config, "r") as f:
        cfg.update(yaml.safe_load(f))
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    return cfg


def build_optimizer(model: nn.Module, cfg: dict):
    # The frozen DINOv3 backbone is intentionally excluded from optimization.
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    if cfg["optimizer"] == "sgd":
        return SGD(parameters, cfg["lr"], momentum=0.9)
    if cfg["optimizer"] == "adamw":
        return AdamW(parameters, cfg["lr"])
    if cfg["optimizer"] == "adam":
        return Adam(parameters, cfg["lr"])
    raise NotImplementedError("unsupported optimizer: {}".format(
        cfg["optimizer"]))


def polynomial_lr(base_lr: float, step: int, total_steps: int) -> float:
    progress = min(max(step / total_steps, 0.0), 1.0)
    return base_lr * (1.0 - progress) ** 0.9


def sample_cross_source_donors(images: torch.Tensor,
                               domains: torch.Tensor) -> torch.Tensor:
    """Choose one donor from a different source domain for every image."""

    domains = domains.reshape(-1)
    donor_indices = []
    for sample_index in range(len(domains)):
        candidates = torch.nonzero(
            domains != domains[sample_index], as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise RuntimeError(
                "cross-source FDA requires at least two domains per batch")
        choice = torch.randint(candidates.numel(), (1,),
                               device=domains.device)
        donor_indices.append(candidates[choice].squeeze(0))
    return images[torch.stack(donor_indices)]


def fourier_amplitude_mix(source: torch.Tensor,
                          donor: torch.Tensor,
                          rho: float,
                          low_frequency_ratio: float) -> torch.Tensor:
    """Partially transfer centered low-frequency Fourier amplitudes.

    The source phase is kept intact, so the weak view and expanded view remain
    geometrically aligned and can share a pixel-wise segmentation target.
    """

    if rho == 0.0:
        return source.clone()

    source_fft = torch.fft.fft2(source, dim=(-2, -1))
    donor_fft = torch.fft.fft2(donor, dim=(-2, -1))
    source_amplitude = torch.fft.fftshift(
        source_fft.abs(), dim=(-2, -1))
    donor_amplitude = torch.fft.fftshift(
        donor_fft.abs(), dim=(-2, -1))

    height, width = source.shape[-2:]
    radius = max(1, int(round(min(height, width) * low_frequency_ratio)))
    center_h, center_w = height // 2, width // 2
    top, bottom = center_h - radius, center_h + radius + 1
    left, right = center_w - radius, center_w + radius + 1

    mixed_amplitude = source_amplitude.clone()
    source_low = source_amplitude[..., top:bottom, left:right]
    donor_low = donor_amplitude[..., top:bottom, left:right]
    mixed_amplitude[..., top:bottom, left:right] = (
        source_low + rho * (donor_low - source_low))

    mixed_amplitude = torch.fft.ifftshift(
        mixed_amplitude, dim=(-2, -1))
    source_phase = torch.angle(source_fft)
    mixed_fft = mixed_amplitude * torch.exp(1j * source_phase)
    expanded = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real
    return expanded.clamp_(0.0, 1.0)


class FinalPatchTokenCapture:
    """Capture normalized final-block patch tokens without changing model code.

    ``DINOv3_Adapter`` already asks the frozen backbone for intermediate
    features.  A hook on the backbone's final normalization reuses those
    tokens, avoiding two additional DINOv3 forward passes per iteration.
    """

    def __init__(self, backbone: nn.Module):
        self.backbone = backbone
        self._tokens: torch.Tensor | None = None
        self._handle = backbone.norm.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        if self.backbone.untie_cls_and_patch_norms:
            # In this variant ``norm`` receives patch tokens only.
            patch_tokens = output
        else:
            first_patch = self.backbone.n_storage_tokens + 1
            patch_tokens = output[:, first_patch:]
        # DINOv3 is frozen; detach explicitly to make the CSRV control signal
        # independent from student/teacher gradient graphs.
        self._tokens = patch_tokens.detach()

    def pop(self) -> torch.Tensor:
        if self._tokens is None:
            raise RuntimeError("DINOv3 patch-token hook did not run")
        tokens = self._tokens
        self._tokens = None
        return tokens

    def clear(self) -> None:
        self._tokens = None

    def close(self) -> None:
        self._handle.remove()


def local_relation_distribution(tokens: torch.Tensor,
                                grid_size: tuple[int, int],
                                temperature: float) -> torch.Tensor:
    """Compute row-normalized 3x3-neighborhood relation distributions."""

    grid_h, grid_w = grid_size
    if tokens.ndim != 3 or tokens.shape[1] != grid_h * grid_w:
        raise ValueError(
            "patch-token count {} does not match grid {}x{}".format(
                tokens.shape[1], grid_h, grid_w))

    features = F.normalize(tokens.float(), p=2, dim=-1, eps=1e-6)
    features = features.reshape(tokens.shape[0], grid_h, grid_w, -1)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    logits = torch.full(
        (tokens.shape[0], len(offsets), grid_h, grid_w),
        -torch.inf,
        dtype=torch.float32,
        device=tokens.device,
    )

    for neighbor_index, (offset_h, offset_w) in enumerate(offsets):
        center_h0 = max(0, -offset_h)
        center_h1 = min(grid_h, grid_h - offset_h)
        center_w0 = max(0, -offset_w)
        center_w1 = min(grid_w, grid_w - offset_w)
        neighbor_h0, neighbor_h1 = (center_h0 + offset_h,
                                    center_h1 + offset_h)
        neighbor_w0, neighbor_w1 = (center_w0 + offset_w,
                                    center_w1 + offset_w)

        center = features[:, center_h0:center_h1, center_w0:center_w1]
        neighbor = features[:, neighbor_h0:neighbor_h1,
                            neighbor_w0:neighbor_w1]
        similarity = (center * neighbor).sum(dim=-1) / temperature
        logits[:, neighbor_index, center_h0:center_h1,
               center_w0:center_w1] = similarity

    return torch.softmax(logits, dim=1)


def relation_stability_map(weak_tokens: torch.Tensor,
                           appearance_tokens: torch.Tensor,
                           grid_size: tuple[int, int],
                           temperature: float) -> torch.Tensor:
    """Bhattacharyya stability for every patch, with values in [0, 1]."""

    weak_relations = local_relation_distribution(
        weak_tokens, grid_size, temperature)
    appearance_relations = local_relation_distribution(
        appearance_tokens, grid_size, temperature)
    stability = torch.sqrt(
        (weak_relations * appearance_relations).clamp_min(0.0)).sum(dim=1)
    return stability.clamp_(0.0, 1.0)


def segmentation_loss(logits: torch.Tensor,
                      target: torch.Tensor,
                      criterion_ce: nn.Module,
                      criterion_dice: nn.Module) -> torch.Tensor:
    # Eq. (supervised_loss): ell_seg = CE + Dice.
    return (criterion_ce(logits, target) +
            criterion_dice(logits, target,
                           softmax="softmax", onehot=True))


def rctc_loss(student_logits: torch.Tensor,
              teacher_weak_logits: torch.Tensor,
              teacher_appearance_logits: torch.Tensor,
              patch_stability: torch.Tensor,
              include_distribution_diagnostics: bool = False,
              ) -> tuple[torch.Tensor, dict]:
    """Compute RCTC and detached diagnostics for reliability saturation.

    Distribution quantiles are optional because computing them over every
    prediction pixel is relatively expensive.  The training loop enables
    them only on iterations that are written to the log.
    """

    teacher_weak = torch.softmax(teacher_weak_logits.float(), dim=1).detach()
    teacher_appearance = torch.softmax(
        teacher_appearance_logits.float(), dim=1).detach()
    mean_posterior = 0.5 * (teacher_weak + teacher_appearance)
    eps = 1e-8

    js_weak = (teacher_weak *
               (torch.log(teacher_weak.clamp_min(eps)) -
                torch.log(mean_posterior.clamp_min(eps)))).sum(dim=1)
    js_appearance = (teacher_appearance *
                     (torch.log(teacher_appearance.clamp_min(eps)) -
                      torch.log(mean_posterior.clamp_min(eps)))).sum(dim=1)
    js_divergence = 0.5 * (js_weak + js_appearance)
    semantic_stability = (1.0 - js_divergence / math.log(2.0)).clamp_(0.0, 1.0)

    n_classes = teacher_weak.shape[1]
    confidence = teacher_weak.max(dim=1).values
    confidence = ((n_classes * confidence - 1.0) /
                  (n_classes - 1.0)).clamp_(0.0, 1.0)
    structural_stability = F.interpolate(
        patch_stability.unsqueeze(1),
        size=student_logits.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    weight = (structural_stability * semantic_stability * confidence).detach()
    student_log_posterior = F.log_softmax(student_logits.float(), dim=1)
    pixel_cross_entropy = -(teacher_weak * student_log_posterior).sum(dim=1)
    # A plain mean is the fixed B_u*H*W normalization from the method.
    loss = (weight * pixel_cross_entropy).mean()
    raw_rctc = pixel_cross_entropy.mean()
    teacher_disagreement = (
        teacher_weak.argmax(dim=1) != teacher_appearance.argmax(dim=1)
    ).float().mean()

    # Stack the detached scalars before transferring them to the CPU.  This
    # keeps diagnostics from affecting gradients and incurs only one device
    # synchronization per iteration.
    diagnostic_names = [
        "weight", "structural", "semantic", "confidence",
        "teacher_disagreement", "raw_rctc", "weighted_rctc",
    ]
    diagnostic_tensors = [
        weight.mean(), structural_stability.mean(),
        semantic_stability.mean(), confidence.mean(),
        teacher_disagreement, raw_rctc, loss,
    ]
    if include_distribution_diagnostics:
        quantile_levels = weight.new_tensor([0.10, 0.50, 0.90])
        weight_quantiles = torch.quantile(
            weight.flatten(), quantile_levels)
        structural_quantiles = torch.quantile(
            structural_stability.detach().flatten(), quantile_levels)
        diagnostic_names.extend([
            "weight_p10", "weight_p50", "weight_p90",
            "structural_p10", "structural_p50", "structural_p90",
        ])
        diagnostic_tensors.extend([
            *weight_quantiles.unbind(), *structural_quantiles.unbind(),
        ])
    diagnostic_values = torch.stack(
        [value.detach() for value in diagnostic_tensors]).cpu().tolist()
    diagnostics = dict(zip(diagnostic_names, diagnostic_values))
    return loss, diagnostics


@torch.no_grad()
def update_ema_teacher(student: nn.Module,
                       teacher: nn.Module,
                       decay: float) -> None:
    """EMA-update trainable parameters and synchronize non-backbone buffers."""

    teacher_parameters = dict(teacher.named_parameters())
    for name, student_parameter in student.named_parameters():
        if not student_parameter.requires_grad:
            # The frozen DINOv3 backbone object is shared by both models.
            continue
        teacher_parameter = teacher_parameters[name]
        teacher_parameter.mul_(decay).add_(student_parameter,
                                           alpha=1.0 - decay)

    teacher_buffers = dict(teacher.named_buffers())
    for name, student_buffer in student.named_buffers():
        if ".backbone." not in name:
            teacher_buffers[name].copy_(student_buffer)


def create_student_teacher(model_name: str,
                           n_channels: int,
                           n_classes: int,
                           device: torch.device,
                           pretrained_path: str | None = None):
    """Create student/teacher models with one shared frozen DINOv3 encoder."""

    student = factory(model_name, n_channels, n_classes, pretrained_path)
    try:
        student_backbone = student.encoder.dinov3_adapter.backbone
    except AttributeError as error:
        raise ValueError(
            "DINO-Bridge requires a DinoUNet model with a DINOv3 backbone") \
            from error

    teacher = deepcopy(student)
    # Drop the copied frozen encoder and share the student's frozen reference,
    # as specified in the method.  All trainable adapter/decoder parameters
    # remain separate EMA teacher parameters.
    teacher.encoder.dinov3_adapter.backbone = student_backbone
    student_backbone.requires_grad_(False)
    teacher.requires_grad_(False)
    student.to(device)
    teacher.to(device)
    return student, teacher, student_backbone


def make_checkpoint(student: nn.Module,
                    teacher: nn.Module,
                    optimizer,
                    epoch: int,
                    global_step: int,
                    rho: float,
                    smoothed_stability: float,
                    method_config: dict) -> dict:
    return {
        "model": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        # ``rho`` is rho_(e+1), ready for the next epoch after resume.
        "rho": rho,
        "smoothed_stability": smoothed_stability,
        "method_config": method_config,
    }


def main() -> None:
    args = parse_ssdg_args()
    validate_method_args(args)
    fix_seed(args.seed)
    cfg = load_config(args)

    if cfg["dataset"] != "polyp":
        raise ValueError("this release supports only the polyp dataset")
    if cfg["n_domains"] != 5 or not 0 <= args.domain < 5:
        raise ValueError("polyp requires five domains and --domain in [0, 4]")
    if (type(args.ratio) is float and not 0.0 < args.ratio < 1.0) or args.ratio <= 0:
        raise ValueError("--ratio must be a fraction in (0, 1) or a positive count")
    if cfg["epochs"] <= 0 or cfg["iters"] <= 0:
        raise ValueError("epochs and iters must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DinoUNet training")
    from model.dinounet.network_architecture.ms_deform_attn import MSDA
    if MSDA is None:
        raise RuntimeError("build the deformable-attention CUDA extension before training; see README.md")
    torch.set_num_threads(cfg["num_threads"])
    device = torch.device("cuda")

    trainset = SSDGDataset(
        name=cfg["dataset"],
        root=cfg["data_root"],
        target_domain=args.domain,
        mode="train",
        n_domains=cfg["n_domains"],
        image_size=cfg["image_size"],
    )
    trainset_u, trainset_l, split_indices = trainset.split_ulb_lb(args.ratio)
    if any(length == 0 for length in trainset_l.lengths):
        raise ValueError("every source domain needs at least one labeled sample")
    if any(length == 0 for length in trainset_u.lengths):
        raise ValueError("every source domain needs at least one unlabeled sample")

    logger = init_log("global", logging.INFO)
    logger.propagate = 0  # type: ignore
    logger.info("labeled: \n%s", trainset_l)
    logger.info("unlabeled: \n%s", trainset_u)

    n_iters = cfg["iters"]
    total_iters = n_iters * cfg["epochs"]
    rampup_iters = (args.rampup_iters if args.rampup_iters is not None
                    else 5 * n_iters)
    default_stream_batch = max(1, cfg["batch_size"] // 2)
    labeled_batch_size = args.labeled_batch_size or default_stream_batch
    unlabeled_batch_size = args.unlabeled_batch_size or default_stream_batch
    if labeled_batch_size <= 0 or unlabeled_batch_size <= 0:
        raise ValueError("labeled/unlabeled batch sizes must be positive")

    method_config = {
        "name": "DINO-Bridge",
        "ema_decay": args.decay,
        "lambda_u_max": args.lambda_u_max,
        "rampup_iters": rampup_iters,
        "relation_temperature": args.relation_temperature,
        "low_frequency_ratio": args.low_frequency_ratio,
        "rho_init": args.rho_init,
        "rho_max": args.rho_max,
        "rho_step": args.rho_step,
        "rho_beta": args.rho_beta,
        "rho_target": args.rho_target,
        "labeled_batch_size": labeled_batch_size,
        "unlabeled_batch_size": unlabeled_batch_size,
    }
    env = {"env": get_module_version([
        "numpy", "PIL", "scipy", "torch", "torchvision",
    ])}
    logger.info("cfg: \n%s\n", pprint.pformat(
        {**cfg, **vars(args), "dino_bridge": method_config, **env}))

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "split"), "w") as split_file:
        split_file.write(str(split_indices))

    labeled_loader = iter(DataLoader(
        trainset_l,
        batch_size=labeled_batch_size,
        pin_memory=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        sampler=MultiDomainSampler(trainset_l.lengths,
                                   balanced=cfg["balanced"]),
    ))
    unlabeled_loader = iter(DataLoader(
        trainset_u,
        batch_size=unlabeled_batch_size,
        pin_memory=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        sampler=MultiDomainSampler(trainset_u.lengths,
                                   balanced=cfg["balanced"]),
    ))
    student, teacher, backbone = create_student_teacher(
        args.model, cfg["n_channels"], cfg["n_classes"], device, args.pretrained)
    token_capture = FinalPatchTokenCapture(backbone)
    optimizer = build_optimizer(student, cfg)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg["n_classes"])
    convert_mask = converter[cfg["dataset"]]

    completed_epoch = -1
    global_step = 0
    rho = args.rho_init
    smoothed_stability = args.rho_target
    latest_path = os.path.join(args.save_path, "latest.pth")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, map_location="cpu")
        student.load_state_dict(checkpoint["model"])
        teacher.load_state_dict(checkpoint.get("teacher", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_epoch = checkpoint["epoch"]
        global_step = checkpoint.get(
            "global_step", (completed_epoch + 1) * n_iters)
        rho = checkpoint.get("rho", rho)
        smoothed_stability = checkpoint.get(
            "smoothed_stability", smoothed_stability)
        logger.info("************ Load from checkpoint at epoch %i\n",
                    completed_epoch)
        del checkpoint

    optimizer.param_groups[0]["lr"] = polynomial_lr(
        cfg["lr"], global_step, total_iters)
    writer = SummaryWriter(args.save_path)
    log_interval = max(1, n_iters // 8)

    for epoch in range(completed_epoch + 1, cfg["epochs"]):
        logger.info(
            "===========> Epoch: %i, LR: %.5f, rho: %.4f",
            epoch, optimizer.param_groups[0]["lr"], rho)

        student.train()
        teacher.eval()
        # Teacher.eval() is called after student.train(), keeping the shared
        # frozen DINOv3 encoder deterministic while student heads remain train.
        backbone.eval()

        total_loss_meter = AverageMeter()
        supervised_loss_meter = AverageMeter()
        unsupervised_loss_meter = AverageMeter()
        weight_meter = AverageMeter()
        raw_rctc_meter = AverageMeter()
        structural_meter = AverageMeter()
        semantic_meter = AverageMeter()
        confidence_meter = AverageMeter()
        teacher_disagreement_meter = AverageMeter()
        appearance_shift_meter = AverageMeter()
        epoch_stability_scores = []

        for iteration in range(n_iters):
            image_l, domain_l, mask_l = next(labeled_loader)
            image_u, domain_u, _mask_u = next(unlabeled_loader)
            image_l = image_l.to(device, non_blocking=True)
            image_u = image_u.to(device, non_blocking=True)
            domain_l = domain_l.to(device, non_blocking=True)
            domain_u = domain_u.to(device, non_blocking=True)
            mask_l = convert_mask(mask_l.to(device, non_blocking=True))

            weak_images = torch.cat([image_l, image_u], dim=0)
            source_domains = torch.cat([domain_l, domain_u], dim=0)
            donor_images = sample_cross_source_donors(
                weak_images, source_domains)
            appearance_images = fourier_amplitude_mix(
                weak_images, donor_images, rho,
                args.low_frequency_ratio)
            appearance_shift_l1 = (
                appearance_images - weak_images).abs().mean().item()
            appearance_l, appearance_u = appearance_images.split(
                [len(image_l), len(image_u)], dim=0)

            # Supervise both aligned labeled views.  This student forward also
            # provides frozen final-block tokens for labeled CSRV statistics.
            labeled_logits = student(torch.cat([image_l, appearance_l], dim=0))
            labeled_tokens = token_capture.pop()
            logits_l_weak, logits_l_appearance = labeled_logits.chunk(2)
            tokens_l_weak, tokens_l_appearance = labeled_tokens.chunk(2)

            patch_size = backbone.patch_size
            if isinstance(patch_size, tuple):
                patch_h, patch_w = patch_size
            else:
                patch_h = patch_w = patch_size
            patch_grid = (image_l.shape[-2] // patch_h,
                          image_l.shape[-1] // patch_w)
            stability_l = relation_stability_map(
                tokens_l_weak, tokens_l_appearance, patch_grid,
                args.relation_temperature)

            loss_sup = 0.5 * (
                segmentation_loss(logits_l_weak, mask_l,
                                  criterion_ce, criterion_dice) +
                segmentation_loss(logits_l_appearance, mask_l,
                                  criterion_ce, criterion_dice))

            # The EMA teacher predicts both unlabeled views in eval/no-grad
            # mode.  Its shared frozen backbone tokens provide unlabeled CSRV.
            with torch.no_grad():
                teacher_logits = teacher(torch.cat(
                    [image_u, appearance_u], dim=0))
            teacher_tokens = token_capture.pop()
            teacher_weak_logits, teacher_appearance_logits = \
                teacher_logits.chunk(2)
            tokens_u_weak, tokens_u_appearance = teacher_tokens.chunk(2)
            stability_u = relation_stability_map(
                tokens_u_weak, tokens_u_appearance, patch_grid,
                args.relation_temperature)

            student_u_logits = student(appearance_u)
            token_capture.clear()  # Tokens from this forward are not needed.
            should_log = iteration % log_interval == 0
            loss_u, rctc_diagnostics = rctc_loss(
                student_u_logits,
                teacher_weak_logits,
                teacher_appearance_logits,
                stability_u,
                include_distribution_diagnostics=should_log,
            )
            if rampup_iters == 0:
                lambda_u = args.lambda_u_max
            else:
                lambda_u = args.lambda_u_max * min(
                    1.0, global_step / rampup_iters)
            loss = loss_sup + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_teacher(student, teacher, args.decay)

            total_loss_meter.update(loss.item())
            supervised_loss_meter.update(loss_sup.item())
            unsupervised_loss_meter.update(
                rctc_diagnostics["weighted_rctc"])
            weight_meter.update(rctc_diagnostics["weight"])
            raw_rctc_meter.update(rctc_diagnostics["raw_rctc"])
            structural_meter.update(rctc_diagnostics["structural"])
            semantic_meter.update(rctc_diagnostics["semantic"])
            confidence_meter.update(rctc_diagnostics["confidence"])
            teacher_disagreement_meter.update(
                rctc_diagnostics["teacher_disagreement"])
            appearance_shift_meter.update(appearance_shift_l1)
            epoch_stability_scores.extend([
                stability_l.detach().float().cpu().flatten(),
                stability_u.detach().float().cpu().flatten(),
            ])

            if should_log:
                weight_effect = (
                    unsupervised_loss_meter.avg /
                    max(raw_rctc_meter.avg, 1e-8))
                logger.info("Iters: %i, Total loss: %.3f",
                            iteration, total_loss_meter.avg)
                logger.info(
                    "DINO-Bridge: Sup: %.3f, RCTC: %.3f, RawRCTC: %.3f, "
                    "WeightedRCTC: %.3f, lambda_u: %.3f",
                    supervised_loss_meter.avg,
                    unsupervised_loss_meter.avg,
                    raw_rctc_meter.avg,
                    unsupervised_loss_meter.avg,
                    lambda_u,
                )
                logger.info(
                    "RCTC Reliability: S_mean: %.4f, V_mean: %.4f, "
                    "C_mean: %.4f, W_mean: %.4f, W_p10: %.4f, "
                    "W_p50: %.4f, W_p90: %.4f, TeacherDisagree: %.4f, "
                    "WeightEffect: %.4f",
                    structural_meter.avg,
                    semantic_meter.avg,
                    confidence_meter.avg,
                    weight_meter.avg,
                    rctc_diagnostics["weight_p10"],
                    rctc_diagnostics["weight_p50"],
                    rctc_diagnostics["weight_p90"],
                    teacher_disagreement_meter.avg,
                    weight_effect,
                )
                logger.info(
                    "RCTC Structural: S_p10: %.4f, S_p50: %.4f, "
                    "S_p90: %.4f",
                    rctc_diagnostics["structural_p10"],
                    rctc_diagnostics["structural_p50"],
                    rctc_diagnostics["structural_p90"],
                )
                logger.info("AppearanceShift_L1: %.6f",
                            appearance_shift_meter.avg)
                writer.add_scalar("train/loss_all", total_loss_meter.avg,
                                  global_step)
                writer.add_scalar("train/loss_sup",
                                  supervised_loss_meter.avg, global_step)
                writer.add_scalar("train/loss_rctc",
                                  unsupervised_loss_meter.avg, global_step)
                writer.add_scalar("train/lambda_u", lambda_u, global_step)
                writer.add_scalar("train/reliability_weight",
                                  weight_meter.avg, global_step)
                writer.add_scalar("rctc/raw_loss", raw_rctc_meter.avg,
                                  global_step)
                writer.add_scalar("rctc/weighted_loss",
                                  unsupervised_loss_meter.avg, global_step)
                writer.add_scalar("rctc/weight_effect", weight_effect,
                                  global_step)
                writer.add_scalar("rctc/structural_mean",
                                  structural_meter.avg, global_step)
                writer.add_scalar("rctc/semantic_mean",
                                  semantic_meter.avg, global_step)
                writer.add_scalar("rctc/confidence_mean",
                                  confidence_meter.avg, global_step)
                writer.add_scalar("rctc/weight_mean", weight_meter.avg,
                                  global_step)
                writer.add_scalar("rctc/weight_p10",
                                  rctc_diagnostics["weight_p10"], global_step)
                writer.add_scalar("rctc/weight_p50",
                                  rctc_diagnostics["weight_p50"], global_step)
                writer.add_scalar("rctc/weight_p90",
                                  rctc_diagnostics["weight_p90"], global_step)
                writer.add_scalar("rctc/structural_p10",
                                  rctc_diagnostics["structural_p10"],
                                  global_step)
                writer.add_scalar("rctc/structural_p50",
                                  rctc_diagnostics["structural_p50"],
                                  global_step)
                writer.add_scalar("rctc/structural_p90",
                                  rctc_diagnostics["structural_p90"],
                                  global_step)
                writer.add_scalar(
                    "rctc/teacher_disagreement",
                    teacher_disagreement_meter.avg, global_step)
                writer.add_scalar("train/appearance_shift_l1",
                                  appearance_shift_meter.avg, global_step)
                writer.add_scalar("train/rho", rho, global_step)

            global_step += 1
            optimizer.param_groups[0]["lr"] = polynomial_lr(
                cfg["lr"], global_step, total_iters)

        # Eq. (adaptive_strength): update once using detached source scores.
        all_stability = torch.cat(epoch_stability_scores)
        stability_mean = all_stability.mean().item()
        stability_quantiles = torch.quantile(
            all_stability, torch.tensor([0.10, 0.50, 0.90]))
        stability_p10, stability_p50, stability_p90 = (
            stability_quantiles.tolist())
        median_stability = stability_p50
        smoothed_stability = (
            args.rho_beta * smoothed_stability +
            (1.0 - args.rho_beta) * median_stability)
        control_error = smoothed_stability - args.rho_target
        next_rho = min(args.rho_max, max(
            0.0,
            rho + args.rho_step * control_error,
        ))
        delta_rho = next_rho - rho
        logger.info(
            "CSRV Stability: mean: %.4f, p10: %.4f, p50: %.4f, p90: %.4f",
            stability_mean, stability_p10, stability_p50, stability_p90)
        logger.info(
            "CSRV Controller: SmoothedStability: %.4f, rho_target: %.4f, "
            "ControlError: %.4f, rho: %.4f -> %.4f, DeltaRho: %.4f",
            smoothed_stability, args.rho_target, control_error,
            rho, next_rho, delta_rho)
        writer.add_scalar("csrv/stability_mean", stability_mean, epoch)
        writer.add_scalar("csrv/stability_p10", stability_p10, epoch)
        writer.add_scalar("csrv/stability_p50", stability_p50, epoch)
        writer.add_scalar("csrv/stability_p90", stability_p90, epoch)
        writer.add_scalar("csrv/median_stability", median_stability, epoch)
        writer.add_scalar("csrv/smoothed_stability",
                          smoothed_stability, epoch)
        writer.add_scalar("csrv/rho_target", args.rho_target, epoch)
        writer.add_scalar("csrv/control_error", control_error, epoch)
        writer.add_scalar("csrv/rho_current", rho, epoch)
        writer.add_scalar("csrv/rho_next", next_rho, epoch)
        writer.add_scalar("csrv/delta_rho", delta_rho, epoch)
        rho = next_rho

        checkpoint = make_checkpoint(
            student, teacher, optimizer, epoch,
            global_step, rho, smoothed_stability, method_config)
        torch.save(checkpoint, latest_path)
        if epoch == cfg["epochs"] - 1:
            # Keep latest.pth for compatible resume semantics and also expose
            # the explicitly requested final/last checkpoint.
            torch.save(checkpoint,
                       os.path.join(args.save_path, "last.pth"))
        del checkpoint

    writer.close()
    token_capture.close()

    if completed_epoch >= cfg["epochs"] - 1:
        logger.info("************ Skip trained checkpoint at epoch %i\n",
                    completed_epoch)
        if not os.path.exists(os.path.join(args.save_path, "last.pth")):
            checkpoint = torch.load(latest_path, map_location="cpu")
            torch.save(checkpoint, os.path.join(args.save_path, "last.pth"))
            del checkpoint


if __name__ == "__main__":
    main()
