import torch
import torch.nn.functional as F
from torch import nn
from typing import Union, List, Tuple
import sys
import os
from typing import Union, Type, List, Tuple
import pydoc

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
import math

# Add DINOv3 imports
from model.dinounet.dinov3.models.vision_transformer import DinoVisionTransformer
from model.dinounet.network_architecture.dinov3_adapter import DINOv3_Adapter
from model.dinounet.dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16


DINOv3_MODEL_FACTORIES = {
    "dinounet_s": dinov3_vits16,
    "dinounet_b": dinov3_vitb16,
    "dinounet_l": dinov3_vitl16,
    "dinounet_7b": dinov3_vit7b16,
}

DINOv3_INTERACTION_INDEXES = {
    "dinounet_s": [2, 5, 8, 11],
    "dinounet_b": [2, 5, 8, 11],
    "dinounet_l": [4, 11, 17, 23],
    "dinounet_7b": [9, 19, 29, 39],
}

DINOv3_MODEL_INFO = {
    "dinounet_s": {"embed_dim": 384, "depth": 12, "num_heads": 6, "params": "~22M"},
    "dinounet_b": {"embed_dim": 768, "depth": 12, "num_heads": 12, "params": "~86M"},
    "dinounet_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "params": "~300M"},
    "dinounet_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "params": "~7B"},
}


def load_dinov3_model(model_name: str, pretrained_path: str = None) -> DinoVisionTransformer:
    """Load DINOv3 model with pretrained weights"""
    
    if model_name not in DINOv3_MODEL_FACTORIES:
        supported_models = list(DINOv3_MODEL_FACTORIES.keys())
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_models}")
    
    model_factory = DINOv3_MODEL_FACTORIES[model_name]
    
    # If pretrained path is provided, use custom weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading custom pretrained weights from {pretrained_path}")
        # Create model first
        model = model_factory(pretrained=False)
        # Load custom weights
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded custom pretrained weights")
    else:
        # Use default pretrained weights
        print(f"Loading default pretrained weights for {model_name}")
        model = model_factory(pretrained=True)
        print("Successfully loaded default pretrained weights")

    return model


def load_dinov3_vits16(pretrained_path: str) -> DinoVisionTransformer:
    """Load DINOv3 ViT-S/16 model with pretrained weights (backward compatibility)"""
    return load_dinov3_model("dinov3_vits16", pretrained_path)


class DINOv3EncoderAdapter(nn.Module):
    """
    Adapter to make DINOv3_Adapter compatible with PlainConvEncoder interface
    """
    def __init__(self,
                 dinov3_adapter: DINOv3_Adapter,
                 target_channels: List[int],
                 conv_op: Type[_ConvNd] = nn.Conv2d,
                 norm_op: Union[None, Type[nn.Module]] = nn.BatchNorm2d,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = nn.ReLU,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = False):
        super().__init__()

        self.dinov3_adapter = dinov3_adapter
        self.target_channels = target_channels

        # Store encoder properties for compatibility with UNetDecoder
        self.conv_op = conv_op
        self.norm_op = norm_op if norm_op is not None else nn.BatchNorm2d
        self.norm_op_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {}
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs if dropout_op_kwargs is not None else {}
        self.nonlin = nonlin if nonlin is not None else nn.ReLU
        self.nonlin_kwargs = nonlin_kwargs if nonlin_kwargs is not None else {'inplace': True}
        self.conv_bias = conv_bias

        # DINOv3_Adapter outputs embed_dim features, need to project to target channels
        dinov3_feature_dim = self.dinov3_adapter.backbone.embed_dim

        # Create projection layers for each scale
        self.projections = nn.ModuleList()
        for target_ch in target_channels:
            if target_ch != dinov3_feature_dim:
                proj = nn.Sequential(
                    conv_op(dinov3_feature_dim, target_ch, kernel_size=1, bias=conv_bias),
                    self.norm_op(target_ch, **self.norm_op_kwargs),
                    self.nonlin(**self.nonlin_kwargs)
                )
            else:
                proj = nn.Identity()
            self.projections.append(proj)

        # Set output channels and strides for compatibility
        self.output_channels = target_channels

        # Define strides and kernel_sizes for compatibility with UNetDecoder
        # These should match the downsampling factor of the features from DINOv3
        # The features are at 1/4, 1/8, 1/16, 1/32 scales, so each stage is a 2x downsample
        self.strides = [[2, 2]] * len(target_channels)
        self.kernel_sizes = [[3, 3]] * len(target_channels) # Dummy value, not used by decoder for transpconv
        # DINOv3_Adapter outputs at scales: 1/4, 1/8, 1/16, 1/32

    def forward(self, x):
        """
        Forward pass that returns skips in PlainConvEncoder format
        """
        
        B, C, H, W = x.shape

        # Handle single channel input: DINOv3 requires 3-channel input
        if C == 1:
            # Repeat single channel to 3 channels
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            # If not 1 channel and not 3 channels, need adaptation
            if C < 3:
                # Less than 3 channels, repeat to 3 channels
                x = x.repeat(1, 3 // C + (1 if 3 % C != 0 else 0), 1, 1)[:, :3, :, :]
            else:
                # More than 3 channels, take first 3 channels
                x = x[:, :3, :, :]
                
        # Get features from DINOv3_Adapter
        features_dict = self.dinov3_adapter(x)

        # Convert to list format and apply projections
        skips = []
        feature_keys = ["1", "2", "3", "4"]  # From highest to lowest resolution
        
        # DINOv3_Adapter outputs at different scales, we need to upsample them to match input resolution
        # Scale factors: 1/4, 1/8, 1/16, 1/32        
        # Debug: print input and feature shapes
        if hasattr(self, '_debug') and self._debug:
            print(f"Input shape: {x.shape}")
            for key in feature_keys:
                print(f"Feature {key} shape: {features_dict[key].shape}")
        
        for i, key in enumerate(feature_keys):
            feature = features_dict[key]
            projected_feature = self.projections[i](feature)
            
            # Upsample to match the expected resolution for this skip level
            # For a 4-stage UNet, we expect: [H, H/2, H/4, H/8]
            target_H = H // (2 ** i)
            target_W = W // (2 ** i)
            
            if projected_feature.shape[2] != target_H or projected_feature.shape[3] != target_W:
                projected_feature = F.interpolate(
                    projected_feature, 
                    size=(target_H, target_W), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Debug: print target and actual shapes
            if hasattr(self, '_debug') and self._debug:
                print(f"Skip {i}: target={target_H}x{target_W}, actual={projected_feature.shape[2]}x{projected_feature.shape[3]}")
            
            skips.append(projected_feature)

        return skips

    def compute_conv_feature_map_size(self, input_size):
        """Dummy implementation for compatibility"""
        return 0  # This will be overridden by the decoder's computation

    def enable_debug(self, enabled=True):
        """Enable debug mode to print feature shapes"""
        self._debug = enabled


# ========================= Adapter Variants and Helper Blocks ========================= #

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 bias: bool = False, norm: Type[nn.Module] = nn.BatchNorm2d, act: Type[nn.Module] = nn.ReLU,
                 norm_kwargs: dict = None, act_kwargs: dict = None):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {'inplace': True} if act_kwargs is None else act_kwargs
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = norm(out_ch, **norm_kwargs) if norm is not None else nn.Identity()
        self.act = act(**act_kwargs) if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LearnableUpsampleBlock(nn.Module):
    """Lightweight learnable upsampling (transpose conv) as an alternative to bilinear."""
    def __init__(self, channels: int):
        super().__init__()
        self.up2 = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, bias=True)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        out = x
        # Upsample by factors of 2 until we reach or exceed target, then final bilinear to exact size
        while h * 2 <= target_size[0] and w * 2 <= target_size[1]:
            out = self.up2(out)
            h, w = out.shape[2], out.shape[3]
        if (h, w) != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out


class GatedChannelSelection(nn.Module):
    """Soft gating before projection to suppress redundant channels."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x)
        return x * w



class DualBranchSharedBasis(nn.Module):
    """
    Dual-branch shared basis module.
    - Shared Branch: Captures cross-scale common information.
    - Specific Branch: Captures scale-specific information.
    """
    def __init__(self, in_ch: int, shared_rank: int, specific_rank: int, num_scales: int, bias: bool = False):
        """
        Args:
            in_ch: Input channel count (from DINOv3).
            shared_rank: Output channel count of shared branch.
            specific_rank: Output channel count of specific branch.
            num_scales: Number of scales (e.g., 4 scales).
            bias: Whether to use bias.
        """
        super().__init__()
        self.num_scales = num_scales

        # 1. Shared branch: a 1x1 convolution shared across all scales
        self.shared_branch = nn.Conv2d(in_ch, shared_rank, kernel_size=1, bias=bias)

        # 2. Specific branch: a ModuleList creating independent 1x1 convolutions for each scale
        self.specific_branches = nn.ModuleList([
            nn.Conv2d(in_ch, specific_rank, kernel_size=1, bias=bias) for _ in range(num_scales)
        ])

    def forward(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """
        Args:
            x: Input feature map.
            scale_idx: Current scale index (0, 1, 2, ...), used to select the correct specific branch.
        
        Returns:
            Fused feature map.
        """
        # Compute shared features
        z_shared = self.shared_branch(x)

        # Compute specific features
        # Select corresponding specific branch based on scale_idx
        z_specific = self.specific_branches[scale_idx](x)

        # Concatenate along channel dimension to fuse both types of information
        z_combined = torch.cat([z_shared, z_specific], dim=1)
        
        return z_combined

class SharedBasisProjector(nn.Module):
    """Low-rank shared basis across scales: x -> U (shared) -> V_s (per-scale) -> target."""
    def __init__(self, in_ch: int, rank: int, out_ch_list: List[int],
                 norm: Type[nn.Module] = nn.BatchNorm2d, act: Type[nn.Module] = nn.ReLU,
                 norm_kwargs: dict = None, act_kwargs: dict = None, bias: bool = False):
        super().__init__()
        norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        act_kwargs = {'inplace': True} if act_kwargs is None else act_kwargs
        self.shared = nn.Conv2d(in_ch, rank, kernel_size=1, bias=bias)
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(rank, oc, kernel_size=1, bias=bias),
                norm(oc, **norm_kwargs) if norm is not None else nn.Identity(),
                act(**act_kwargs) if act is not None else nn.Identity(),
            ) for oc in out_ch_list
        ])

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for i, x in enumerate(x_list):
            z = self.shared(x)
            out.append(self.projs[i](z))
        return out


class DINOv3EncoderAdapter(nn.Module):
    def __init__(self,
                 dinov3_adapter: DINOv3_Adapter,
                 target_channels: List[int],
                 adapter_type: str = "default",
                 rank: int = 256,
                 conv_op: Type[_ConvNd] = nn.Conv2d,
                 norm_op: Union[None, Type[nn.Module]] = nn.BatchNorm2d,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = nn.ReLU,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = False):
        super().__init__()
        self.dinov3_adapter = dinov3_adapter
        self.target_channels = target_channels
        self.conv_op = conv_op
        self.norm_op = norm_op if norm_op is not None else nn.BatchNorm2d
        self.norm_op_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {}
        self.nonlin = nonlin if nonlin is not None else nn.ReLU
        self.nonlin_kwargs = nonlin_kwargs if nonlin_kwargs is not None else {'inplace': True}
        self.conv_bias = conv_bias

        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs

        in_ch = self.dinov3_adapter.backbone.embed_dim

        """self.fapm = FAPM(in_ch, rank, target_channels,
            norm=self.norm_op, act=self.nonlin,
            norm_kwargs=self.norm_op_kwargs,
            act_kwargs=self.nonlin_kwargs,
            bias=conv_bias)"""
        # Independently project each DINOv3 feature scale with a plain 1x1
        # convolution. This intentionally replaces the former FAPM (shared and
        # scale-specific bases, FiLM, depthwise refinement, SE, and residuals).
        self.projections = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=conv_bias)
            for out_ch in target_channels
        ])
        
        # Learnable upsampling for spatial alignment
        self.ups = nn.ModuleList()
        for oc in target_channels:
            self.ups.append(LearnableUpsampleBlock(oc))

        self.output_channels = target_channels
        self.strides = [[2, 2]] * len(target_channels)
        self.kernel_sizes = [[3, 3]] * len(target_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            if C < 3:
                x = x.repeat(1, 3 // C + (1 if 3 % C != 0 else 0), 1, 1)[:, :3, :, :]
            else:
                x = x[:, :3, :, :]
        feats = self.dinov3_adapter(x)
        keys = ["1", "2", "3", "4"]
        x_list = [feats[k] for k in keys]
        
        # Apply independent 1x1 channel projections at each feature scale.
        ys = [projection(feature)
              for projection, feature in zip(self.projections, x_list)]
        
        # Apply learnable upsampling
        skips = []
        for i, y in enumerate(ys):
            target = (H // (2 ** i), W // (2 ** i))
            y = self.ups[i](y, target)
            skips.append(y)
        return skips

    def compute_conv_feature_map_size(self, input_size):
        return 0


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: PlainConvEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


class DinoUNet(nn.Module):
    """
    U-Net with DINOv3_Adapter as encoder, compatible with PlainConvUNet interface
    """
    def __init__(self,
                 network_config: dict = None,
                 input_channels: int = None,
                 num_classes: int = None,
                 dinov3_pretrained_path: str = "dinounet/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                 dinov3_model_name: str = "dinov3_vits16",
                 adapter_type: str = "default",
                 # 原始参数（向后兼容）
                 n_stages: int = None,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]] = None,
                 conv_op: Type[_ConvNd] = None,
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]] = None,
                 strides: Union[int, List[int], Tuple[int, ...]] = None,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]] = None,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]] = None,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()

        # If network_config is provided, parse parameters from configuration
        if network_config is not None:
            arch = network_config['architecture']

            # Parse string-type operations to actual classes
            def _resolve_op(op_str):
                if op_str is None:
                    return None
                if isinstance(op_str, str):
                    return pydoc.locate(op_str)
                return op_str

            # Extract parameters from configuration
            input_channels = input_channels or 3  # DINOv3 requires 3-channel input
            self.adapter_type = adapter_type
            num_classes = num_classes or 2  # Default value
            n_stages = arch['n_stages']
            features_per_stage = arch['features_per_stage']
            conv_op = _resolve_op(arch['conv_op'])
            kernel_sizes = arch['kernel_sizes']
            strides = arch['strides']
            n_conv_per_stage = arch['n_conv_per_stage']
            n_conv_per_stage_decoder = arch['n_conv_per_stage_decoder']
            conv_bias = arch.get('conv_bias', False)
            norm_op = _resolve_op(arch['norm_op'])
            norm_op_kwargs = arch.get('norm_op_kwargs', {})
            dropout_op = _resolve_op(arch['dropout_op'])
            dropout_op_kwargs = arch.get('dropout_op_kwargs', {})
            nonlin = _resolve_op(arch['nonlin'])
            nonlin_kwargs = arch.get('nonlin_kwargs', {})
            deep_supervision = arch.get('deep_supervision', False)
            nonlin_first = arch.get('nonlin_first', False)

        # Standalone construction (used by model.factory) has no nnU-Net plan.
        # Supply the minimal 2D U-Net decoder configuration explicitly.
        if n_stages is None:
            n_stages = 4
        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256]
        if conv_op is None:
            conv_op = nn.Conv2d
        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * n_stages
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)

        # Validate parameters
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # Ensure we have 4 stages to match DINOv3_Adapter output
        if n_stages != 4:
            print(f"Warning: DINOv3_Adapter outputs 4 scales, but n_stages={n_stages}. Adjusting to 4.")
            n_stages = 4
            if isinstance(features_per_stage, int):
                features_per_stage = [features_per_stage * (2**i) for i in range(4)]
            elif len(features_per_stage) != 4:
                # Adjust features_per_stage to 4 stages
                base_features = features_per_stage[0] if features_per_stage else 32
                features_per_stage = [base_features * (2**i) for i in range(4)]

        # Create DINOv3 encoder
        self.encoder = self._create_dinov3_encoder(
            dinov3_pretrained_path,
            dinov3_model_name,
            features_per_stage,
            conv_op, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, conv_bias,
            adapter_type
        )

        # Create decoder
        self.decoder = UNetDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
            nonlin_first=nonlin_first
        )

    def _create_dinov3_encoder(self, pretrained_path, model_name, features_per_stage,
                              conv_op, norm_op, norm_op_kwargs,
                              dropout_op, dropout_op_kwargs,
                              nonlin, nonlin_kwargs, conv_bias, adapter_type="default"):
        """Create DINOv3 encoder"""

        # Get model information
        if model_name not in DINOv3_MODEL_INFO:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = DINOv3_MODEL_INFO[model_name]
        interaction_indexes = DINOv3_INTERACTION_INDEXES[model_name]
        
        print(f"🔧 Creating DINOv3 encoder: {model_name}")
        print(f"   Embedding dimension: {model_info['embed_dim']}")
        print(f"   Model depth: {model_info['depth']}")
        print(f"   Number of attention heads: {model_info['num_heads']}")
        print(f"   Parameter count: {model_info['params']}")
        print(f"   Interaction layer indices: {interaction_indexes}")
        
        # Load DINOv3 backbone
        dinov3_backbone = load_dinov3_model(model_name, pretrained_path)
        
        # Create DINOv3_Adapter using correct interaction layer indices
        dinov3_adapter = DINOv3_Adapter(
            backbone=dinov3_backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.3,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=True,
        )

        encoder_adapter = DINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=features_per_stage,
            conv_op=conv_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias
        )

        return encoder_adapter

    def forward(self, x):
        skips = self.encoder(x)
        
        # Debug: print skip shapes
        if hasattr(self, '_debug') and self._debug:
            print(f"Encoder output shapes:")
            for i, skip in enumerate(skips):
                print(f"  Skip {i}: {skip.shape}")
        
        output = self.decoder(skips)
        
        # Debug: print final output shape
        if hasattr(self, '_debug') and self._debug:
            print(f"Final output shape: {output.shape}")
            if isinstance(output, list):
                for i, out in enumerate(output):
                    print(f"  Output {i}: {out.shape}")
        
        return output

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    @classmethod
    def from_config(cls, network_config: dict, input_channels: int, num_classes: int,
                   dinov3_pretrained_path: str = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                   dinov3_model_name: str = "dinov3_vits16"):
        """
        Create DinoUNet instance from network configuration dictionary
        """
        return cls(
            network_config=network_config,
            input_channels=input_channels,
            num_classes=num_classes,
            dinov3_pretrained_path=dinov3_pretrained_path,
            dinov3_model_name=dinov3_model_name,
        )
