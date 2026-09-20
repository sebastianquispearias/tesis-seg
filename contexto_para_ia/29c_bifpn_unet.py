"""
Kim-inspired BiFPN-U-Net(T) adaptation for binary segmentation.

Based on Kim et al. 2021 "Hyoid Bone Tracking in a Videofluoroscopic
Swallowing Study Using a Deep-Learning-Based Segmentation Network"
(Diagnostics 2021).

This is an adaptation that preserves the main architectural ideas:
- VGG16 encoder (no BN, no pretrained weights)
- One-round BiFPN with fast normalized fusion
- Bottleneck Transformer (MHSA) on P5
- U-Net-style decoder with skip connections from final BiFPN outputs

Not an exact reproduction — some implementation details not specified
in the paper are filled in with standard choices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Encoder: VGG16 (no batch norm, random init)
# =====================================================================

def _vgg16_block(in_ch, out_ch, n_convs):
    """VGG-style conv block: n_convs x (Conv3x3 + ReLU)."""
    layers = []
    for i in range(n_convs):
        layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class VGG16Encoder(nn.Module):
    """
    VGG16 encoder producing 5 feature maps P1-P5.
    No batch normalization, no pretrained weights (per Kim 2021).

    Output channels: P1=64, P2=128, P3=256, P4=512, P5=512
    Output scales:   H/2,   H/4,   H/8,   H/16,  H/32
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = _vgg16_block(in_channels, 64, 2)
        self.block2 = _vgg16_block(64, 128, 2)
        self.block3 = _vgg16_block(128, 256, 3)
        self.block4 = _vgg16_block(256, 512, 3)
        self.block5 = _vgg16_block(512, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        p1 = self.pool(self.block1(x))   # (B, 64,  H/2, W/2)
        p2 = self.pool(self.block2(p1))   # (B, 128, H/4, W/4)
        p3 = self.pool(self.block3(p2))   # (B, 256, H/8, W/8)
        p4 = self.pool(self.block4(p3))   # (B, 512, H/16, W/16)
        p5 = self.pool(self.block5(p4))   # (B, 512, H/32, W/32)
        return [p1, p2, p3, p4, p5]


# =====================================================================
# BiFPN: One-round bidirectional feature pyramid with fast fusion
# =====================================================================

class FastNormalizedFusion(nn.Module):
    """
    Fast normalized fusion from Tan et al. 2020 (EfficientDet).
    O = sum(w_i * I_i) / (sum(w_j) + eps), where w_i >= 0 (ReLU).
    """

    def __init__(self, n_inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs))
        self.eps = 1e-4

    def forward(self, inputs):
        w = F.relu(self.weights)
        w_sum = w.sum() + self.eps
        out = sum(w[i] * inputs[i] for i in range(len(inputs)))
        return out / w_sum


class BiFPN(nn.Module):
    """
    One round of BiFPN: top-down pass (P5->P1), then bottom-up pass (P1->P5).

    All features are first projected to bifpn_channels via 1x1 conv.
    Fusion uses fast normalized fusion with learnable weights.
    """

    def __init__(self, encoder_channels=(64, 128, 256, 512, 512), bifpn_channels=128):
        super().__init__()
        self.n_levels = len(encoder_channels)

        # 1x1 projections: encoder channels -> bifpn_channels
        self.input_projs = nn.ModuleList([
            nn.Conv2d(ch, bifpn_channels, 1) for ch in encoder_channels
        ])

        # Top-down fusion nodes (P4_td, P3_td, P2_td, P1_td)
        # Each fuses 2 inputs: higher-level (upsampled) + same-level input
        self.td_fusions = nn.ModuleList([
            FastNormalizedFusion(2) for _ in range(self.n_levels - 1)
        ])
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1),
                nn.BatchNorm2d(bifpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(self.n_levels - 1)
        ])

        # Bottom-up fusion nodes (P2_out, P3_out, P4_out, P5_out)
        # P2..P4: 3 inputs (original proj, td output, lower-level bu)
        # P5: 2 inputs (original proj, lower-level bu)
        self.bu_fusions = nn.ModuleList()
        self.bu_convs = nn.ModuleList()
        for i in range(self.n_levels - 1):
            level = i + 1  # bu goes from level 1 to level 4 (0-indexed)
            n_inputs = 2 if level == self.n_levels - 1 else 3
            self.bu_fusions.append(FastNormalizedFusion(n_inputs))
            self.bu_convs.append(nn.Sequential(
                nn.Conv2d(bifpn_channels, bifpn_channels, 3, padding=1),
                nn.BatchNorm2d(bifpn_channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, features):
        """
        Args:
            features: list of 5 tensors [P1, P2, P3, P4, P5] from encoder
        Returns:
            list of 5 tensors at same spatial resolutions, all bifpn_channels
        """
        # Project all features to bifpn_channels
        projected = [self.input_projs[i](features[i]) for i in range(self.n_levels)]

        # --- Top-down pass: from P5 down to P1 ---
        td = [None] * self.n_levels
        td[self.n_levels - 1] = projected[self.n_levels - 1]  # P5 unchanged

        for i in range(self.n_levels - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=projected[i].shape[2:],
                               mode='bilinear', align_corners=False)
            fused = self.td_fusions[i]([projected[i], up])
            td[i] = self.td_convs[i](fused)

        # --- Bottom-up pass: from P1 up to P5 ---
        bu = [None] * self.n_levels
        bu[0] = td[0]  # P1 from top-down

        for idx in range(self.n_levels - 1):
            level = idx + 1
            down = F.interpolate(bu[level - 1], size=td[level].shape[2:],
                                 mode='bilinear', align_corners=False)
            if level == self.n_levels - 1:
                # P5_out: fuse original projection + bottom-up from below
                fused = self.bu_fusions[idx]([projected[level], down])
            else:
                # P2..P4_out: fuse original projection + td output + bu from below
                fused = self.bu_fusions[idx]([projected[level], td[level], down])
            bu[level] = self.bu_convs[idx](fused)

        return bu


# =====================================================================
# Bottleneck Transformer (BOT) on P5
# =====================================================================

class BottleneckTransformer(nn.Module):
    """
    Bottleneck Transformer applied to P5 (smallest feature map).
    Conv1x1 (reduce) -> MHSA -> Conv1x1 (restore) + residual.

    Simplification vs paper: uses standard nn.MultiheadAttention without
    relative position encoding. The feature map is flattened to a sequence
    for attention, then reshaped back.
    """

    def __init__(self, channels, bottleneck_channels=None, n_heads=4):
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = channels // 2

        self.reduce = nn.Conv2d(channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=bottleneck_channels, num_heads=n_heads, batch_first=True
        )
        self.expand = nn.Conv2d(bottleneck_channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        # Reduce channels
        x = self.relu(self.bn1(self.reduce(x)))

        # Flatten spatial dims -> (B, H*W, C_bottleneck)
        x = x.flatten(2).permute(0, 2, 1)

        # Self-attention
        x, _ = self.attn(x, x, x)

        # Reshape back -> (B, C_bottleneck, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # Expand channels + residual
        x = self.bn2(self.expand(x))
        x = self.relu(x + residual)
        return x


# =====================================================================
# Decoder: U-Net style with skip connections from BiFPN outputs
# =====================================================================

class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers (standard U-Net decoder block)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoder(nn.Module):
    """
    U-Net-style decoder: upsample + concat skip + conv block at each level.
    Goes from P5 -> P4 -> P3 -> P2 -> P1 -> full resolution output.
    """

    def __init__(self, bifpn_channels=128, n_classes=1):
        super().__init__()
        ch = bifpn_channels

        # Each level: upsampled features + skip (both ch) -> 2*ch -> ch
        self.dec4 = ConvBlock(ch * 2, ch)
        self.dec3 = ConvBlock(ch * 2, ch)
        self.dec2 = ConvBlock(ch * 2, ch)
        self.dec1 = ConvBlock(ch * 2, ch)

        # Final: upsample from P1 (H/2) to full resolution, then head
        self.final_conv = ConvBlock(ch, ch // 2)
        self.head = nn.Conv2d(ch // 2, n_classes, 1)

    def forward(self, bifpn_features, input_size):
        """
        Args:
            bifpn_features: [P1, P2, P3, P4, P5] from BiFPN, all bifpn_channels
            input_size: (H, W) of the original input for final upsample
        """
        p1, p2, p3, p4, p5 = bifpn_features

        x = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False)
        x = self.dec4(torch.cat([x, p4], dim=1))

        x = F.interpolate(x, size=p3.shape[2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, p3], dim=1))

        x = F.interpolate(x, size=p2.shape[2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, p2], dim=1))

        x = F.interpolate(x, size=p1.shape[2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, p1], dim=1))

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        x = self.head(x)

        return x  # raw logits, no sigmoid


# =====================================================================
# Full model: BiFPN-U-Net(T)
# =====================================================================

class BiFPNUNet(nn.Module):
    """
    Kim-inspired BiFPN-U-Net(T) adaptation for binary segmentation.
    Based on Kim et al. 2021 (Diagnostics).

    Architecture:
        VGG16 encoder (no BN, random init) -> BiFPN (one round) ->
        Bottleneck Transformer on P5 -> U-Net decoder with BiFPN skips.

    Args:
        in_channels: input image channels (default 3)
        n_classes: output segmentation classes (default 1)
        encoder_channels: VGG16 block output channels
        bifpn_channels: internal BiFPN channel width
        use_bot: whether to apply Bottleneck Transformer on P5
        bot_heads: number of attention heads in BOT
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        encoder_channels=(64, 128, 256, 512, 512),
        bifpn_channels=128,
        use_bot=True,
        bot_heads=4,
        pretrained=False,
    ):
        super().__init__()
        self.encoder = VGG16Encoder(in_channels)
        if pretrained:
            self._load_pretrained_encoder()
        self.bifpn = BiFPN(encoder_channels, bifpn_channels)
        self.use_bot = use_bot
        if use_bot:
            self.bot = BottleneckTransformer(
                channels=bifpn_channels, n_heads=bot_heads
            )
        self.decoder = UNetDecoder(bifpn_channels, n_classes)

    def _load_pretrained_encoder(self):
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        mapping = {
            'features.0':  'block1.0', 'features.2':  'block1.2',
            'features.5':  'block2.0', 'features.7':  'block2.2',
            'features.10': 'block3.0', 'features.12': 'block3.2',
            'features.14': 'block3.4',
            'features.17': 'block4.0', 'features.19': 'block4.2',
            'features.21': 'block4.4',
            'features.24': 'block5.0', 'features.26': 'block5.2',
            'features.28': 'block5.4',
        }
        src_sd = vgg.features.state_dict()
        tgt_sd = {}
        for src_prefix, tgt_prefix in mapping.items():
            tgt_sd[f'{tgt_prefix}.weight'] = src_sd[f'{src_prefix}.weight']
            tgt_sd[f'{tgt_prefix}.bias'] = src_sd[f'{src_prefix}.bias']
        self.encoder.load_state_dict(tgt_sd, strict=True)
        print(f'[BiFPN-UNet] Loaded ImageNet VGG16 weights ({len(tgt_sd)} tensors), all keys matched correctly')
        del vgg

    def forward(self, x):
        # Handle single-channel input by repeating to 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        input_size = x.shape[2:]

        features = self.encoder(x)
        bifpn_out = self.bifpn(features)

        if self.use_bot:
            bifpn_out[4] = self.bot(bifpn_out[4])

        logits = self.decoder(bifpn_out, input_size)
        return logits  # (B, n_classes, H, W) raw logits
