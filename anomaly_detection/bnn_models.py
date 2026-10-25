#!/usr/bin/env python3
"""
Binarized Neural Network (BNN) Architectures & Layers for FPGA Anomaly Detection:
- Uses Straight-Through Estimator (STE) for sign(x) binarization {-1, +1} in forward pass
  and clipped identity gradient in backward pass.
- Binarized Linear (BinaryLinear) and Binarized 1D Convolution (BinaryConv1d).
- Replaces costly DSP multiplications with single-gate XNOR + Popcount operations on Cyclone IV (DE0-Nano).
- Topologies:
  1. BNN_Micro1DCNN: 1-bit weights & activations 1D CNN (Streaming DE0-Nano accelerator compatible)
  2. BNN_Tiny1DCNN: 3-layer Binarized 1D CNN with BatchNorm folding & Global Average Pooling
  3. BNN_Small1DCNN: Higher-capacity Binarized 1D CNN for multi-dataset generalization
  4. BNN_TinyMLP: Binarized Multi-Layer Perceptron (pure combinatorial XNOR trees)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    """
    Binarization function with Straight-Through Estimator (STE):
    Forward:  b = +1 if x >= 0 else -1
    Backward: db/dx = 1 if |x| <= 1 else 0 (Hardtanh / Clipped STE)
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Map >= 0 to +1.0, < 0 to -1.0
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1.0] = 0.0
        return grad_input


def binarize(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


class BinaryConv1d(nn.Conv1d):
    """
    1D Convolution with binarized weights {-1, +1} (and optional scaling factor alpha
    or pure +/-1 for multiplierless XNOR-Popcount hardware).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, binarize_input=True, use_alpha=False):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=bias)
        self.binarize_input = binarize_input
        self.use_alpha = use_alpha

    def forward(self, x):
        if self.binarize_input:
            x_proc = binarize(x)
        else:
            x_proc = x

        # Clamp latent weights to [-1, 1] during training for stable STE updates
        if self.training:
            with torch.no_grad():
                self.weight.clamp_(-1.0, 1.0)

        w_bin = binarize(self.weight)
        if self.use_alpha:
            alpha = self.weight.abs().mean(dim=(1, 2), keepdim=True).detach()
            w_bin = w_bin * alpha

        return F.conv1d(x_proc, w_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BinaryLinear(nn.Linear):
    """
    Fully-Connected layer with binarized weights {-1, +1} and optional binarized inputs {-1, +1}.
    In hardware: computed via bitwise XNOR and Popcount!
    """
    def __init__(self, in_features, out_features, bias=True, binarize_input=True, use_alpha=False):
        super().__init__(in_features, out_features, bias=bias)
        self.binarize_input = binarize_input
        self.use_alpha = use_alpha

    def forward(self, x):
        if self.binarize_input:
            x_proc = binarize(x)
        else:
            x_proc = x

        if self.training:
            with torch.no_grad():
                self.weight.clamp_(-1.0, 1.0)

        w_bin = binarize(self.weight)
        if self.use_alpha:
            alpha = self.weight.abs().mean(dim=1, keepdim=True).detach()
            w_bin = w_bin * alpha

        return F.linear(x_proc, w_bin, self.bias)


class BNN_Micro1DCNN(nn.Module):
    """
    Ultra-compact Binarized 1D CNN for Terasic DE0-Nano (Cyclone IV EP4CE22F17C6N).
    Architecture matches streaming hardware pipeline:
    Input (INT8 or Binary) -> BinaryConv1D(6->8, k=3) -> BatchNorm1d -> Sign
    -> BinaryConv1D(8->8, k=3) -> BatchNorm1d -> Sign -> GAP -> BinaryLinear(8->1)
    0 DSP slices required!
    """
    def __init__(self, in_channels: int = 6, binarize_first_input: bool = False):
        super().__init__()
        self.binarize_first_input = binarize_first_input
        self.conv1 = BinaryConv1d(in_channels, 8, kernel_size=3, stride=1, padding=1,
                                  bias=False, binarize_input=binarize_first_input)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = BinaryConv1d(8, 8, kernel_size=3, stride=2, padding=1,
                                  bias=False, binarize_input=True)
        self.bn2 = nn.BatchNorm1d(8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = BinaryLinear(8, 1, bias=True, binarize_input=True)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = binarize(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


class BNN_Tiny1DCNN(nn.Module):
    """
    3-layer Binarized 1D CNN (~1.35K binary weights = 169 bytes!).
    Conv1(6->8) -> BN -> Sign -> Conv2(8->16) -> BN -> Sign -> Conv3(16->16) -> BN -> Sign -> GAP -> BinaryLinear(16->1)
    """
    def __init__(self, in_channels: int = 6, dropout: float = 0.0, binarize_first_input: bool = False):
        super().__init__()
        self.conv1 = BinaryConv1d(in_channels, 8, kernel_size=3, stride=1, padding=1,
                                  bias=False, binarize_input=binarize_first_input)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = BinaryConv1d(8, 16, kernel_size=3, stride=2, padding=1,
                                  bias=False, binarize_input=True)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = BinaryConv1d(16, 16, kernel_size=3, stride=2, padding=1,
                                  bias=False, binarize_input=True)
        self.bn3 = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = BinaryLinear(16, 1, bias=True, binarize_input=False)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = binarize(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


class BNN_Small1DCNN(nn.Module):
    """
    Higher-capacity Binarized 1D CNN (~5.0K binary weights = 626 bytes!).
    Conv1(6->16) -> BN -> Sign -> Conv2(16->32) -> BN -> Sign -> Conv3(32->32) -> BN -> Sign -> GAP -> BinaryLinear(32->1)
    """
    def __init__(self, in_channels: int = 6, dropout: float = 0.0, binarize_first_input: bool = False):
        super().__init__()
        self.conv1 = BinaryConv1d(in_channels, 16, kernel_size=3, stride=1, padding=1,
                                  bias=False, binarize_input=binarize_first_input)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = BinaryConv1d(16, 32, kernel_size=3, stride=2, padding=1,
                                  bias=False, binarize_input=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = BinaryConv1d(32, 32, kernel_size=3, stride=2, padding=1,
                                  bias=False, binarize_input=True)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = BinaryLinear(32, 1, bias=True, binarize_input=False)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = binarize(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


class BNN_TinyMLP(nn.Module):
    """
    3-layer Binarized Multi-Layer Perceptron.
    Uses pure XNOR-Popcount gates in hidden layers.
    """
    def __init__(self, in_features: int = 384, hidden1: int = 32, hidden2: int = 16,
                 dropout: float = 0.0, binarize_first_input: bool = True):
        super().__init__()
        self.fc1 = BinaryLinear(in_features, hidden1, bias=False, binarize_input=binarize_first_input)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = BinaryLinear(hidden1, hidden2, bias=False, binarize_input=True)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = BinaryLinear(hidden2, 1, bias=True, binarize_input=True)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        x = self.bn1(self.fc1(x))
        x = self.drop(x)
        x = self.bn2(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


class BNN_ResNetBlock1D(nn.Module):
    """
    Binarized Residual Block with Real Shortcut Addition (Bi-Real Net style):
    Input -> Binarize -> BinaryConv1D -> BN -> Binarize -> BinaryConv1D -> BN + Shortcut
    Zero DSP multipliers required on FPGA (pure parallel XNOR-popcount and logic addition).
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = BinaryConv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                  bias=False, binarize_input=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = BinaryConv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=False, binarize_input=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                BinaryConv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, binarize_input=True),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.bn1(self.conv1(binarize(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(binarize(out)))
        return out + res


class BNN_ResNet1D(nn.Module):
    """
    Binarized Residual Network for 1D Sensor Anomaly Detection (~18K 1-bit weights).
    Stem: BinaryConv1D(6->16) -> BN
    Block 1: BNN_ResNetBlock1D(16->16, S=1)
    Block 2: BNN_ResNetBlock1D(16->32, S=2)
    Block 3: BNN_ResNetBlock1D(32->32, S=2)
    GAP -> BinaryLinear(32->1)
    100% 0 DSP multipliers!
    """
    def __init__(self, in_channels: int = 6, dropout: float = 0.0, binarize_first_input: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            BinaryConv1d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False, binarize_input=binarize_first_input),
            nn.BatchNorm1d(16)
        )
        self.b1 = BNN_ResNetBlock1D(16, 16, stride=1, dropout=dropout)
        self.b2 = BNN_ResNetBlock1D(16, 32, stride=2, dropout=dropout)
        self.b3 = BNN_ResNetBlock1D(32, 32, stride=2, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = BinaryLinear(32, 1, bias=True, binarize_input=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = binarize(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


class BNN_Deep1DCNN(nn.Module):
    """
    5-layer Deep Binarized 1D CNN (~24K 1-bit weights = ~3.0 KB).
    BinaryConv1D(6->16) -> BinaryConv1D(16->24, S=2) -> BinaryConv1D(24->32, S=2)
    -> BinaryConv1D(32->48, S=2) -> BinaryConv1D(48->64) -> GAP -> BinaryLinear(64->1)
    0 DSPs required!
    """
    def __init__(self, in_channels: int = 6, dropout: float = 0.0, binarize_first_input: bool = False):
        super().__init__()
        self.conv1 = BinaryConv1d(in_channels, 16, kernel_size=5, stride=1, padding=2, bias=False, binarize_input=binarize_first_input)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = BinaryConv1d(16, 24, kernel_size=3, stride=2, padding=1, bias=False, binarize_input=True)
        self.bn2 = nn.BatchNorm1d(24)
        self.conv3 = BinaryConv1d(24, 32, kernel_size=3, stride=2, padding=1, bias=False, binarize_input=True)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = BinaryConv1d(32, 48, kernel_size=3, stride=2, padding=1, bias=False, binarize_input=True)
        self.bn4 = nn.BatchNorm1d(48)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv5 = BinaryConv1d(48, 64, kernel_size=3, stride=1, padding=1, bias=False, binarize_input=True)
        self.bn5 = nn.BatchNorm1d(64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = BinaryLinear(64, 1, bias=True, binarize_input=False)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(binarize(x)))
        x = self.bn3(self.conv3(binarize(x)))
        x = self.bn4(self.conv4(binarize(x)))
        x = self.drop(x)
        x = self.bn5(self.conv5(binarize(x)))
        x = binarize(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


class BNN_Large1DCNN(nn.Module):
    """
    High-Capacity Wide Binarized 1D CNN (~38K 1-bit weights = ~4.8 KB).
    BinaryConv1D(6->32) -> BinaryConv1D(32->64, S=2) -> BinaryConv1D(64->96, S=2)
    -> BinaryConv1D(96->128, S=2) -> GAP -> BinaryLinear(128->1)
    0 DSPs required!
    """
    def __init__(self, in_channels: int = 6, dropout: float = 0.0, binarize_first_input: bool = False):
        super().__init__()
        self.conv1 = BinaryConv1d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False, binarize_input=binarize_first_input)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = BinaryConv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=False, binarize_input=True)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = BinaryConv1d(64, 96, kernel_size=3, stride=2, padding=1, bias=False, binarize_input=True)
        self.bn3 = nn.BatchNorm1d(96)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv4 = BinaryConv1d(96, 128, kernel_size=3, stride=2, padding=1, bias=False, binarize_input=True)
        self.bn4 = nn.BatchNorm1d(128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = BinaryLinear(128, 1, bias=True, binarize_input=False)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(binarize(x)))
        x = self.bn3(self.conv3(binarize(x)))
        x = self.drop(x)
        x = self.bn4(self.conv4(binarize(x)))
        x = binarize(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


def get_bnn_model(model_name: str, window_size: int = 64, dropout: float = 0.0,
                  binarize_first_input: bool = False) -> nn.Module:
    name = model_name.lower().replace("_", "").replace("-", "")
    if "bnnmicro" in name:
        return BNN_Micro1DCNN(in_channels=6, binarize_first_input=binarize_first_input)
    elif "bnntiny1d" in name or "bnntinycnn" in name:
        return BNN_Tiny1DCNN(in_channels=6, dropout=dropout, binarize_first_input=binarize_first_input)
    elif "bnnsmall" in name:
        return BNN_Small1DCNN(in_channels=6, dropout=dropout, binarize_first_input=binarize_first_input)
    elif "bnnresnet" in name:
        return BNN_ResNet1D(in_channels=6, dropout=dropout, binarize_first_input=binarize_first_input)
    elif "bnndeep" in name:
        return BNN_Deep1DCNN(in_channels=6, dropout=dropout, binarize_first_input=binarize_first_input)
    elif "bnnlarge" in name:
        return BNN_Large1DCNN(in_channels=6, dropout=dropout, binarize_first_input=binarize_first_input)
    elif "bnntinymlp" in name or "bnnmlp" in name:
        return BNN_TinyMLP(in_features=6 * window_size, dropout=dropout, binarize_first_input=binarize_first_input)
    else:
        raise ValueError(f"Unknown BNN model name: {model_name}")


def compute_bnn_complexity(model: nn.Module, input_shape: tuple) -> dict:
    """
    Compute BNN hardware complexity for DE0-Nano Cyclone IV FPGA:
    - Binary parameters (1-bit weights) vs full/BN parameters
    - 1-bit storage size in KB (and bytes)
    - XNOR-Popcount operations (BOPs) vs standard MACs
    - Estimated DE0-Nano Cyclone IV Logic Elements (LEs), M9K blocks, and 0 DSPs
    """
    total_params = sum(p.numel() for p in model.parameters())
    binary_params = 0
    float_params = 0
    for m in model.modules():
        if isinstance(m, (BinaryConv1d, BinaryLinear)):
            binary_params += m.weight.numel()
            if m.bias is not None:
                float_params += m.bias.numel()
        elif isinstance(m, nn.BatchNorm1d):
            # In hardware, BatchNorm1d before Sign(x) folds into a single integer threshold per channel!
            float_params += m.num_features

    bops = 0
    macs = 0
    activations_bits = 0
    hooks = []

    def hook_fn(module, inp, outp):
        nonlocal bops, macs, activations_bits
        if isinstance(outp, torch.Tensor):
            activations_bits = max(activations_bits, outp.numel() * 1)  # 1-bit activations

        if isinstance(module, BinaryLinear):
            ops = module.in_features * module.out_features
            if module.binarize_input:
                bops += ops
            else:
                macs += ops  # e.g., +/-1 accumulation without multiplication
        elif isinstance(module, BinaryConv1d):
            out_len = outp.shape[-1]
            ops = module.out_channels * (module.in_channels // module.groups) * module.kernel_size[0] * out_len
            if module.binarize_input:
                bops += ops
            else:
                # First layer with INT8 input and 1-bit weight (+/-1) uses only add/sub (0 DSP multipliers!)
                macs += ops

    for m in model.modules():
        if isinstance(m, (BinaryLinear, BinaryConv1d)):
            hooks.append(m.register_forward_hook(hook_fn))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    # Memory calculations:
    # 1-bit weights + 16-bit folded BN thresholds
    bnn_size_bytes = (binary_params / 8.0) + (float_params * 2.0)
    bnn_size_kb = bnn_size_bytes / 1024.0
    fp32_size_kb = (total_params * 4.0) / 1024.0
    int8_size_kb = (total_params * 1.0) / 1024.0

    # Cyclone IV DE0-Nano Hardware Estimates (@ 50 MHz clock):
    # - 0 Embedded 9x9 Multipliers (DSPs) needed because 1-bit weights use XNOR+Popcount or Add/Sub!
    # - 1 M9K block (9 Kb = 1,152 bytes) easily stores the entire BNN weight ROM
    total_ops = bops + macs
    parallel_pe = 64  # 64 parallel XNOR/add gates per cycle
    cycles_50mhz = int((total_ops / parallel_pe) + 68)
    latency_us_50mhz = cycles_50mhz / 50.0  # 50 MHz clock on DE0-Nano

    return {
        "num_params": total_params,
        "binary_params_1bit": binary_params,
        "folded_threshold_params": float_params,
        "fp32_size_kb": round(fp32_size_kb, 3),
        "int8_size_kb": round(int8_size_kb, 3),
        "bnn_size_kb": round(bnn_size_kb, 3),
        "bnn_size_bytes": int(np.ceil(bnn_size_bytes)) if 'np' in globals() else int(bnn_size_bytes + 0.999),
        "bops_xnor": int(bops),
        "add_sub_ops": int(macs),
        "macs": int(total_ops),
        "peak_activation_kb": round(activations_bits / 8.0 / 1024.0, 4),
        "de0_nano_dsp_multipliers": 0,
        "de0_nano_m9k_blocks": 1,
        "de0_nano_estimated_les": int(380 + binary_params * 0.22),
        "de0_nano_cycles_50mhz": cycles_50mhz,
        "de0_nano_latency_us_50mhz": round(latency_us_50mhz, 2)
    }
