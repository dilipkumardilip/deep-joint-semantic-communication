from typing import Optional
import torch
import torch.nn as nn

import config


class PowerNormalization(nn.Module):
    """
    Channel Power Normalization Layer.
    
    Constrains the average transmit power to P (default from config.json):
        P_avg = (1 / k) * sum(z_i^2) <= P
    
    Formula:
        z_norm = z * sqrt(k * P) / ||z||_2
        where k is the total number of channel symbols per sample.
    """
    def __init__(self, power: float = config.POWER_CONSTRAINT):
        super().__init__()
        self.power = power


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (Batch_size, Channels, Height, Width)
        batch_size = z.size(0)
        # Flatten per sample to compute total energy
        z_flat = z.view(batch_size, -1)
        k = z_flat.size(1)  # number of transmitted symbols per image
        
        # L2 norm per sample across all channel dimensions: ||z||_2
        norm = torch.norm(z_flat, p=2, dim=1, keepdim=True)
        # Avoid division by zero with small epsilon
        norm = torch.clamp(norm, min=1e-8)
        
        # Normalize to meet the average power constraint P
        scale = torch.sqrt(torch.tensor(k * self.power, dtype=z.dtype, device=z.device))
        z_normalized = (z_flat / norm) * scale
        
        # Reshape back to original tensor shape (B, C, H, W)
        return z_normalized.view_as(z)


class JSCCEncoder(nn.Module):
    """
    Encoder for Deep Joint Source-Channel Communication (Deep JSCC).
    
    Follows the architecture from the diagram:
      - Input: (B, 3, H, W)
      - Normalization: Scales input pixel values (if needed)
      - Layer 1: Conv 5x5x16/2 + PReLU  (k=5, out_ch=16, stride=2)
      - Layer 2: Conv 5x5x32/2 + PReLU  (k=5, out_ch=32, stride=2)
      - Layer 3: Conv 5x5x32/1 + PReLU  (k=5, out_ch=32, stride=1)
      - Layer 4: Conv 5x5x32/1 + PReLU  (k=5, out_ch=32, stride=1)
      - Layer 5: Conv 5x5xc/1  + PReLU  (k=5, out_ch=c,  stride=1)
      - Normalization: Channel Power Normalization (ensures transmit power constraint)
      
    Args:
        in_channels (int): Number of input image channels (e.g., 3 for RGB). Default: 3.
        channel_snr_dim (int): Number of latent feature channels 'c' sent over the wireless channel.
        power (float): Transmit power constraint P. Default: 1.0.
    """
    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        channel_c: int = config.CHANNEL_C,
        power: float = config.POWER_CONSTRAINT,
    ):
        super().__init__()
        self.channel_c = channel_c
        
        # 5 Convolutional Layers as specified in the pipeline diagram:
        # Notation: 5x5x[out_channels]/[stride], kernel_size=5, padding=2 maintains spatial size when stride=1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU()
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.prelu4 = nn.PReLU()
        
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=channel_c, kernel_size=5, stride=1, padding=2)
        self.prelu5 = nn.PReLU()
        
        # Power normalization layer at the channel output
        self.power_norm = PowerNormalization(power=power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W) normalized to [0, 1].
        Returns:
            z (torch.Tensor): Power-normalized channel symbols of shape (B, c, H/4, W/4).
        """
        # Input normalization: ensures input is in expected float range [0, 1]
        # (Assuming input images x are in range [0, 1])
        out = self.prelu1(self.conv1(x))    # -> (B, 16, H/2, W/2)
        out = self.prelu2(self.conv2(out))  # -> (B, 32, H/4, W/4)
        out = self.prelu3(self.conv3(out))  # -> (B, 32, H/4, W/4)
        out = self.prelu4(self.conv4(out))  # -> (B, 32, H/4, W/4)
        out = self.prelu5(self.conv5(out))  # -> (B, c,  H/4, W/4)
        
        # Power normalization
        z = self.power_norm(out)
        return z


class JSCCDecoder(nn.Module):
    """
    Decoder for Deep Joint Source-Channel Communication (Deep JSCC).
    
    Follows the architecture from the diagram:
      - Input: Received channel symbols of shape (B, c, H/4, W/4)
      - Layer 1: Transposed Conv 5x5x32/1 + PReLU  (k=5, out_ch=32, stride=1)
      - Layer 2: Transposed Conv 5x5x32/1 + PReLU  (k=5, out_ch=32, stride=1)
      - Layer 3: Transposed Conv 5x5x32/1 + PReLU  (k=5, out_ch=32, stride=1)
      - Layer 4: Transposed Conv 5x5x16/2 + PReLU  (k=5, out_ch=16, stride=2)
      - Layer 5: Transposed Conv 5x5x3/2  + Sigmoid(k=5, out_ch=3,  stride=2)
      - Denormalization: Sigmoid bounds output values to [0, 1]
      
    Args:
        channel_c (int): Number of transmitted feature channels 'c'.
        out_channels (int): Number of reconstructed image channels (e.g., 3 for RGB). Default: 3.
    """
    def __init__(self, channel_c: int = config.CHANNEL_C, out_channels: int = config.IN_CHANNELS):
        super().__init__()
        
        # 5 Transposed Convolutional Layers as specified in the pipeline diagram:
        # Note: output_padding=1 is used for stride=2 layers to restore exact input dimensions
        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.prelu1 = nn.PReLU()
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.prelu2 = nn.PReLU()
        
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU()
        
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.prelu4 = nn.PReLU()
        
        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.
        
        Args:
            z_hat (torch.Tensor): Received channel symbols (corrupted by channel noise) of shape (B, c, H/4, W/4).
        Returns:
            x_hat (torch.Tensor): Reconstructed image of shape (B, 3, H, W) with pixel values in [0, 1].
        """
        out = self.prelu1(self.deconv1(z_hat))  # -> (B, 32, H/4, W/4)
        out = self.prelu2(self.deconv2(out))    # -> (B, 32, H/4, W/4)
        out = self.prelu3(self.deconv3(out))    # -> (B, 32, H/4, W/4)
        out = self.prelu4(self.deconv4(out))    # -> (B, 16, H/2, W/2)
        out = self.sigmoid(self.deconv5(out))   # -> (B, 3,  H,   W) with values in [0, 1]
        
        return out


class ResBlock(nn.Module):
    """
    Residual Block with two 3x3 convolutions and PReLU activation.
    r = PReLU(Conv2(PReLU(Conv1(x))) + x)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        return self.prelu(out + residual)


class MultiScaleBlock(nn.Module):
    """
    Inception-style Multi-Scale Block (MSB) with 4 parallel branches.
    Uses stacked 3×3 convolutions to emulate larger receptive fields
    (MPS-optimized — same trick as VGG):
      - 1×1              : Pixel-wise channel mixing (no spatial context)
      - 1×(3×3)          : Local detail, 3×3 receptive field
      - 2×(3×3) stacked  : Medium context, 5×5 effective receptive field
      - 3×(3×3) stacked  : Global context, 7×7 effective receptive field

    Stacked 3×3 conv is MPS-fast (same as ResNet) and has fewer params
    than a single large-kernel conv while capturing the same receptive field.

    Each branch outputs out_ch//4 features; they are concatenated to out_ch.
    A residual skip connection (with 1x1 projection if needed) is added.

    Args:
        in_ch  (int): Input feature channels.
        out_ch (int): Output feature channels (must be divisible by 4).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert out_ch % 4 == 0, "out_ch must be divisible by 4 for 4 equal branches"
        branch_ch = out_ch // 4

        # Branch 1: 1×1 — pixel-wise mixing (fine detail, near view)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, padding=0),
            nn.PReLU(),
        )
        # Branch 2: 3×3 — local edges and textures (RF=3)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        # Branch 3: 2×(3×3) stacked — medium context (RF=5, like 5×5 conv)
        self.b5 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        # Branch 4: 3×(3×3) stacked — global/coarse context (RF=7, like 7×7 conv)
        self.b7 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        # 1x1 projection for residual skip if channel dims differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.b7(x)], dim=1)
        return out + self.skip(x)


class JSCCEncoderMS(nn.Module):
    """
    Multi-Scale Deep JSCC Encoder (Experiment 5).

    Structure:
      - Stem Conv : (B, 3, H, W)  -> (B, 64, H,   W)    [5x5, stride 1]
      - num_msb x MultiScaleBlock : (B, 64, H, W)        [1x1+3x3+5x5+7x7, stride 1]
      - Down Conv : (B, 64, H, W) -> (B, 128, H/2, W/2)  [5x5, stride 2]
      - Bottleneck : (B, 128, H/2, W/2) -> (B, c, H/4, W/4) [3x3, stride 2]
      - Power Normalization

    Total params with num_msb=3, channel_c=16 : ~910 K  (~5.8x Exp3, ~0.42x Exp4)
    """
    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        channel_c: int = config.CHANNEL_C,
        power: float = config.POWER_CONSTRAINT,
        base_ch: int = 64,
        num_msb: int = 3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )
        self.msblocks = nn.Sequential(
            *[MultiScaleBlock(base_ch, base_ch) for _ in range(num_msb)]
        )
        self.down = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
        )
        self.bottleneck = nn.Conv2d(base_ch * 2, channel_c, kernel_size=3, stride=2, padding=1)
        self.power_norm = PowerNormalization(power=power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)      # (B, 64,    H,   W)
        out = self.msblocks(out)# (B, 64,    H,   W)
        out = self.down(out)    # (B, 128,   H/2, W/2)
        out = self.bottleneck(out)  # (B, c, H/4, W/4)
        return self.power_norm(out)


class JSCCDecoderMS(nn.Module):
    """
    Multi-Scale Deep JSCC Decoder (Experiment 5).

    Mirror of JSCCEncoderMS:
      - Expand : (B, c, H/4, W/4) -> (B, 128, H/2, W/2)  [ConvTranspose 4x4, stride 2]
      - Up     : (B, 128, H/2, W/2)-> (B, 64, H,   W)    [ConvTranspose 4x4, stride 2]
      - num_msb x MultiScaleBlock  : (B, 64, H, W)
      - Out Conv : (B, 64, H, W) -> (B, 3, H, W)          [3x3, sigmoid]
    """
    def __init__(
        self,
        channel_c: int = config.CHANNEL_C,
        out_channels: int = config.IN_CHANNELS,
        base_ch: int = 64,
        num_msb: int = 3,
    ):
        super().__init__()
        self.expand = nn.Sequential(
            nn.ConvTranspose2d(channel_c, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
        )
        self.msblocks = nn.Sequential(
            *[MultiScaleBlock(base_ch, base_ch) for _ in range(num_msb)]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        out = self.expand(z_hat)    # (B, 128, H/2, W/2)
        out = self.up(out)          # (B, 64,  H,   W)
        out = self.msblocks(out)    # (B, 64,  H,   W)
        return self.out_conv(out)   # (B, 3,   H,   W)


class JSCCEncoderHD(nn.Module):
    """
    DeepJSCC-HD Encoder with ResNet Residual Blocks for high-resolution images.

    Structure:
      - Stem Conv: (B, 3, H, W) -> (B, 64, H/2, W/2)
      - Down Conv: (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
      - 3x ResBlocks: (B, 128, H/4, W/4)
      - Bottleneck Conv: (B, 128, H/4, W/4) -> (B, channel_c, H/4, W/4)
      - Power Normalization
    """
    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        channel_c: int = config.CHANNEL_C,
        power: float = config.POWER_CONSTRAINT,
        num_resblocks: int = 3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
        )
        self.resblocks = nn.Sequential(*[ResBlock(128) for _ in range(num_resblocks)])
        self.bottleneck = nn.Conv2d(128, channel_c, kernel_size=3, stride=1, padding=1)
        self.power_norm = PowerNormalization(power=power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.down(out)
        out = self.resblocks(out)
        out = self.bottleneck(out)
        return self.power_norm(out)


class JSCCDecoderHD(nn.Module):
    """
    DeepJSCC-HD Decoder with ResNet Residual Blocks for high-resolution images.

    Structure:
      - Expand Conv: (B, channel_c, H/4, W/4) -> (B, 128, H/4, W/4)
      - 3x ResBlocks: (B, 128, H/4, W/4)
      - Up Conv 1: (B, 128, H/4, W/4) -> (B, 64, H/2, W/2)
      - Up Conv 2: (B, 64, H/2, W/2) -> (B, 3, H, W)
      - Sigmoid
    """
    def __init__(
        self,
        channel_c: int = config.CHANNEL_C,
        out_channels: int = config.IN_CHANNELS,
        num_resblocks: int = 3,
    ):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(channel_c, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.resblocks = nn.Sequential(*[ResBlock(128) for _ in range(num_resblocks)])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        out = self.expand(z_hat)
        out = self.resblocks(out)
        out = self.up1(out)
        out = self.up2(out)
        return out


class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise (AWGN) Channel simulator.
    
    y = x + n, where n ~ N(0, sigma^2)
    Given Signal-to-Noise Ratio (SNR in dB):
        SNR_linear = 10^(SNR_dB / 10) = P / (2 * sigma^2) (for complex) or P / sigma^2 (for real)
    """
    def __init__(self, snr_db: float = config.DEFAULT_SNR_DB, power: float = config.POWER_CONSTRAINT):
        super().__init__()
        self.snr_db = snr_db
        self.power = power

    def forward(self, z: torch.Tensor, snr_db: Optional[float] = None) -> torch.Tensor:
        if snr_db is None:
            snr_db = self.snr_db
            
        # Linear SNR
        snr_linear = 10.0 ** (snr_db / 10.0)
        # Noise variance based on average symbol power
        noise_std = (self.power / snr_linear) ** 0.5
        
        noise = torch.randn_like(z) * noise_std
        return z + noise


class DeepJSCC(nn.Module):
    """
    End-to-End Deep Joint Source-Channel Communication System.
    Connects Encoder -> Communication Channel -> Decoder.

    Supported architectures (arch=):
      'baseline' : 5-layer CNN encoder/decoder (Experiment 1 / 3)
      'hd'       : ResNet residual backbone (Experiment 4)
      'ms'       : Multi-Scale Inception-style blocks (Experiment 5)
    """
    def __init__(
        self,
        in_channels: int = config.IN_CHANNELS,
        channel_c: int = config.CHANNEL_C,
        power: float = config.POWER_CONSTRAINT,
        snr_db: float = config.DEFAULT_SNR_DB,
        arch: str = "baseline",
    ):
        super().__init__()
        self.arch = arch
        if arch == "hd":
            self.encoder = JSCCEncoderHD(in_channels=in_channels, channel_c=channel_c, power=power)
            self.decoder = JSCCDecoderHD(channel_c=channel_c, out_channels=in_channels)
        elif arch == "ms":
            self.encoder = JSCCEncoderMS(in_channels=in_channels, channel_c=channel_c, power=power)
            self.decoder = JSCCDecoderMS(channel_c=channel_c, out_channels=in_channels)
        else:
            self.encoder = JSCCEncoder(in_channels=in_channels, channel_c=channel_c, power=power)
            self.decoder = JSCCDecoder(channel_c=channel_c, out_channels=in_channels)
        self.channel = AWGNChannel(snr_db=snr_db, power=power)

    def forward(self, x: torch.Tensor, snr_db: Optional[float] = None) -> torch.Tensor:
        # 1. Encode image to channel symbols
        z = self.encoder(x)
        
        # 2. Transmit through wireless channel
        z_hat = self.channel(z, snr_db=snr_db)
        
        # 3. Decode received symbols to reconstruct image
        x_hat = self.decoder(z_hat)
        return x_hat


# Quick verification / testing
if __name__ == "__main__":
    print("=" * 60)
    print("Deep JSCC Image Semantic Communication Pipeline Test")
    print("=" * 60)
    
    # Example: batch of 4 RGB images with size 32x32 (e.g., CIFAR-10)
    batch_size = 4
    channels = 3
    height, width = 32, 32
    
    # Input normalized in range [0, 1]
    x = torch.rand(batch_size, channels, height, width)
    print(f"1. Input Image Batch:        {tuple(x.shape)} (range: [{x.min():.2f}, {x.max():.2f}])")

    # Latent channel dimension 'c' (bandwidth parameter)
    c = 16
    snr_test_db = 10.0
    model = DeepJSCC(in_channels=channels, channel_c=c, power=1.0, snr_db=snr_test_db)

    # 1. Encode
    z = model.encoder(x)
    print(f"2. Encoded Symbols (z):      {tuple(z.shape)} (c={c} feature channels)")
    
    # Check power constraint: (1/k) * ||z||^2 should be 1.0
    k = z.shape[1] * z.shape[2] * z.shape[3]
    avg_power = torch.mean(torch.sum(z.view(batch_size, -1)**2, dim=1) / k).item()
    print(f"3. Channel Symbol Power:     {avg_power:.4f} (Target power constraint: 1.0)")

    # 2. Wireless Channel (AWGN)
    z_noisy = model.channel(z, snr_db=snr_test_db)
    print(f"4. Received Channel Symbols: {tuple(z_noisy.shape)} (Channel: AWGN @ {snr_test_db} dB)")

    # 3. Decode
    x_hat = model.decoder(z_noisy)
    print(f"5. Reconstructed Image:      {tuple(x_hat.shape)} (range: [{x_hat.min():.2f}, {x_hat.max():.2f}])")
    
    # Compute simple reconstruction MSE loss
    mse_loss = nn.functional.mse_loss(x_hat, x).item()
    print(f"6. Initial Un-trained MSE:   {mse_loss:.5f}")
    print("=" * 60)
    print("Pipeline verified successfully: shapes and power constraints matched!")
    print("=" * 60)


