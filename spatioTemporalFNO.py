import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure this import points to your correct SpectralConv module.
from neuralop.layers.spectral_convolution import SpectralConv

class SpatioTemporalBlock(nn.Module):
    def __init__(self, hidden_channels, n_modes, temporal_kernel=3):
        super(SpatioTemporalBlock, self).__init__()
        # Spatial processing: combine Fourier-based and conventional convolutions.
        self.spectral_conv = SpectralConv(hidden_channels, hidden_channels, n_modes=n_modes)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        # Temporal processing: 3D convolution along the time dimension.
        self.temporal_conv = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel // 2, 0, 0)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Process each time slice with spectral and spatial convolutions.
        x_flat = x.view(B * T, C, H, W)
        spec_out = self.spectral_conv(x_flat)
        spat_out = self.spatial_conv(x_flat)
        combined = spec_out + spat_out
        combined = self.relu(combined)
        combined = combined.view(B, T, C, H, W)
        
        # Apply temporal convolution.
        x_temp = combined.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        temp_out = self.temporal_conv(x_temp)
        temp_out = self.relu(temp_out)
        temp_out = temp_out.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        
        # Residual connection: add the input to the temporal output.
        out = x + temp_out
        return out

class SpatioTemporalFNO(nn.Module):
    def __init__(self, n_modes: tuple, in_channels: int, out_channels: int,
                 hidden_channels: int, n_layers: int = 4, T_in: int = 2,
                 temporal_kernel: int = 3):
        super(SpatioTemporalFNO, self).__init__()
        self.T_in = T_in
        self.hidden_channels = hidden_channels
        
        self.relu = nn.ReLU()
        
        # Improved lifting: use a 3x3 convolution plus a residual block.
        self.lifting_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.lifting_residual = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        
        # Stack the enhanced spatio-temporal blocks.
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(hidden_channels, n_modes=n_modes, temporal_kernel=temporal_kernel)
            for _ in range(n_layers)
        ])
        
        # Refinement module to help sharpen output details.
        self.refinement = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        )
        
        # Final projection to produce the output channels.
        self.projection_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass for SpatioTemporalFNO.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (B, T, C, H, W) or (B, C, H, W).
            If 4D, a time dimension is added (T becomes 1).
        
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (B, out_channels, H, W).
        """
        if x.dim() == 4:
            # Add a time dimension if missing.
            x = x.unsqueeze(1)  # Now shape: (B, 1, C, H, W)
        
        B, T, C, H, W = x.shape
        
        # Apply the improved lifting block for each time frame.
        lifted_frames = []
        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            lifted = self.lifting_conv(frame)  # (B, hidden_channels, H, W)
            lifted_res = self.lifting_residual(lifted)
            lifted = self.relu(lifted + lifted_res)
            lifted_frames.append(lifted)
        x_lifted = torch.stack(lifted_frames, dim=1)  # (B, T, hidden_channels, H, W)
        
        # Process through the enhanced spatio-temporal blocks.
        out = x_lifted
        for block in self.blocks:
            out = block(out)
        
        # Aggregate temporal information using the last time step.
        x_last = out[:, -1, :, :, :]  # (B, hidden_channels, H, W)
        refined = self.refinement(x_last)
        refined = self.relu(x_last + refined)  # Residual connection for refinement.
        
        # Final projection to output the optical flow.
        output = self.projection_conv(refined)  # (B, out_channels, H, W)
        return output
