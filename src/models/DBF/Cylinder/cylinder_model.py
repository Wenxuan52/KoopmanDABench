from torch.nn.modules import Module

import torch
import torch.nn as nn
import torch.nn.functional as F


CYLINDER_settings = {"obs_dim": [2, 64, 64], 
                    "state_dim": [2, 64, 64], 
                    "seq_length": 12,
                    "obs_feature_dim": [512, 128, 64, 32, 16, 8], 
                    "state_filter_feature_dim": [16, 32, 64, 128, 256],
                    "state_feature_dim": [4096, 512]}


class CYLINDER_K_S(Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CYLINDER_K_S, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = CYLINDER_settings["state_dim"]
        self.filter_dims = CYLINDER_settings["state_filter_feature_dim"]
        self.hidden_dims = CYLINDER_settings["state_feature_dim"] # [Dim before linear, state_feature_dim]

        # First convolution layer with larger kernel for feature extraction
        self.Conv2D_size7_1 = nn.Conv2d(in_channels=self.input_dim, out_channels=self.filter_dims[0], 
                                  kernel_size=7, stride=1, padding=3)
        
        # Second convolution layer
        self.Conv2D_size5_1 =  nn.Conv2d(in_channels=self.filter_dims[0], out_channels=self.filter_dims[1], 
                                  kernel_size=5, stride=1, padding=2)
        
        # Third convolution layer
        self.Conv2D_size3_1 = nn.Conv2d(in_channels=self.filter_dims[1], out_channels=self.filter_dims[2], 
                                              kernel_size=3, stride=1, padding=1)
        
        # Fourth convolution layer
        self.Conv2D_size3_2 = nn.Conv2d(in_channels=self.filter_dims[2], out_channels=self.filter_dims[3], 
                                              kernel_size=3, stride=1, padding=1)
        
        # Fifth convolution layer
        self.Conv2D_size3_3 = nn.Conv2d(in_channels=self.filter_dims[3], out_channels=self.filter_dims[4], 
                                              kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
    
        self.linear = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])


    def forward(self, state: torch.Tensor):
        # First layer: 7x7 conv + pooling
        en_state_1 = self.Conv2D_size7_1(state)
        en_state_1 = self.pooling(en_state_1)
        en_state_1 = self.relu(en_state_1)

        # Second layer: 5x5 conv + pooling
        en_state_2 = self.Conv2D_size5_1(en_state_1)
        en_state_2 = self.relu(en_state_2)
        en_state_2 = self.pooling(en_state_2)

        # Third layer: 3x3 conv + pooling
        en_state_3 = self.Conv2D_size3_1(en_state_2)
        en_state_3 = self.pooling(en_state_3)
        en_state_3 = self.relu(en_state_3)

        # Fourth layer: 3x3 conv + pooling
        en_state_4 = self.Conv2D_size3_2(en_state_3)
        en_state_4 = self.pooling(en_state_4)
        en_state_4 = self.relu(en_state_4)

        # Fifth layer: 3x3 conv
        en_state_5 = self.Conv2D_size3_3(en_state_4)
        en_state_5 = self.relu(en_state_5)
        en_state_5 = self.dropout(en_state_5)

        # Flatten and linear transformation
        en_state_5 = self.flatten(en_state_5)
        z = self.linear(en_state_5)

        return z


class CYLINDER_K_S_preimage(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CYLINDER_K_S_preimage, self).__init__(*args, **kwargs)
        self.input_dim, self.w, self.h = CYLINDER_settings["state_dim"]
        self.filter_dims = CYLINDER_settings["state_filter_feature_dim"]
        self.hidden_dims = CYLINDER_settings["state_feature_dim"] # [Dim before linear, state_feature_dim]

        self.linear = nn.Linear(self.hidden_dims[1], self.hidden_dims[0])
        
        # Transpose convolution layers (reverse order of encoder)
        self.ConvTranspose2D_size3_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[4], out_channels=self.filter_dims[3], 
                                                          kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2D_size3_2 = nn.ConvTranspose2d(in_channels=self.filter_dims[3], out_channels=self.filter_dims[2],
                                                          kernel_size=3, stride=1, padding=1)
        self.ConvTranspose2D_size3_3 = nn.ConvTranspose2d(in_channels=self.filter_dims[2], out_channels=self.filter_dims[1],
                                                          kernel_size=3, stride=1, padding=1)
        
        self.Upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.ConvTranspose2D_size5_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[1], out_channels=self.filter_dims[0],
                                                          kernel_size=5, stride=1, padding=2)
        
        self.ConvTranspose2D_size7_1 = nn.ConvTranspose2d(in_channels=self.filter_dims[0], out_channels=self.input_dim,
                                                          kernel_size=7, stride=1, padding=3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Output refinement layers
        self.output_conv = nn.Sequential(nn.Conv2d(in_channels=self.input_dim, out_channels=64, 
                                         kernel_size=1, stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=64, out_channels=32, 
                                         kernel_size=1, stride=1),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=32, out_channels=self.input_dim, 
                                         kernel_size=1, stride=1))


    def forward(self, z: torch.Tensor):
        # Linear transformation and reshape
        de_state_5 = self.linear(z)
        de_state_5 = self.relu(de_state_5)
        de_state_5 = self.dropout(de_state_5)
        de_state_5 = de_state_5.view(-1, self.filter_dims[4], 4, 4)  # Reshape to [batch, 256, 4, 4]

        # First transpose conv
        de_state_4 = self.ConvTranspose2D_size3_1(de_state_5)
        de_state_4 = self.relu(de_state_4)

        # Second transpose conv with upsampling
        de_state_3 = self.Upsampling(de_state_4)
        de_state_3 = self.ConvTranspose2D_size3_2(de_state_3)
        de_state_3 = self.relu(de_state_3)

        # Third transpose conv with upsampling
        de_state_2 = self.Upsampling(de_state_3) 
        de_state_2 = self.ConvTranspose2D_size3_3(de_state_2)
        de_state_2 = self.relu(de_state_2)

        # Fourth transpose conv with upsampling
        de_state_1 = self.Upsampling(de_state_2)
        de_state_1 = self.ConvTranspose2D_size5_1(de_state_1)
        de_state_1 = self.relu(de_state_1)

        # Final transpose conv with upsampling
        de_state_0 = self.Upsampling(de_state_1)
        de_state_0 = self.ConvTranspose2D_size7_1(de_state_0)
        
        # Output refinement
        recon_s = self.output_conv(de_state_0)

        return recon_s


class CylinderEncoder(nn.Module):
    """
    Wrapper around CYLINDER_K_S to produce latent z for each frame in a sequence.
    Input:  [B,T,C,H,W]
    Output: [B,T,dim_z]
    """
    def __init__(self, dim_z: int):
        super().__init__()
        self.encoder = CYLINDER_K_S()
        # CYLINDER_K_S outputs size 512 (as defined in CYLINDER_settings["state_feature_dim"][-1])
        assert dim_z == 512, f"CylinderEncoder latent dim should be 512, got {dim_z}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        z_list = []
        for t in range(T):
            frame = x[:, t]              # [B,C,H,W]
            z = self.encoder(frame)      # [B,512]
            z_list.append(z.unsqueeze(1))
        z_seq = torch.cat(z_list, dim=1) # [B,T,512]
        return z_seq


class CylinderDecoder(nn.Module):
    """
    Wrapper around CYLINDER_K_S_preimage to reconstruct frames from latent z.
    Input:  [B,T,dim_z]
    Output: [B,T,C,H,W]
    """
    def __init__(self, dim_z: int):
        super().__init__()
        self.decoder = CYLINDER_K_S_preimage()
        assert dim_z == 512, f"CylinderDecoder latent dim should be 512, got {dim_z}"

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = z_seq.shape
        frames = []
        for t in range(T):
            z = z_seq[:, t]              # [B,512]
            frame = self.decoder(z)      # [B,C,H,W]
            frames.append(frame.unsqueeze(1))
        x_rec = torch.cat(frames, dim=1) # [B,T,C,H,W]
        return x_rec
