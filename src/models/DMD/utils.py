from torch.nn import Module
from torch import Tensor, FloatTensor, pow, sin, cos, arange
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import *

def count_parameters(model:nn.Module)->int:
    """
    Count the number of parameters in a model.
    
    Args:
    - model: nn.Module, the model to count parameters.
    
    Returns:
    - int, the number of parameters in the model.
    """
    num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Number of parameters: {num_params}")
    return num_params


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class weighted_MSELoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return ((inputs - targets)**2 ) * weights
    
class View(Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class PositionalEncodingLayer(Module):
    
    def __init__(self, input_dim: int, max_len: int=100):
        super(PositionalEncodingLayer, self).__init__()
        self.input_dim = input_dim
        self.max_len = max_len
    
    def get_angles(self, positions: Tensor, indexes: Tensor):
        input_dim_tensor = FloatTensor([[self.input_dim]]).to(positions.device)
        angle_rates = pow(10000, (2 * (indexes // 2)) / input_dim_tensor)
        return positions / angle_rates

    def forward(self, input_sequences: Tensor, channel_first: bool=False):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, input_dim] position_encoding
        """
        assert len(input_sequences.shape) == 3, "input_sequences must be of shape [batch_size, seq_len, input_dim]"
        if channel_first:
            input_sequences = input_sequences.permute(0, 2, 1)
        positions = arange(input_sequences.size(1)).unsqueeze(1).to(input_sequences.device) # [seq_len, 1]
        indexes = arange(self.input_dim).unsqueeze(0).to(input_sequences.device) # [1, input_dim]
        angles = self.get_angles(positions, indexes) # [seq_len, input_dim]
        angles[:, 0::2] = sin(angles[:, 0::2]) # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = cos(angles[:, 1::2]) # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(input_sequences.size(0), 1, 1) # [batch_size, seq_len, input_dim]
        if channel_first:
            position_encoding = position_encoding.permute(0, 2, 1)
        return position_encoding


def is_symmetric(matrix, tol=1e-8):
    return torch.allclose(matrix, matrix.T, atol=tol)  


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Transformer_Based_Inv_Obs_Model(nn.Module):
    def __init__(self, in_channel:int=50, out_channel:int=5, LayerNorm_type = 'WithBias',
                 ffn_expansion_factor = 2.66, bias = False, num_blocks = [2, 2, 2, 2]):
        super(Transformer_Based_Inv_Obs_Model, self).__init__()

        dim_list = [in_channel*2, in_channel*4, in_channel*2, out_channel]
        num_heads = [5, 10, 5, 1]
        num_blocks = num_blocks

        self.patch_embed = OverlapPatchEmbed(in_channel, embed_dim=dim_list[0])
        self.Upsample_1 = Upsample_Flex(dim_list[0], dim_list[1])
        self.Upsample_2 = Upsample_Flex(dim_list[1], dim_list[2])
        self.Upsample_3 = Upsample_Flex(dim_list[2], dim_list[3])

        self.block1 = nn.Sequential(*[TransformerBlock(dim=dim_list[0], 
                                                       num_heads=num_heads[0], 
                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                       bias=bias, LayerNorm_type=LayerNorm_type) 
                                                       for i in range(num_blocks[0])])
        self.block2 = nn.Sequential(*[TransformerBlock(dim=dim_list[1], 
                                                       num_heads=num_heads[1], 
                                                       ffn_expansion_factor=ffn_expansion_factor, 
                                                       bias=bias, LayerNorm_type=LayerNorm_type) 
                                                       for i in range(num_blocks[1])])
        self.block3 = nn.Sequential(*[TransformerBlock(dim=dim_list[2],
                                                         num_heads=num_heads[2], 
                                                         ffn_expansion_factor=ffn_expansion_factor, 
                                                         bias=bias, LayerNorm_type=LayerNorm_type) 
                                                         for i in range(num_blocks[2])])
        self.block4 = nn.Sequential(*[TransformerBlock(dim=dim_list[3],
                                                        num_heads=num_heads[3], 
                                                        ffn_expansion_factor=ffn_expansion_factor, 
                                                        bias=bias, LayerNorm_type=LayerNorm_type) 
                                                        for i in range(num_blocks[3])])
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.block1(x)
        x = self.Upsample_1(x)

        x = self.block2(x)
        x = self.Upsample_2(x)

        x = self.block3(x)
        x = self.Upsample_3(x)

        x = self.block4(x)
        return x