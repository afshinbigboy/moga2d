import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from .blocks import SepConvBN

def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ['BN', 'GN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"] 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DoGEdge(nn.Module):
  """
  
  """
  def __init__(self,dim: int, scale_factor: list) -> None:
      super().__init__()
      
      self.scale_factor = scale_factor
      self.pwconv = nn.Conv2d(in_channels = dim,
                              out_channels = dim,
                              kernel_size = 1, 
                              stride = 1,
                              padding = 0)
      self.norm = LayerNorm(dim, 
                            eps=1e-6,
                            data_format="channels_first")  
      self.act = nn.GELU()
  def forward(self, x):
      x_ = x.clone()
      _, C, H, W = x.shape
      # DOG
      x_1 = F.interpolate(x, 
                          scale_factor = self.scale_factor[0], 
                          mode = "bilinear")
      x_2 = F.interpolate(x, 
                          scale_factor = self.scale_factor[1], 
                          mode = "bilinear")
      # Upsample 
      x_1 = F.interpolate(x_1, 
                       size = H, 
                       mode = 'bilinear')
      
      x_2 = F.interpolate(x_2, 
                       size = H, 
                       mode = 'bilinear')     
      # Projection
      ex = x_1 - x_2
      x = ex + x
      x = self.norm(self.act(self.pwconv(x)))
      return x + x_

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class MultiOrderDWConv(nn.Module):
#     """Multi-order Features with Dilated DWConv Kernel.

#     Args:
#         embed_dims (int): Number of input channels.
#         dw_dilation (list): Dilations of three DWConv layers.
#         channel_split (list): The raletive ratio of three splited channels.
#     """

#     def __init__(self,
#                  embed_dims,
#                  dw_dilation=[1, 2, 3,],
#                  channel_split=[1, 3, 4,],
#                 ):
#         super(MultiOrderDWConv, self).__init__()

#         self.split_ratio = [i / sum(channel_split) for i in channel_split]
#         self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
#         self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
#         self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
#         self.embed_dims = embed_dims
#         assert len(dw_dilation) == len(channel_split) == 3
#         assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
#         assert embed_dims % sum(channel_split) == 0

#         # basic DW conv
#         self.DW_conv0 = nn.Conv2d(
#             in_channels=self.embed_dims,
#             out_channels=self.embed_dims,
#             kernel_size=5,
#             padding=(1 + 4 * dw_dilation[0]) // 2,
#             groups=self.embed_dims,
#             stride=1, dilation=dw_dilation[0],
#         )
#         # DW conv 1
#         self.DW_conv1 = nn.Conv2d(
#             in_channels=self.embed_dims_1,
#             out_channels=self.embed_dims_1,
#             kernel_size=5,
#             padding=(1 + 4 * dw_dilation[1]) // 2,
#             groups=self.embed_dims_1,
#             stride=1, dilation=dw_dilation[1],
#         )
#         # DW conv 2
#         self.DW_conv2 = nn.Conv2d(
#             in_channels=self.embed_dims_2,
#             out_channels=self.embed_dims_2,
#             kernel_size=7,
#             padding=(1 + 6 * dw_dilation[2]) // 2,
#             groups=self.embed_dims_2,
#             stride=1, dilation=dw_dilation[2],
#         )
#         # a channel convolution
#         self.PW_conv = nn.Conv2d(  # point-wise convolution
#             in_channels=embed_dims,
#             out_channels=embed_dims,
#             kernel_size=1)

#     def forward(self, x):
#         x_0 = self.DW_conv0(x)
#         x_1 = self.DW_conv1(
#             x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
#         x_2 = self.DW_conv2(
#             x_0[:, self.embed_dims-self.embed_dims_2:, ...])
#         x = torch.cat([
#             x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
#         x = self.PW_conv(x)
#         return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3],
                 channel_split=[1, 3, 4, 2],
                 rates=[6, 12, 18],
                 flag_useAllChannels=False,
                ):
        super(MultiOrderDWConv, self).__init__()
        
        channel_split=[5, 5, 5, 1]

        self.useAllChannels = flag_useAllChannels
        if flag_useAllChannels:
            channel_indices = [(0, embed_dims)] * len(channel_split)
        else:
            split_ratio = [i / sum(channel_split) for i in channel_split]
            channel_indices = [(0, int(split_ratio[0] * embed_dims))]
            for cr in split_ratio[1:]:
                nci = int(cr * embed_dims)
                assert nci > 0, "Ops. Channel split ratio is not correct"
                channel_indices.append((channel_indices[-1][1], channel_indices[-1][1] + nci))
        self.channel_indices = channel_indices
        
        assert len(rates)+1 == len(channel_split) == 4

        self.dlps = nn.ModuleList()
        for rate, cids in zip(rates, channel_indices):
            kernel_size_effective = 3 + (3 - 1) * (rate - 1)
            padding = (kernel_size_effective - 1) // 2
            self.dlps.append(
                SepConvBN(
                    in_channels=cids[1]-cids[0],
                    filters=cids[1]-cids[0],
                    kernel_size=3,
                    stride=1,
                    rate=rate,
                    depth_activation=True,
                    epsilon=1e-5
                )
            )
            
        ipd = channel_indices[-1][1] - channel_indices[-1][0]
        # image pooling
        self.dlps.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(ipd, ipd, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ipd, eps=1e-5),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=14, mode='bilinear', align_corners=False),
            nn.UpsamplingBilinear2d(scale_factor=7)
        ))
        self.embed_dims = embed_dims
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        dls_res = []
        # print(self.embed_dims)
        for dlp, csps in zip(self.dlps, self.channel_indices):
            # print(dlp, "/n", csps)
            y = dlp(x[:, csps[0]:csps[1], ...])
            if y.shape[2] != x.shape[2] or y.shape[3] != x.shape[3]:
                y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            dls_res.append(y)
        
        if self.useAllChannels:
            x = torch.sum(dls_res)
        else:
            x = torch.cat(dls_res, dim=1)

        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x

class MultiOrderGatedAggregationV2(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation Version 2 (With Boundary Enhancement).

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 scale_factors = [0.8, 0.4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 use_feat_decompose = False
                ):
        super(MultiOrderGatedAggregationV2, self).__init__()

        self.use_feat_decompose = use_feat_decompose
        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

        # boundary 
        self.boundary = DoGEdge(
            dim = embed_dims, scale_factor = scale_factors)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        if self.use_feat_decompose == True:
            x = self.feat_decompose(x)
        else:
            x = self.boundary(x)

        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x

class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 scale_factors = [0.8, 0.4],
                 use_feat_decompose = False,
                 order_conv = "v2"
                ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        if order_conv == "v2":
            self.attn = MultiOrderGatedAggregationV2(
                embed_dims,
                attn_dw_dilation=attn_dw_dilation,
                attn_channel_split=attn_channel_split,
                attn_act_type=attn_act_type,
                attn_force_fp32=attn_force_fp32,
                scale_factors = scale_factors,
                use_feat_decompose = use_feat_decompose
            )
        else:
            self.attn = MultiOrderGatedAggregation(
                embed_dims,
                attn_dw_dilation=attn_dw_dilation,
                attn_channel_split=attn_channel_split,
                attn_act_type=attn_act_type,
                attn_force_fp32=attn_force_fp32
            )

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


class ConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of PatchEmbed.
            Defaults to 3.
        stride (int): The conv stride of PatchEmbed. Defaults to 2.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class StackConvPatchEmbed(nn.Module):
    """An implementation of Stack Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_type='GELU',
                 norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size