import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from fvcore.nn import FlopCountAnalysis
from contextlib import redirect_stderr
import io

# <<<<<<< HEAD
import sys  
sys.path.append('./..')  # Assuming test.ipynb is in the lib directory  
from mogamain.modifiedmoganet import MogaBlock
from mogamain.moga import MogaBlock as MogaBlock_af
from mogamain.moga import DoGEdge
from mogamain.multihead_diffattn import MultiheadDiffAttn, MultiheadDiffAttnCrossV1, MultiheadDiffAttnCrossV2
# # import sys  
# # sys.path.append('./..')  # Assuming test.ipynb is in the lib directory  
# from moga.modifiedmoganet import MogaBlock
# >>>>>>> b2141d5 (updated from source)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs

    
class DWConvLKA(nn.Module):
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConvLKA(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(MSCB, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        
#   Multi-scale convolution block (MSCB)
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        """
        create a series of multi-scale convolution blocks.
        """
        convs = []
        mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        convs.append(mscb)
        if n > 1:
            for i in range(1, n):
                mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
                convs.append(mscb)
        conv = nn.Sequential(*convs)
        return conv

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
	        nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

#   Large-kernel grouped attention gate (LGAG)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x*psi
    
#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    
#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

#   Efficient multi-scale convolutional attention decoding (EMCAD)
class EMCAD(nn.Module):
    """
    The Efficient multi-scale convolutional attention decoding (EMCAD) module.

    Parameters: 5.17 M,	FLOPs: 19.73 G
    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCAD,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
	
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=lgag_ks, groups=channels[1]//2)
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2]//2, kernel_size=lgag_ks, groups=channels[2]//2)
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        
        self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        d4 = self.cab4(x)*x
        d4 = self.sab(d4)*d4 
        d4 = self.mscb4(d4)
        
        # EUCB3
        d3 = self.eucb3(d4)
                
        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        d3 = self.cab3(d3)*d3
        d3 = self.sab(d3)*d3  
        d3 = self.mscb3(d3)
        
        # EUCB2
        d2 = self.eucb2(d3)
        
        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        d2 = self.cab2(d2)*d2
        d2 = self.sab(d2)*d2
        d2 = self.mscb2(d2)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        d1 = self.cab1(d1)*d1
        d1 = self.sab(d1)*d1
        d1 = self.mscb1(d1)
        
        return [d4, d3, d2, d1]


class EMCADv2(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet.
    The other components are the same as EMCAD.

    Parameters: 5.03 M,	FLOPs: 17.90 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv2,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag3 = LGAG(F_g=channels[1], 
                          F_l=channels[1], 
                          F_int=channels[1]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[1]//2)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag2 = LGAG(F_g=channels[2], 
                          F_l=channels[2], 
                          F_int=channels[2]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[2]//2)
        
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))

        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        
        self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        d4 = self.cab4(x)*x
        d4 = self.sab(d4)*d4 
        d4 = self.moga4(d4)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        d3 = self.cab3(d3)*d3
        d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        d2 = self.cab2(d2)*d2
        d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        d1 = self.cab1(d1)*d1
        d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return [d4, d3, d2, d1]
    
class EMCADv3(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB and SAB layers.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv3,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag3 = LGAG(F_g=channels[1], 
                          F_l=channels[1], 
                          F_int=channels[1]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[1]//2)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag2 = LGAG(F_g=channels[2], 
                          F_l=channels[2], 
                          F_int=channels[2]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[2]//2)
        
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))

        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return [d4, d3, d2, d1]
    
class EMCADv3(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv3,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag3 = LGAG(F_g=channels[1], 
                          F_l=channels[1], 
                          F_int=channels[1]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[1]//2)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag2 = LGAG(F_g=channels[2], 
                          F_l=channels[2], 
                          F_int=channels[2]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[2]//2)
        
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))

        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1


class EMCADv4(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv4,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag3 = LGAG(F_g=channels[1], 
                          F_l=channels[1], 
                          F_int=channels[1]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[1]//2)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag2 = LGAG(F_g=channels[2], 
                          F_l=channels[2], 
                          F_int=channels[2]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[2]//2)
        
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))

        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        
        self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        d4 = self.cab4(x)*x
        d4 = self.sab(d4)*d4 
        d4 = self.moga4(d4)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        d3 = self.cab3(d3)*d3
        d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        d2 = self.cab2(d2)*d2
        d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        d1 = self.cab1(d1)*d1
        d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1
    

class EMCADv5(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv5,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.proj3 = nn.Conv2d(in_channels= channels[1], out_channels = channels[1], kernel_size = 1, stride = 1)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.proj2 = nn.Conv2d(in_channels= channels[2], out_channels = channels[2], kernel_size = 1, stride = 1)
    
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        
        self.proj1 = nn.Conv2d(in_channels= channels[3], out_channels = channels[3], kernel_size = 1, stride = 1)
        
        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        
        # Additive aggregation 3
        d3 = d3 + skips[0]
        
        self.proj3(d3)
        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        
        # Additive aggregation 2
        d2 = d2 + skips[1] 
        self.proj2(d2)
        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        
        # Additive aggregation 1
        d1 = d1 + skips[2]
        self.proj1(d1)
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1    

    
class EMCADv4(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv4,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag3 = LGAG(F_g=channels[1], 
                          F_l=channels[1], 
                          F_int=channels[1]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[1]//2)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.lgag2 = LGAG(F_g=channels[2], 
                          F_l=channels[2], 
                          F_int=channels[2]//2, 
                          kernel_size=lgag_ks, 
                          groups=channels[2]//2)
        
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))

        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        
        self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        d4 = self.cab4(x)*x
        d4 = self.sab(d4)*d4 
        d4 = self.moga4(d4)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        x3 = self.lgag3(g=d3, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        d3 = self.cab3(d3)*d3
        d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        x2 = self.lgag2(g=d2, x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        d2 = self.cab2(d2)*d2
        d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1, x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        d1 = self.cab1(d1)*d1
        d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1
    

class EMCADv5(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv5,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.proj3 = nn.Conv2d(in_channels= channels[1], out_channels = channels[1], kernel_size = 1, stride = 1)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.proj2 = nn.Conv2d(in_channels= channels[2], out_channels = channels[2], kernel_size = 1, stride = 1)
    
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        
        self.proj1 = nn.Conv2d(in_channels= channels[3], out_channels = channels[3], kernel_size = 1, stride = 1)
        
        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        
        # Additive aggregation 3
        d3 = d3 + skips[0]
        
        self.proj3(d3)
        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        
        # Additive aggregation 2
        d2 = d2 + skips[1] 
        self.proj2(d2)
        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        
        # Additive aggregation 1
        d1 = d1 + skips[2]
        self.proj1(d1)
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1    

class EMCADv6(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD) + Diffattn
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], input_size = [14, 28, 56, 112],kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv6,self).__init__()
        num_heads = [16,8,4]
        eucb_ks = 3 # kernel size for eucb
        self.input_size = input_size

        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        

        self.diffattn3 = MultiheadDiffAttn(embed_dim= channels[1] * 2, depth= 1, num_heads= num_heads[0])

        self.proj3 = nn.Conv2d(in_channels= channels[1] * 2 , out_channels = channels[1], kernel_size = 1, stride = 1)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        

        self.diffattn2 = MultiheadDiffAttn(embed_dim= channels[2] * 2, depth= 2, num_heads= num_heads[1])

        self.proj2 = nn.Conv2d(in_channels= channels[2] *2 , out_channels = channels[2], kernel_size = 1, stride = 1)
    
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.diffattn1 = MultiheadDiffAttn(embed_dim= channels[3] * 2, depth= 3, num_heads= num_heads[2])

        self.proj1 = nn.Conv2d(in_channels= channels[3] * 2, out_channels = channels[3], kernel_size = 1, stride = 1)
        
        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        
        # Additive aggregation 3
        
        d3 = torch.cat([d3, skips[0]], dim=1)
        d3 = d3.view(d3.shape[0],-1, d3.shape[1]) # B, L, C
        d3 = self.diffattn3(d3) * d3
        d3 = d3.view(d3.shape[0], d3.shape[2], d3.shape[1]//self.input_size[0], d3.shape[1]//self.input_size[0])

        d3 = self.proj3(d3) + skips[0]

        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)

        d2 = torch.cat([d2, skips[1]], dim=1)

        d2 = d2.view(d2.shape[0],-1, d2.shape[1]) # B L C # B 784 256

        d2 = self.diffattn2(d2) * d2

        d2 = d2.view(d2.shape[0], d2.shape[2], d2.shape[1]//self.input_size[1], d2.shape[1]//self.input_size[1])

        d2 = self.proj2(d2) + skips[1]

        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        # LGAG1        
        # Additive aggregation 1
        #d1 = d1 + skips[2]
        d1 = torch.cat([d1, skips[2]], dim=1)
        d1 = d1.view(d1.shape[0],-1, d1.shape[1])
        d1 = self.diffattn1(d1) * d1
        d1 = d1.view(d1.shape[0], d1.shape[2], d1.shape[1]//self.input_size[2], d1.shape[1]//self.input_size[2])
        d1 = self.proj1(d1) + skips[2]
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1 

class EMCADv7(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD) + Diffattn (Cross - query is from decoder)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], 
                 input_size = [14, 28, 56, 112],
                 kernel_sizes=[1,3,5], 
                 expansion_factor=6, dw_parallel=True, 
                 add=True, lgag_ks=3, activation='relu6'):
        
        super(EMCADv7, self).__init__()
        num_heads = [16,8,4]
        eucb_ks = 3 # kernel size for eucb
        self.input_size = input_size

        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.diffattn3 = MultiheadDiffAttnCrossV1(embed_dim= channels[1],
                                                  depth= 1, 
                                                  num_heads= num_heads[0],
                                                  H = input_size[0],
                                                  W = input_size[0])

        self.proj3 = nn.Conv2d(in_channels= channels[1] , 
                               out_channels = channels[1], 
                               kernel_size = 1, stride = 1)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        

        self.diffattn2 = MultiheadDiffAttnCrossV1(embed_dim= channels[2],
                                                  depth= 2, 
                                                  num_heads= num_heads[1],
                                                  H = input_size[1],
                                                  W = input_size[1])

        self.proj2 = nn.Conv2d(in_channels= channels[2], 
                               out_channels = channels[2], 
                               kernel_size = 1, stride = 1)
    
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], 
                          out_channels=channels[3], 
                          kernel_size=eucb_ks, stride=eucb_ks//2)
        
        self.diffattn1 = MultiheadDiffAttnCrossV1(embed_dim= channels[3],
                                                  depth= 3, 
                                                  num_heads= num_heads[2],
                                                  H = input_size[2],
                                                  W = input_size[2])
        
        self.proj1 = nn.Conv2d(in_channels= channels[3], 
                               out_channels = channels[3], 
                               kernel_size = 1, stride = 1)
        
        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
         
    def forward(self, x, skips):
            
        d4 = self.moga4(x)

        # EUCB3
        d3 = self.eucb3(d4)
        
        d3 = self.diffattn3(skips[0], d3)

        d3 = self.proj3(d3)

        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)

        d2 = self.diffattn2(skips[1] ,d2)

        d2 = self.proj2(d2)

        d2 = self.moga2(d2)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        d1 = self.diffattn1(skips[2] ,d1)
        d1 = self.proj1(d1)

        d1 = self.moga1(d1)
        
        return d1

class EMCADv8(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD) + Diffattn (Cross - query is from decoder)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], 
                 input_size = [14, 28, 56, 112],
                 kernel_sizes=[1,3,5], 
                 expansion_factor=6, dw_parallel=True, 
                 add=True, lgag_ks=3, activation='relu6'):
        
        super(EMCADv8, self).__init__()
        mlp_ratio = 4.
        num_heads = [16,8,4]
        eucb_ks = 3 # kernel size for eucb
        self.input_size = input_size

        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.diffattn3 = MultiheadDiffAttnCrossV2(embed_dim= channels[1],
                                                  depth= 1, 
                                                  num_heads= num_heads[0],
                                                  H = input_size[0],
                                                  W = input_size[0])

        self.proj3 = nn.Conv2d(in_channels= channels[1] , 
                               out_channels = channels[1], 
                               kernel_size = 1, stride = 1)
        
                        
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        

        self.diffattn2 = MultiheadDiffAttnCrossV2(embed_dim= channels[2],
                                                  depth= 2, 
                                                  num_heads= num_heads[1],
                                                  H = input_size[1],
                                                  W = input_size[1])

        self.proj2 = nn.Conv2d(in_channels= channels[2], 
                               out_channels = channels[2], 
                               kernel_size = 1, stride = 1)
        
    
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], 
                          out_channels=channels[3], 
                          kernel_size=eucb_ks, stride=eucb_ks//2)
        
        self.diffattn1 = MultiheadDiffAttnCrossV2(embed_dim= channels[3],
                                                  depth= 3, 
                                                  num_heads= num_heads[2],
                                                  H = input_size[2],
                                                  W = input_size[2])
        
        self.proj1 = nn.Conv2d(in_channels= channels[3], 
                               out_channels = channels[3], 
                               kernel_size = 1, stride = 1)
                    
        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
         
    def forward(self, x, skips):
            
        d4 = self.moga4(x)

        # EUCB3
        d3 = self.eucb3(d4)
        
        d3 = self.diffattn3(d3, skips[0])
        d3 = self.proj3(d3)

        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)

        d2 = self.diffattn2(d2, skips[1])
        d2 = self.proj2(d2)
        

        d2 = self.moga2(d2)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        d1 = self.diffattn1(d1, skips[2])
        d1 = self.proj1(d1)

        d1 = self.moga1(d1)
        
        return d1
    
class EMCADv9(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD) + Diffattn (addition)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD.

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], input_size = [14, 28, 56, 112],kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv9,self).__init__()
        num_heads = [16,8,4]
        eucb_ks = 3 # kernel size for eucb
        self.input_size = input_size

        self.moga4 = MogaBlock(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.diffattn3 = MultiheadDiffAttn(embed_dim= channels[1], depth= 1, num_heads= num_heads[0])

        self.proj3 = nn.Conv2d(in_channels= channels[1] , out_channels = channels[1], kernel_size = 1, stride = 1)
        
        self.moga3 = MogaBlock(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.diffattn2 = MultiheadDiffAttn(embed_dim= channels[2], depth= 2, num_heads= num_heads[1])

        self.proj2 = nn.Conv2d(in_channels= channels[2] , out_channels = channels[2], kernel_size = 1, stride = 1)
    
        self.moga2 = MogaBlock(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.diffattn1 = MultiheadDiffAttn(embed_dim= channels[3] , depth= 3, num_heads= num_heads[2])

        self.proj1 = nn.Conv2d(in_channels= channels[3] , out_channels = channels[3], kernel_size = 1, stride = 1)
        
        self.moga1 = MogaBlock(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False)
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        
        # Additive aggregation 3
        
        d3 = d3 + skips[0]
        r3 = d3.clone()
        d3 = d3.view(d3.shape[0],-1, d3.shape[1]) # B, L, C
        d3 = self.diffattn3(d3)
        d3 = d3.view(d3.shape[0], d3.shape[2], d3.shape[1]//self.input_size[0], d3.shape[1]//self.input_size[0])
        d3 = self.proj3(d3)
        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3 + r3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        
        # Additive aggregation 2 
        d2 = d2 + skips[1]
        r2 = d2.clone()
        d2 = d2.view(d2.shape[0],-1, d2.shape[1]) # B L C # B 784 256
        d2 = self.diffattn2(d2)
        d2 = d2.view(d2.shape[0], d2.shape[2], d2.shape[1]//self.input_size[1], d2.shape[1]//self.input_size[1])
        d2 = self.proj2(d2)
        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2 + r2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        
        # Additive aggregation 1
        d1 = d1 + skips[2]
        r1 = d1.clone()
        d1 = d1.view(d1.shape[0],-1, d1.shape[1])
        d1 = self.diffattn1(d1)
        d1 = d1.view(d1.shape[0], d1.shape[2], d1.shape[1]//self.input_size[2], d1.shape[1]//self.input_size[2])
        d1 = self.proj1(d1)
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1 + r1)
        
        return d1 

    
class EMCADv5_DW_Skip(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD except the modified version of MOGA (improved by Afshin).

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], scale_factors = [0.8,0.4], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv5_DW_Skip,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock_af(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v1")
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)

        self.boundary3 = DoGEdge(dim = channels[1], scale_factor = scale_factors)
        
        self.moga3 = MogaBlock_af(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v1")
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        self.boundary2 = DoGEdge(dim = channels[2], scale_factor = scale_factors)
    
        self.moga2 = MogaBlock_af(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v1")
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        
        self.boundary1 = DoGEdge(dim = channels[3], scale_factor = scale_factors)
        
        self.moga1 = MogaBlock_af(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v1")
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        
        # Additive aggregation 3
        skips_3 = self.boundary3(skips[0])
        
        d3 = d3 + skips_3
        
        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        
        # Additive aggregation 2
        skips_2 = self.boundary2(skips[1])
        d2 = d2 + skips_2
        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        
        # Additive aggregation 1
        skips_1 = self.boundary1(skips[2])
        d1 = d1 + skips_1
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1    

class EMCADv5_DW(nn.Module):
    """
    Efficient multi-scale convolutional attention decoding (EMCAD)
    
    This version of EMCAD uses the modified MogaBlock from Sinanet and excluded the CAB, LGAG, SAB layers, and Deep supervision.
    The other components are the same as EMCAD except the modified version of MOGA (improved by Afshin).

    Parameters: 4.98 M,	FLOPs: 17.88 G


    """
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], scale_factors = [0.8,0.4], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCADv5_DW,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        self.moga4 = MogaBlock_af(embed_dims= channels[0], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v2")
        	
        self.eucb3 = EUCB(in_channels=channels[0], 
                          out_channels=channels[1], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.proj3 = nn.Conv2d(in_channels= channels[1], out_channels = channels[1], kernel_size = 1, stride = 1)
        
        self.moga3 = MogaBlock_af(embed_dims= channels[1], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v2")
        	
        self.eucb2 = EUCB(in_channels=channels[1], 
                          out_channels=channels[2], 
                          kernel_size=eucb_ks, 
                          stride=eucb_ks//2)
        
        self.proj2 = nn.Conv2d(in_channels= channels[2], out_channels = channels[2], kernel_size = 1, stride = 1)
    
        self.moga2 = MogaBlock_af(embed_dims= channels[2], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v2")
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        
        self.proj1 = nn.Conv2d(in_channels= channels[3], out_channels = channels[3], kernel_size = 1, stride = 1)
        
        self.moga1 = MogaBlock_af(embed_dims= channels[3], 
                               ffn_ratio=4, drop_rate= 0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value= 1e-6,
                               attn_dw_dilation= [1,2,3],
                               attn_channel_split= [1,3,4],
                               attn_act_type= "SiLU",
                               attn_force_fp32= False,
                               scale_factors = scale_factors,
                               use_feat_decompose = False,
                               order_conv = "v2")
        
        # self.cab4 = CAB(channels[0])
        # self.cab3 = CAB(channels[1])
        # self.cab2 = CAB(channels[2])
        # self.cab1 = CAB(channels[3])
        
        # self.sab = SAB()
       
      
    def forward(self, x, skips):
            
        # MSCAM4
        # d4 = self.cab4(x)*x
        # d4 = self.sab(d4)*d4 
        d4 = self.moga4(x)
        
        # EUCB3
        d3 = self.eucb3(d4)
        

        # LGAG3
        
        # Additive aggregation 3
        d3 = d3 + skips[0]
        
        self.proj3(d3)
        # MSCAM3
        # d3 = self.cab3(d3)*d3
        # d3 = self.sab(d3)*d3  
        d3 = self.moga3(d3)
        # print(d3.shape)
        
        # EUCB2
        d2 = self.eucb2(d3)
        # print(d2.shape)
        
        # LGAG2
        
        # Additive aggregation 2
        d2 = d2 + skips[1] 
        self.proj2(d2)
        
        # MSCAM2
        # d2 = self.cab2(d2)*d2
        # d2 = self.sab(d2)*d2
        d2 = self.moga2(d2)
        # print(d2.shape)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        
        # Additive aggregation 1
        d1 = d1 + skips[2]
        self.proj1(d1)
        # MSCAM1
        # d1 = self.cab1(d1)*d1
        # d1 = self.sab(d1)*d1
        d1 = self.moga1(d1)
        
        return d1    
    
if __name__ == '__main__':
    # Test the EMCAD module
    model = EMCADv7().cuda()

    input_tensor = torch.randn(1, 512, 32, 32).cuda()
    skips = [torch.randn(1, 320, 64, 64).cuda(), 
             torch.randn(1, 128, 128, 128).cuda(), 
             torch.randn(1, 64, 256, 256).cuda()]
    print(model(input_tensor, skips).shape)


    # def print_param_flops_skips(net, input_shape):
    #     x = torch.randn(1, *input_shape).to("cuda")
    #     skips = [torch.randn(1, 320, 64, 64).cuda(), 
    #              torch.randn(1, 128, 128, 128).cuda(), 
    #              torch.randn(1, 64, 256, 256).cuda()]
        
    #     params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    #     with redirect_stderr(io.StringIO()):
    #         flops = FlopCountAnalysis(net, (x, skips))
    #         flops_amount = flops.total()

    #     print(f"Parameters: {params/1e6:.2f} M,\tFLOPs: {flops_amount/1e9:.2f} G")

    # print_param_flops_skips(model, (512, 32, 32))
    