import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model


import pdb
from .lns import Fp32LayerNorm, PermuteLayerNorm

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

expansion = 4

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, 
            bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )

def conv_helper(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,  
                kernel_size: int = 3, bias: bool = False) -> nn.Conv2d:
    # assert stride == 1
    assert kernel_size in [3,5,7,9,11]
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias, # currently false here, may need to change at last
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_path_rate: float = 0.0,
        depthwise: bool = False,
        kernel_size: int = 3,
        conv_first: bool = False, # not in use now
        act_layer: Optional[Callable[..., nn.Module]] = None,
        del_act: bool = False,
        reorg_norm: bool = False,
        pre_norm: bool = False,
        ls_init_values: float = 0.,
    ) -> None:
        super().__init__()

        assert not reorg_norm
        assert not pre_norm

        self.del_act = del_act
        assert not reorg_norm # not supported in original bottleneck block for now
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if depthwise:
            groups = width
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv2 = conv_helper(width, width, stride, groups, kernel_size=kernel_size)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = act_layer()
        self.downsample = downsample
        self.stride = stride

        if ls_init_values > 0:
            self.gamma_1 = nn.Parameter(ls_init_values * torch.ones((width * expansion)),requires_grad=True)
        else:
            self.gamma_1 = None

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        if not self.del_act:
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.gamma_1 is not None:
            out = self.gamma_1[:, None, None] * out

        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if not self.del_act:
            out = self.relu(out)

        return out


class InvertedBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: float, # 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_path_rate: float = 0.0,
        depthwise: bool = False,
        kernel_size: int = 3,
        conv_first: bool = False,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        del_act: bool = False,
        reorg_norm: bool = False,
        pre_norm: bool = False,
        ls_init_values: float = 0.,
    ) -> None:
        super().__init__()
        # switching planes and inplanes

        # print("Conv_first", conv_first)
        # print("reorg_norm", reorg_norm)

        self.del_act = del_act
        self.reorg_norm = reorg_norm
        self.pre_norm = pre_norm

        if self.pre_norm:
            assert self.reorg_norm
            assert conv_first

        use_bias = (norm_layer != nn.BatchNorm2d)

        # planes is for the intermediate channels now
        # planes, inplanes = inplanes, planes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups


        self.bn0 = nn.Identity()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if not conv_first:
            assert not self.reorg_norm
            assert not self.pre_norm

            self.conv1 = conv1x1(inplanes, width, bias=use_bias)
            self.bn1 = norm_layer(width) if not self.reorg_norm else nn.Identity()
            if depthwise:
                groups = width

            self.conv2 = conv_helper(width, width, stride, groups, kernel_size=kernel_size, bias=use_bias)
            self.bn2 = norm_layer(width)

            self.conv3 = conv1x1(width, width // expansion, bias=use_bias) # conv3 stays the same
            self.bn3 = norm_layer(width // expansion)

        else:
            # print(self.reorg_norm, self.pre_norm)
            if depthwise:
                groups = inplanes
            if self.reorg_norm and self.pre_norm:
                self.bn0 = norm_layer(inplanes)
                self.conv1 = conv_helper(inplanes, inplanes, stride, groups, kernel_size=kernel_size, bias=use_bias)
                self.bn1 = norm_layer(inplanes)
                self.conv2 = conv1x1(inplanes, width, bias=use_bias)
                self.bn2 = nn.Identity()
                self.conv3 = conv1x1(width, width // expansion, bias=use_bias) # conv3 stays the same
                self.bn3 = nn.Identity()

            elif self.reorg_norm:
                self.bn0 = nn.Identity()
                self.conv1 = conv_helper(inplanes, inplanes, stride, groups, kernel_size=kernel_size, bias=use_bias)
                self.bn1 = norm_layer(inplanes)
                self.conv2 = conv1x1(inplanes, width, bias=use_bias)
                self.bn2 = nn.Identity()
                self.conv3 = conv1x1(width, width // expansion, bias=use_bias) # conv3 stays the same
                self.bn3 = nn.Identity()

            else:
                self.bn0 = nn.Identity()
                self.conv1 = conv_helper(inplanes, inplanes, stride, groups, kernel_size=kernel_size, bias=use_bias)
                self.bn1 = norm_layer(inplanes)
                self.conv2 = conv1x1(inplanes, width, bias=use_bias)
                self.bn2 = norm_layer(width)
                self.conv3 = conv1x1(width, width // expansion, bias=use_bias) # conv3 stays the same
                self.bn3 = norm_layer(width // expansion)

        # self.relu = nn.ReLU(inplace=True)
        self.relu = act_layer()
        self.downsample = downsample
        self.stride = stride

        if ls_init_values > 0:
            self.gamma_1 = nn.Parameter(ls_init_values * torch.ones((width // expansion)),requires_grad=True)
        else:
            self.gamma_1 = None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn0(x)

        out = self.conv1(out)
        out = self.bn1(out)
        if not self.del_act:
            out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.gamma_1 is not None:
            out = self.gamma_1[:, None, None] * out
        out = self.drop_path(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.del_act:
            out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        drop_path_rate: float = 0.0,
        block: Type[Union[Bottleneck, InvertedBottleneck]] = Bottleneck,
        sep_downsample: bool = False,
        stem: str = 'original',
        wide_stem: bool = False,
        depthwise: bool = False,
        kernel_size: int = 3,
        conv_first: bool = False, # conv with ks first or not, only for inverted bottleneck
        embed_dims: List[int] = [64, 128, 256, 512],
        act_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        del_act: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        reorg_norm: bool = False,
        pre_norm: bool = False,
        final_norm: bool = False,
        ds_norm: bool = False,
        init: str = 'default',
        ls_init_values: float = 0.,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        # print("ResNet reorg_norm", reorg_norm)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.sep_downsample = sep_downsample
        self.final_norm = final_norm
        self.conv_bias = reorg_norm # arbitrary design now, need to reset later, should not matter much

        self.inplanes = embed_dims[0] #if not wide_stem else embed_dims[0] * expansion
        

        if block == Bottleneck:
            # ds_layer_sizes = [256, 512, 1024] #, 2048]
            ds_layer_sizes = [x * expansion for x in embed_dims[:3]]
        elif block == InvertedBottleneck:
            # ds_layer_sizes = [64, 128, 256] #, 1024]
            ds_layer_sizes = embed_dims[:3]
            embed_dims = [x * 4 for x in embed_dims]

        self.ds_norm = ds_norm
        if self.ds_norm: # only useful when downsample is separate
            assert self.sep_downsample
        if reorg_norm:
            assert block == InvertedBottleneck

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        if stem == 'original':
            self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif stem == 'patch_embed':
            self.stem = nn.Conv2d(3, self.inplanes, kernel_size=4, stride=4)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        if block == InvertedBottleneck: # if inverted bottleneck this controls the middle planes, not actual inplanes
            # self.inplanes = embed_dims[0] 
            # assert self.sep_downsample
            block = partial(InvertedBottleneck, conv_first=conv_first)

        if self.sep_downsample:
            self.layer1 = self._make_layer(block, embed_dims[0], layers[0], drop_path_rates=dpr[:layers[0]], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)
            self.ds1 = nn.Conv2d(ds_layer_sizes[0], ds_layer_sizes[0] * 2, kernel_size=2, stride=2)
            self.norm_ds1 = norm_layer(ds_layer_sizes[0] * 2) if self.ds_norm else nn.Identity()
            self.inplanes = self.inplanes * 2

            self.layer2 = self._make_layer(block, embed_dims[1], layers[1], stride=1, drop_path_rates=dpr[sum(layers[:1]):sum(layers[:2])], 
                                            dilate=replace_stride_with_dilation[0], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)

            self.ds2 = nn.Conv2d(ds_layer_sizes[1], ds_layer_sizes[1] * 2, kernel_size=2, stride=2)
            self.norm_ds2 = norm_layer(ds_layer_sizes[1] * 2) if self.ds_norm else nn.Identity()
            self.inplanes = self.inplanes * 2

            self.layer3 = self._make_layer(block, embed_dims[2], layers[2], stride=1, drop_path_rates=dpr[sum(layers[:2]):sum(layers[:3])],
                                            dilate=replace_stride_with_dilation[1], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)
            self.ds3 = nn.Conv2d(ds_layer_sizes[2], ds_layer_sizes[2] * 2, kernel_size=2, stride=2)
            self.norm_ds3 = norm_layer(ds_layer_sizes[2] * 2) if self.ds_norm else nn.Identity()
            self.inplanes = self.inplanes * 2

            self.layer4 = self._make_layer(block, embed_dims[3], layers[3], stride=1, drop_path_rates=dpr[sum(layers[:3]):sum(layers[:4])],
                                            dilate=replace_stride_with_dilation[2], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)


        else:
            self.layer1 = self._make_layer(block, embed_dims[0], layers[0], drop_path_rates=dpr[:layers[0]], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)
            # print(sum(p.numel() for p in self.layer1.parameters()))

            self.layer2 = self._make_layer(block, embed_dims[1], layers[1], stride=2, drop_path_rates=dpr[sum(layers[:1]):sum(layers[:2])], 
                                            dilate=replace_stride_with_dilation[0], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer,
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)
            # print(sum(p.numel() for p in self.layer2.parameters()))


            self.layer3 = self._make_layer(block, embed_dims[2], layers[2], stride=2, drop_path_rates=dpr[sum(layers[:2]):sum(layers[:3])],
                                            dilate=replace_stride_with_dilation[1], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer,
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)
            # print(sum(p.numel() for p in self.layer3.parameters()))


            self.layer4 = self._make_layer(block, embed_dims[3], layers[3], stride=2, drop_path_rates=dpr[sum(layers[:3]):sum(layers[:4])],
                                            dilate=replace_stride_with_dilation[2], 
                                            depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer,
                                            del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                                            ls_init_values=ls_init_values)
            # print(sum(p.numel() for p in self.layer4.parameters()))


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_dim = embed_dims[3] * expansion if block == Bottleneck else embed_dims[3] // expansion
        if self.final_norm:
            self.norm = norm_layer(final_dim)

        self.fc = nn.Linear(final_dim, num_classes)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if init == 'trunc':
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, PermuteLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        drop_path_rates: List[float] = [-1],
        stride: int = 1,
        dilate: bool = False,
        depthwise: bool = False,
        kernel_size: int = 3,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        del_act: bool = False,
        reorg_norm: bool = False,
        pre_norm: bool = False,
        ls_init_values: float = 0.,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []

        if block == Bottleneck:
        # if True:
            if stride != 1 or self.inplanes != planes * expansion: # now this will only trigger when self.se_downsample is False
                print("constructing residual block downsampling at stage start")
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * expansion, stride),
                    norm_layer(planes * expansion),
                )
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                    drop_path_rate=drop_path_rates[0], 
                    depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                    del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                    ls_init_values=ls_init_values
                    )
                ) 

            self.inplanes = planes * expansion

        else:
            if stride != 1: # now this will only trigger when self.se_downsample is False
                print("constructing inverted residual block downsampling at stage start")
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes // expansion, stride),
                    norm_layer(planes // expansion)
                )
            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
                    drop_path_rate=drop_path_rates[0], 
                    depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                    del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                    ls_init_values=ls_init_values
                    )
                ) 
            self.inplanes = planes // expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    drop_path_rate=drop_path_rates[i],
                    depthwise=depthwise, kernel_size=kernel_size, act_layer=act_layer, 
                    del_act=del_act, reorg_norm=reorg_norm, pre_norm=pre_norm,
                    ls_init_values=ls_init_values
                )
            ) 

        return nn.Sequential(*layers)


    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.norm_ds1(self.ds1(x)) if self.sep_downsample else x


        x = self.layer2(x)
        x = self.norm_ds2(self.ds2(x)) if self.sep_downsample else x

        x = self.layer3(x)
        x = self.norm_ds3(self.ds3(x)) if self.sep_downsample else x

        x = self.layer4(x)
        x = self.avgpool(x)


        if self.final_norm:
            x = self.norm(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def get_num_layers(self):
        return 0

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}


