# -*- coding: utf-8 -*-
# @Author : YYlin
# @mailbox: ${854280599@qq.com}
# @Time : 2022/2/10 20:28
# @FileName: model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from dual_attention import DANetHead
from torchvision import models
from functools import partial
nonlinearity = partial(F.relu,inplace=True)
from math import exp


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=256, conv_dim=64, repeat_num=6, atten='none', class_num=3):
        super(Discriminator, self).__init__()
        self.class_num = class_num

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convc = nn.Conv2d(curr_dim, self.class_num, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.convc(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


class Discriminator_No_Classifier(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Discriminator_No_Classifier, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode='bilinear', align_corners=False))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        if self.dim_in == self.dim_out:
            return x + self.main(x)
        else:
            return self.main(x)


# no crossing

# 没有使用attention gate连接ppm
class Full_Model(nn.Module):

    def __init__(self, atten='none'):
        super().__init__()

        self.resnet = models.resnet34(pretrained=True)

        ##########################
        # Encoder part - RESNET34
        ##########################
        # stage 0
        self.encoder0 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # stage 1
        self.encoder1 = self.resnet.layer1
        # stage 2
        self.encoder2 = self.resnet.layer2
        # stage 3
        self.encoder3 = self.resnet.layer3
        # stage 4
        self.encoder4 = self.resnet.layer4
        # stage 5
        self.encoder5 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512))
        # stage 6
        self.encoder6 = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512),
            BasicBlock(512, 512))

        ##########################
        ### Decoder part - GLANCE
        ##########################
        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp6 = conv_up_psp(512, 512, 2)
        self.psp5 = conv_up_psp(512, 512, 4)
        self.psp4 = conv_up_psp(512, 256, 8)
        self.psp3 = conv_up_psp(512, 128, 16)
        self.psp2 = conv_up_psp(512, 64, 32)
        self.psp1 = conv_up_psp(512, 64, 32)

        # stage 6g
        self.decoder6_g = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att6_g = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN6 = ResidualBlock(1024, 512)
        self.Att6_g_f = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Att6_f_g = Attention_block(F_g=512, F_l=512, F_int=256)

        # stage 5g
        self.decoder5_g = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att5_g = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = ResidualBlock(1024, 512)
        self.Att5_g_f = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Att5_f_g = Attention_block(F_g=512, F_l=512, F_int=256)

        # stage 4g
        self.decoder4_g = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att4_g = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = ResidualBlock(512, 256)
        self.Att4_g_f = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Att4_f_g = Attention_block(F_g=256, F_l=256, F_int=128)

        # stage 3g
        self.decoder3_g = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att3_g = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = ResidualBlock(256, 128)
        self.Att3_g_f = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att3_f_g = Attention_block(F_g=128, F_l=128, F_int=64)

        # stage 2g
        self.decoder2_g = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att2_g = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = ResidualBlock(128, 64)
        self.Att2_g_f = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att2_f_g = Attention_block(F_g=64, F_l=64, F_int=32)

        # stage 1g
        self.decoder1_g = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.Att1_g = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN1 = ResidualBlock(128, 64)
        self.Att1_g_f = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att1_f_g = Attention_block(F_g=64, F_l=64, F_int=32)

        # stage 0g
        self.decoder0_g = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1))

        ##########################
        ### Decoder part - FOCUS
        ##########################
        self.decoder6_f = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att6_f = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN6_f = ResidualBlock(1024, 512)

        # stage 5f
        self.decoder5_f = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att5_f = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5_f = ResidualBlock(1024, 512)

        # stage 4f
        self.decoder4_f = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att4_f = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4_f = ResidualBlock(512, 256)

        # stage 3f
        self.decoder3_f = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att3_f = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3_f = ResidualBlock(256, 128)

        # stage 2f
        self.decoder2_f = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.Att2_f = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2_f = ResidualBlock(128, 64)

        # stage 1f
        self.decoder1_f = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.Att1_f = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN1_f = ResidualBlock(128, 64)

        # stage 0f
        self.decoder0_f = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, input):
        input_image = input

        e0 = self.encoder0(input_image)

        # e0: N, 64, H, W
        e1 = self.encoder1(e0)
        # e1: N, 64, H, W
        e2 = self.encoder2(e1)
        # e2: N, 128, H/2, W/2
        e3 = self.encoder3(e2)
        # e3: N, 256, H/4, W/4
        e4 = self.encoder4(e3)

        # e4: N, 512, H/8, W/8
        e5 = self.encoder5(e4)
        # e5: N, 512, H/16, W/16
        e6 = self.encoder6(e5)

        # generative model
        psp = self.psp_module(e6)
        d6_g = self.decoder6_g(torch.cat((psp, e6), 1))
        e5_g = self.Att6_g(g=d6_g, x=e5)
        d6_g = torch.cat((e5_g, d6_g), dim=1)
        d6_g = self.Up_RRCNN6(d6_g)

        # attention model
        # d6_g: N, 512, H/16, W/16
        d6_f = self.decoder6_f(e6)
        e5_f = self.Att6_g(g=d6_f, x=e5)
        d6_f = torch.cat((e5_f, d6_f), dim=1)
        d6_f = self.Up_RRCNN6_f(d6_f)

        # 第五层
        d6_g = self.Att6_g_f(d6_g, d6_f)
        d5_g = self.decoder5_g(torch.cat((self.psp6(psp), d6_g), 1))
        e4_g = self.Att5_g(g=d5_g, x=e4)
        d5_g = torch.cat((e4_g, d5_g), dim=1)
        d5_g = self.Up_RRCNN5(d5_g)

        d6_f = self.Att6_f_g(d6_f, d6_g)
        d5_f = self.decoder5_f(torch.cat((d6_f, e5), 1))
        e4_f = self.Att5_f(g=d5_f, x=e4)
        d5_f = torch.cat((e4_f, d5_f), dim=1)
        d5_f = self.Up_RRCNN5_f(d5_f)

        # 第四层
        d5_g = self.Att5_g_f(d5_g, d5_f)
        d4_g = self.decoder4_g(torch.cat((self.psp5(psp), d5_g), 1))
        e3_g = self.Att4_g(g=d4_g, x=e3)
        d4_g = torch.cat((e3_g, d4_g), dim=1)
        d4_g = self.Up_RRCNN4(d4_g)

        d5_f = self.Att5_f_g(d5_f, d5_g)
        d4_f = self.decoder4_f(torch.cat((d5_f, e4), 1))
        e3_f = self.Att4_f(g=d4_f, x=e3)
        d4_f = torch.cat((e3_f, d4_f), dim=1)
        d4_f = self.Up_RRCNN4_f(d4_f)

        # 第三层
        d4_g = self.Att4_g_f(d4_g, d4_f)
        d3_g = self.decoder3_g(torch.cat((self.psp4(psp), d4_g), 1))
        e2_g = self.Att3_g(g=d3_g, x=e2)
        d3_g = torch.cat((e2_g, d3_g), dim=1)
        d3_g = self.Up_RRCNN3(d3_g)

        d4_f = self.Att4_f_g(d4_f, d4_g)
        d3_f = self.decoder3_f(torch.cat((d4_f, e3), 1))
        e2_f = self.Att3_f(g=d3_f, x=e2)
        d3_f = torch.cat((e2_f, d3_f), dim=1)
        d3_f = self.Up_RRCNN3_f(d3_f)

        # 第二层
        d3_g = self.Att3_g_f(d3_g, d3_f)
        d2_g = self.decoder2_g(torch.cat((self.psp3(psp), d3_g), 1))
        e1_g = self.Att2_g(g=d2_g, x=e1)
        d2_g = torch.cat((e1_g, d2_g), dim=1)
        d2_g = self.Up_RRCNN2(d2_g)

        d3_f = self.Att3_f_g(d3_f, d3_g)
        d2_f = self.decoder2_f(torch.cat((d3_f, e2), 1))
        e1_f = self.Att2_f(g=d2_f, x=e1)
        d2_f = torch.cat((e1_f, d2_f), dim=1)
        d2_f = self.Up_RRCNN2_f(d2_f)

        # 第一层
        d2_g = self.Att2_g_f(d2_g, d2_f)
        d1_g = self.decoder1_g(torch.cat((self.psp2(psp), d2_g), 1))
        e0_g = self.Att1_g(g=d1_g, x=e0)
        d1_g = torch.cat((e0_g, d1_g), dim=1)
        d1_g = self.Up_RRCNN1(d1_g)

        d2_f = self.Att2_f_g(d2_f, d2_g)
        d1_f = self.decoder1_f(torch.cat((d2_f, e1), 1))
        e0_f = self.Att1_f(g=d1_f, x=e0)
        d1_f = torch.cat((e0_f, d1_f), dim=1)
        d1_f = self.Up_RRCNN1_f(d1_f)

        # d1_g: N, 64, H, W
        content_mask = self.decoder0_g(d1_g)
        attention_mask = torch.sigmoid(self.decoder0_f(d1_f))

        # result =  torch.tanh(content_mask)
        result = content_mask * attention_mask + input * (1 - attention_mask)

        return result, attention_mask, content_mask, e6


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_Unet(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self, n1=32, in_ch=1, out_ch=1, atten='none'):
        super(Attention_Unet, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.atten = atten

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = ResidualBlock(in_ch, filters[0])
        self.RRCNN2 = ResidualBlock(filters[0], filters[1])
        self.RRCNN3 = ResidualBlock(filters[1], filters[2])
        self.RRCNN4 = ResidualBlock(filters[2], filters[3])
        self.RRCNN5 = ResidualBlock(filters[3], filters[4])

        # 以下为
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = ResidualBlock(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = ResidualBlock(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = ResidualBlock(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = ResidualBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # input_image = x
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4_v1 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4_v1, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3_v1 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3_v1, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2_v1 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2_v1, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1_v1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1_v1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        output = F.sigmoid(self.Conv(d2))
        return output




#

