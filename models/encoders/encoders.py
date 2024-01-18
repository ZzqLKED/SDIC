import math
import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, _upsample_add
from models.stylegan2.model import EqualLinear,ScaledLeakyReLU,EqualConv2d

import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .helpers import *
import math
import gc
import time
import timm
import torchvision.transforms as transforms


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)
        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class Encoder4Editing(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)


    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it



    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        features = c3
        for i in range(1, min(18,self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w

class SpatialAttFus(Module):
    def __init__(self,  opts=None):
        super(SpatialAttFus, self).__init__()
        self.conv_layer1 = Sequential(Conv2d(3, 32, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(32),
                                      PReLU(32))
        self.semantic = nn.Sequential(
            BasicConv(32, 64, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BasicConv(64, 512, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1), #stride=2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False) #stride=1
            )

        
        self.att = nn.Sequential(BasicConv(512, 512, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
                                 )       #512
        self.aff1 = nn.Sequential(BasicConv(512, 512, bn=True, relu=True, kernel_size=3,
                                 padding=1, stride=1, dilation=1),
                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
                                 )
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.aff2 = nn.Sequential(BasicConv(1024, 512, bn=True, relu=True, kernel_size=3,
                                 padding=1, stride=1, dilation=1),
                                 nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
                                 )
        self.aff1d = Affine1d(18,144) #face18 cars16
        
        self.condition_shift3 = nn.Sequential(
                    EqualConv2d(3, 18, 3, stride=1, padding=1, bias=True ),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(18, 18, 3, stride=1, padding=1, bias=True )) 

        self.grid_transform = transforms.RandomPerspective(distortion_scale=0.15, p=0.9)

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it



    def forward(self, x, G7,W):
        
        conditions = []
        feat1 = self.conv_layer1(x)
        feat2 = self.semantic(feat1) 
        feat3 = self.att(feat2+G7) 
        scales=[]
        shifts=[]
        shift = self.condition_shift3(x)
        shift = torch.nn.functional.interpolate(shift, size=(64,64) , mode='bilinear')
        shift = shift.view([shift.shape[0],-1,512])
        F = torch.sigmoid(feat3)*feat2+G7
        Wp = self.aff1d(W,shift)

        return  F,Wp

class Affine1d(Module):
    def __init__(self, cv_chan, im_chan):
        super(Affine1d, self).__init__()

        self.sen = nn.Sequential(nn.Conv1d(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0) ,
                                nn.BatchNorm1d( im_chan//2),
                                nn.LeakyReLU(),
                                nn.Conv1d(im_chan//2, cv_chan, kernel_size=1))
        self.att = nn.Sequential(nn.Conv1d(cv_chan, cv_chan, 3, 1,1) ,
                                nn.BatchNorm1d(cv_chan),
                                nn.LeakyReLU(),
                                nn.Conv1d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.att2 = nn.Sequential(nn.Conv1d(cv_chan, cv_chan, 3, 1,1) ,
                                nn.BatchNorm1d(cv_chan),
                                nn.LeakyReLU(),
                                nn.Conv1d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.agg = nn.Sequential(nn.Conv1d(cv_chan, cv_chan, 3, 1,1) ,
                                nn.BatchNorm1d(cv_chan),
                                nn.LeakyReLU())


    def forward(self, cv, feat):
        feat = self.sen(feat)
        att = self.att(feat)
        att2 = self.att2(feat)
        cv = att*cv + cv+att2
        return cv


class DIP(nn.Module):
    def __init__(self, maxdisp):
        super(DIP, self).__init__()
        self.maxdisp = maxdisp 
        self.feature1 = Feature()
        self.feature2 = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.rgb1 = nn.Sequential(
            BasicConv(48, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False ),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.rgb = Sequential(*[bottleneck_IR(16,3,1), bottleneck_IR(3,3,1), bottleneck_IR(3,3,1)])

        self.conv = BasicConv(32, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 24, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg = BasicConv(8, 8, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.hourglass_fusion = hourglass_fusion(8)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
    
    def forward(self, left, right):
        features_left = self.feature1(left)
        features_right = self.feature2(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        corr_volume = torch.cat((match_left, match_right),1)# 4 48 256 256
        corr_volume=torch.unsqueeze(corr_volume,1)# 4 1 48 256 256
        volume = self.corr_stem(corr_volume) # 4 8 48 256 256
        cost = self.hourglass_fusion(volume, features_left) # [4, 1, 48, 256, 256]
        cost = torch.squeeze(cost,1)

        pred_up = self.rgb1(cost)
        pred_up = self.rgb(pred_up)

        return pred_up
        

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        # BasicConv have bn and RELU
        self.fe1 = BasicConv(3, 16, kernel_size=3, stride=1, padding=1) #16 256 256
        self.fe2 = BasicConv(16, 24, kernel_size=3, stride=2, padding=1) # 24 128 128
        self.fe3 = BasicConv(24, 32, kernel_size=3, stride=1, padding=1) # 32 128 128
        self.fe4 = BasicConv(32, 96, kernel_size=3, stride=2, padding=1) # 96 64 64
        self.fe5 = BasicConv(96, 160, kernel_size=3, stride=2, padding=1) # 160 32 32

    def forward(self, x):
        x = self.fe1(x)
        x2 = self.fe2(x)
        x2 = self.fe3(x2)
        x4 = self.fe4(x2)
        x8 = self.fe5(x4)

        return [x, x2, x4, x8]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        
        self.deconv8_4 = Conv2x(chans[4], chans[3], deconv=True, concat=True) #192x64x64
        self.deconv4_2 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)#64x128x128
        self.deconv2_1 = Conv2x(chans[2]*2, chans[0], deconv=True, concat=True)#32x256x256
        self.conv4 = BasicConv(chans[0]*2, chans[0]*2, kernel_size=3, stride=1, padding=1)#32x256x256

    def forward(self, featL, featR=None):
        x, x2, x4, x8  = featL

        y, y2, y4, y8  = featR

        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x2 = self.deconv4_2(x4, x2)
        y2 = self.deconv4_2(y4, y2)
        x = self.deconv2_1(x2, x)
        y = self.deconv2_1(y2, y)
        x = self.conv4(x)
        y = self.conv4(y)

        return [x, x2, x4, x8], [y, y2, y4, y8]


class Context_Geometry_Fusion(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(Context_Geometry_Fusion, self).__init__()

        self.semantic = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.att = nn.Sequential(BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                                 padding=(0,2,2), stride=1, dilation=1),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.agg = BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                             padding=(0,2,2), stride=1, dilation=1)


    def forward(self, cv, feat):
        feat = self.semantic(feat).unsqueeze(2)
        att = self.att(feat+cv)
        cv = torch.sigmoid(att)*feat + cv
        cv = self.agg(cv)
        return cv


class hourglass_fusion(nn.Module):

    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 160) #160
        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192) #192
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64) #64

    def forward(self, x, imgs):
        conv1 = self.conv1(x)  #4 16  48 128 128
        conv2 = self.conv2(conv1) #4 32 48 64 64
        conv3 = self.conv3(conv2) #4 48 48 32 32

        conv3 = self.CGF_32(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3) #4 32 48 64 64

        conv2 = torch.cat((conv3_up, conv2), dim=1)#4 64 48 64 64
        conv2 = self.agg_0(conv2)#4 32 48 64 64

        conv2 = self.CGF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2) #4 16 48 128 128

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        
        conv1 = self.CGF_8(conv1, imgs[1])
        conv = self.conv1_up(conv1)# 4 8 48 256 256

        return conv

