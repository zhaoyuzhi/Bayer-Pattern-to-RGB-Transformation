import torch
import torch.nn as nn
import torch.nn.functional as F

from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
class UResNet444(nn.Module):
    def __init__(self, opt):
        super(UResNet444, self).__init__()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottle neck
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm),
            ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        )
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 1, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'tanh')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 224 * 224
        E2 = self.E2(E1)                                        # out: batch * 128 * 112 * 112
        E3 = self.E3(E2)                                        # out: batch * 256 * 56 * 56
        E4 = self.E4(E3)                                        # out: batch * 512 * 28 * 28
        # Bottle neck
        E4 = self.BottleNeck(E4)                                # out: batch * 512 * 28 * 28
        # Decode the center code
        D1 = self.D1(E4)                                        # out: batch * 256 * 56 * 56
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 56 * 56
        D2 = self.D2(D1)                                        # out: batch * 128 * 112 * 112
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 112 * 112
        D3 = self.D3(D2)                                        # out: batch * 64 * 224 * 224
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 224 * 224
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256

        return x
        
# ----------------------------------------
#               Discriminator
# ----------------------------------------
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.out_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 1, 1, pad_type = opt.pad, norm = opt.norm, sn = True)
        self.final2 = Conv2dLayer(opt.start_channels * 4, 1, 4, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 6 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 256 * 256
        x = self.block2(x)                                      # out: batch * 128 * 128 * 128
        x = self.block3(x)                                      # out: batch * 256 * 64 * 64
        x = self.block4(x)                                      # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x
