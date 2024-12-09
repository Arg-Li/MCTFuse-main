import torch.nn as nn
import torch
from sympy import pprint


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if activation_type == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif hasattr(nn, activation_type.capitalize()):
        return getattr(nn, activation_type.capitalize())()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Sequential(nn.ReflectionPad2d(1),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2)
        # self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0))
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class VIEncoder(nn.Module):
    def __init__(self, n_channels=1, o_channels=1):
        '''
        n_channels : number of channels of the input.
        n_labels : number of channels of the ouput.
        '''
        super().__init__()
        self.n_channels = n_channels # 1
        self.o_classes = o_channels  # 1
        # Question here
        block = Res_block
        in_channels = 32
        self.pool = nn.AvgPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, o_channels, kernel_size=(1,1))
        self.last_activation = nn.ReLU()

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        x5 = self.down_encoder4(self.pool(x4))
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.last_activation(self.outc(x))

        return out

    def fuse_foward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        return x1, x2, x3, x4

class IREncoder(nn.Module):
    def __init__(self, n_channels=1, o_channels=1):
        '''
        n_channels : number of channels of the input.
        n_labels : number of channels of the ouput.
        '''
        super().__init__()
        self.n_channels = n_channels # 1
        self.o_classes = o_channels  # 1
        # Question here
        block = Res_block
        in_channels = 32
        self.pool = nn.AvgPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.down_encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)
        self.down_encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)
        self.down_encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)
        self.down_encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, o_channels, kernel_size=(1,1))
        self.last_activation = nn.ReLU()

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        x5 = self.down_encoder4(self.pool(x4))
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.last_activation(self.outc(x))

        return out

    def fuse_foward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        return x1, x2, x3, x4

if __name__ == '__main__':
    x = torch.randn(1, 1, 575, 475).cuda()
    y = torch.randn(1, 1, 128, 128).cuda()
    # ir_net = IREncoder().cuda()
    vi_net =IREncoder().cuda()
    out = vi_net(y)

