from utils import *
from network.MCTrans import ChannelTransformer

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

class ConvLastBlock(nn.Module):
    """Upscaling one tensor then conv"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(ConvLastBlock, self).__init__()
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        return self.nConvs(x)

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

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', train_flag = True):
        super(UpBlock, self).__init__()

        # todo
        self.up = nn.PixelShuffle(2)
        self.train_flag = train_flag
        self.shape_adjust = UpsampleReshape_eval()
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        if not self.train_flag:
            out = self.shape_adjust(skip_x, out)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class TransNet(nn.Module):
    def __init__(self, config, n_channels=1, o_channels=1,img_size=128,vis=False, train_flag = True):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.o_channels = o_channels
        in_channels = config.base_channel
        self.train_flag = train_flag
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes, train_flag = self.train_flag)
        self.out4 = ConvLastBlock(in_channels * 8, in_channels * 8, nb_Conv=2, activation='ReLU')
        self.up3 = UpBlock(in_channels * 6, in_channels * 4, nb_Conv=2, activation='ReLU', train_flag = self.train_flag)
        self.up2 = UpBlock(in_channels * 3, in_channels * 2, nb_Conv=2, activation='ReLU', train_flag = self.train_flag)
        self.up1 = UpBlock(in_channels + in_channels // 2, in_channels, nb_Conv=2, activation='ReLU', train_flag = self.train_flag)
        self.outc = nn.Conv2d(in_channels, o_channels, kernel_size=(1, 1))
        self.last_activation = nn.Sigmoid()
        self.shape_adjust = UpsampleReshape_eval()

    def forward(self, vi, ir):
        x1,x2,x3,x4,att_weights = self.mtc(vi, ir)
        x = self.out4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if not self.train_flag:
            x = self.shape_adjust(vi[0], x)
        out = self.outc(x)
        out = self.last_activation(out)
        return out

if __name__ == '__main__':
    config = get_CTranS_config()
    x1 = torch.rand(1, 32, 128, 128)
    x2 = torch.rand(1, 64, 64, 64)
    x3 = torch.rand(1, 128, 32, 32)
    x4 = torch.rand(1, 256, 16, 16)

    vi = (x1, x2, x3, x4)
    ir = (x1, x2, x3, x4)
    net = TransNet(config, train_flag = False)
    out = net(vi, ir)
    print(out.shape)

