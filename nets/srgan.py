import math
import torch
from torch import nn


#----------------------------------------#
#   残差块,通道宽高都不变
#   主干: 3x3Conv -> BN -> PReLU -> 3x3Conv -> BN
#   直接拼接输入的x
#----------------------------------------#
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x + short_cut


#----------------------------------------#
#   上采样,通道不变,宽高翻倍
#   PixelShuffle
#   >>> pixel_shuffle = nn.PixelShuffle(3)
#   >>> input = torch.randn(1, 9, 4, 4)     3**2=9
#   >>> output = pixel_shuffle(input)       9 -> 1
#   >>> print(output.size())
#   torch.Size([1, 1, 12, 12])
#----------------------------------------#
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):  # up_scale = 2
        super(UpsampleBLock, self).__init__()
        #----------------------------------------#
        #   假设in_channels=64
        #   64, 64*2**2=256
        #----------------------------------------#
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        #----------------------------------------#
        #   channnel: 256 -> 64
        #   宽高翻倍
        #----------------------------------------#
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        x = self.conv(x)            # [2, 64, 56, 56] -> [2, 256, 56, 56]
        x = self.pixel_shuffle(x)   # [2,256, 56, 56] -> [2, 64, 112,112]
        x = self.prelu(x)
        return x


#----------------------------------------#
#   生成器
#   进过多次卷积和残差块,再上采样2次,最后到的高分辨率图像
#----------------------------------------#
class Generator(nn.Module):
    def __init__(self, scale_factor, num_residual=16):  # scale_factor = 4
        # 2**2 = 4 重复2次上采样
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()

        #----------------------------------------#
        #   9x9Conv
        #   [b, 3, h, w] -> [b, 64, h, w]
        #----------------------------------------#
        self.block_in = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(64)
        )

        #----------------------------------------#
        #   重复16次残差快
        #   [b, 64, h, w] -> [b, 64, h, w]
        #----------------------------------------#
        self.blocks = []
        for _ in range(num_residual):
            self.blocks.append(ResidualBlock(64))
        self.blocks = nn.Sequential(*self.blocks)

        #----------------------------------------#
        #   3x3Conv
        #   [b, 64, h, w] -> [b, 64, h, w]
        #----------------------------------------#
        self.block_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        #----------------------------------------#
        #   重复2次上采样(scaler=2) + 3x3Conv
        #   [b, 64, h, w] -> [b, 64,4h,4w] -> [b, 3, 4h,4w]
        #----------------------------------------#
        self.upsample = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        self.upsample.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample = nn.Sequential(*self.upsample)

    def forward(self, x):
        #----------------------------------------#
        #   [b, 3, h, w]  -> [b, 64, h, w]
        #----------------------------------------#
        x = self.block_in(x)
        short_cut = x
        #----------------------------------------#
        #   [b, 64, h, w] -> [b, 64, h, w]
        #----------------------------------------#
        x = self.blocks(x)
        x = self.block_out(x)
        #----------------------------------------#
        #   上采样之前和转换通道的x进行相加
        #   [b, 64, h, w]  + [b, 64, h, w]  = [b, 64, h, w]
        #   [b, 64, h, w] -> [b, 64,4h,4w] -> [b, 3, 4h,4w]
        #----------------------------------------#
        upsample = self.upsample(x + short_cut)
        return torch.tanh(upsample)


#----------------------------------------#
#   辨别器
#   将图像经过多次下采样和最后的pooling,得到维度为1的结果
#----------------------------------------#
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),             # [b, 3, h, w] -> [b, 64, h, w]
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # [b, 64, h, w] -> [b, 64, h/2, w/2]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # [b, 64, h/2, w/2] -> [b, 128, h/2, w/2]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# [b, 128, h/2, w/2] -> [b, 128, h/4, w/4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),          # [b, 128, h/4, w/4] -> [b, 256, h/4, w/4]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),# [b, 256, h/4, w/4] -> [b, 256, h/8, w/8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),          # [b, 256, h/8, w/8] -> [b, 512, h/8, w/8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),# [b, 512, h/8, w/8] -> [b, 512, h/16, w/16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),                                # [b, 512, h/16, w/16]-> [b, 512, 1, 1]
            nn.Conv2d(512, 1024, kernel_size=1),                    # [b, 512, 1, 1]-> [b, 1024, 1, 1]
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)                       # [b, 1024, 1, 1]-> [b, 1, 1, 1]
        )

    def forward(self, x):
        batch_size = x.size(0)
        # [b, c, h, w] -> [b, 1, 1, 1] -> [b]
        return self.net(x).view(batch_size)


if __name__ == "__main__":
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(4).to(device)
    summary(generator, input_size=(3,56,56))

    x = torch.randn(1, 3, 56, 56).cuda()
    y = generator(x)
    print(y.size())  # [1, 3, 224, 224]

    discriminator = Discriminator().cuda()
    z = discriminator(y)
    print(z.size()) # [1]