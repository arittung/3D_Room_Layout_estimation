from torch.autograd import Variable
import torch.nn as nn
import torch
import math
from torchinfo import summary


def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def SepConv_3x3(inp, oup):  # input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def InvertedResidual(inp, oup, stride, expand_ratio, kernel):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(inp * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio,
                  bias=False),
        nn.BatchNorm2d(inp * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class MnasNet(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MnasNet, self).__init__()

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 128, 1, 2, 3],  # -> 56x56
            [3, 256, 1, 2, 5],  # -> 28x28
            [6, 512, 1, 2, 5],  # -> 14x14
            [6, 1024, 1, 1, 3],  # -> 14x14
            # [6, 192, 4, 2, 5],  # -> 7x7
            # [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layer
        self.features1 = [Conv_3x3(3, input_channel, 2), SepConv_3x3(input_channel, 16)]
        input_channel = 16
        self.features1 = nn.Sequential(*self.features1)

        # building inverted residual blocks (MBConv)
        output_channel = int(128 * width_mult)
        self.inverted_feature1 = []
        for i in range(3):
            if i == 0:
                self.inverted_feature1.append(InvertedResidual(input_channel, output_channel, 2, 3, 3))
            else:
                self.inverted_feature1.append(InvertedResidual(input_channel, output_channel, 1, 3, 3))
            input_channel = output_channel
        self.inverted_feature1 = nn.Sequential(*self.inverted_feature1)

        output_channel = int(256 * width_mult)
        self.inverted_feature2 = []
        for i in range(3):
            if i == 0:  # s, t, k
                self.inverted_feature2.append(InvertedResidual(input_channel, output_channel, 2, 3, 5))  # s, t, k
            else:
                self.inverted_feature2.append(InvertedResidual(input_channel, output_channel, 1, 3, 5))
            input_channel = output_channel
        self.inverted_feature2 = nn.Sequential(*self.inverted_feature2)

        output_channel = int(512 * width_mult)
        self.inverted_feature3 = []
        for i in range(3):
            if i == 0:  # s, t, k
                self.inverted_feature3.append(InvertedResidual(input_channel, output_channel, 2, 6, 5))  # s, t, k
            else:
                self.inverted_feature3.append(InvertedResidual(input_channel, output_channel, 1, 6, 5))
            input_channel = output_channel
        self.inverted_feature3 = nn.Sequential(*self.inverted_feature3)

        output_channel = int(1024 * width_mult)
        self.inverted_feature4 = []
        for i in range(2):
            if i == 0:  # s, t, k
                self.inverted_feature4.append(InvertedResidual(input_channel, output_channel, 1, 6, 3))  # s, t, k
            else:
                self.inverted_feature4.append(InvertedResidual(input_channel, output_channel, 1, 6, 3))
            input_channel = output_channel
        self.inverted_feature4 = nn.Sequential(*self.inverted_feature4)

        output_channel = int(192 * width_mult)
        self.inverted_feature5 = []
        for i in range(4):
            if i == 0:  # s, t, k
                self.inverted_feature5.append(InvertedResidual(input_channel, output_channel, 2, 6, 5))  # s, t, k
            else:
                self.inverted_feature5.append(InvertedResidual(input_channel, output_channel, 1, 6, 5))
            input_channel = output_channel
        self.inverted_feature5 = nn.Sequential(*self.inverted_feature5)

        output_channel = int(320 * width_mult)
        self.inverted_feature6 = []
        for i in range(1):
            if i == 0:  # s, t, k
                self.inverted_feature6.append(InvertedResidual(input_channel, output_channel, 1, 6, 3))  # s, t, k
            else:
                self.inverted_feature6.append(InvertedResidual(input_channel, output_channel, 1, 6, 3))
            input_channel = output_channel
        self.inverted_feature6 = nn.Sequential(*self.inverted_feature6)

        # building last several layers
        self.features2 = []
        self.features2.append(Conv_1x1(input_channel, self.last_channel))
        # self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential

        self.features2 = nn.Sequential(*self.features2)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        conv_list =[]
        x = self.features1(x)
        x = self.inverted_feature1(x); conv_list.append(x); #print(x.shape)
        x = self.inverted_feature2(x); conv_list.append(x); #print(x.shape)
        x = self.inverted_feature3(x); conv_list.append(x); #print(x.shape)
        x = self.inverted_feature4(x); conv_list.append(x); #print(x.shape)
        x = self.inverted_feature5(x)
        x = self.inverted_feature6(x)

        x = self.features2(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return conv_list

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# if __name__ == '__main__':
#     net = MnasNet()
#     x_image = Variable(torch.randn(1, 3, 224, 224))
#     y = net(x_image)
#     #print(y)