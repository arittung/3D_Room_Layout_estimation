import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchinfo import summary

from torch import nn


def lr_pad(x, padding=1):

    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)

class LR_PAD(nn.Module):
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding
    def forward(self, x):
        return lr_pad(x, self.padding)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetv1(nn.Module):
    def __init__(self, num_classes=100):
        super(MobileNetv1, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=2)
        self.separable_conv2 = SeparableConv2d(32, 64, stride=1)
        self.separable_conv3 = SeparableConv2d(64, 128, stride=2)
        self.separable_conv4 = SeparableConv2d(128, 128, stride=1)
        self.separable_conv5 = SeparableConv2d(128, 256, stride=2)
        self.separable_conv6 = SeparableConv2d(256, 256, stride=1)
        self.separable_conv7 = SeparableConv2d(256, 512, stride=2)
        self.separable_conv8 = SeparableConv2d(512, 512, stride=1)
        self.separable_conv9 = SeparableConv2d(512, 512, stride=1)
        self.separable_conv10 = SeparableConv2d(512, 512, stride=1)
        self.separable_conv11 = SeparableConv2d(512, 512, stride=1)
        self.separable_conv12 = SeparableConv2d(512, 512, stride=1)
        self.separable_conv13 = SeparableConv2d(512, 1024, stride=2)
        self.separable_conv14 = SeparableConv2d(1024, 1024, stride=2)

        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.separable_conv2(x)
        x = self.separable_conv3(x)
        x = self.separable_conv4(x)
        x = self.separable_conv5(x)
        x = self.separable_conv6(x)
        x = self.separable_conv7(x)
        x = self.separable_conv8(x)
        x = self.separable_conv9(x)
        x = self.separable_conv10(x)
        x = self.separable_conv11(x)
        x = self.separable_conv12(x)
        x = self.separable_conv13(x)
        x = self.separable_conv14(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet_v1():
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetv1()

    return model

class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(

            LR_PAD(),
            nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//2, in_c//2, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//4),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//4, out_c, kernel_size=3, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class HorizonNet(nn.Module):
        x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
        x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

        def __init__(self, backbone, use_rnn):
            super(HorizonNet, self).__init__()
            self.backbone = backbone
            self.use_rnn = use_rnn
            if backbone == 'mobilenet':
                self.feature_extractor = mobilenet_v1()
                _exp = 4
            else:
                raise NotImplementedError()

            _out_scale = 8
            self.stage1 = nn.ModuleList([
                GlobalHeightConv(64 * _exp, int(64 * _exp / _out_scale)),
                GlobalHeightConv(128 * _exp, int(128 * _exp / _out_scale)),
                GlobalHeightConv(256 * _exp, int(256 * _exp / _out_scale)),
                GlobalHeightConv(512 * _exp, int(512 * _exp / _out_scale)),
            ])
            self.step_cols = 4
            self.rnn_hidden_size = 512


            if self.use_rnn:
                #self.bi_rnn = nn.LSTM(input_size=_exp * 256,
                #                      hidden_size=self.rnn_hidden_size,
                #                      num_layers=2,
                #                      dropout=0.5,
                #                      batch_first=False,
                #                      bidirectional=True)
                self.bi_rnn = nn.GRU(input_size=_exp * 256,
                                        hidden_size = self.rnn_hidden_size,
                                        num_layers = 2,
                                        dropout = 0.5,
                                        batch_first = False,
                                        bidirectional = True)
                self.drop_out = nn.Dropout(0.5)
                self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                        out_features=3 * self.step_cols)

                self.linear.bias.data[0::4].fill_(-1)
                self.linear.bias.data[4::8].fill_(-0.478)
                self.linear.bias.data[8::12].fill_(0.425)
            else:
                self.linear = nn.Sequential(
                    nn.Linear(_exp * 256, self.rnn_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(self.rnn_hidden_size, 3 * self.step_cols),
                )
                self.linear[-1].bias.data[0::4].fill_(-1)
                self.linear[-1].bias.data[4::8].fill_(-0.478)
                self.linear[-1].bias.data[8::12].fill_(0.425)
            self.x_mean.requires_grad = False
            self.x_std.requires_grad = False



        def freeze_bn(self):
            for m in self.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        def _prepare_x(self, x):
            if self.x_mean.device != x.device:
                self.x_mean = self.x_mean.to(x.device)
                self.x_std = self.x_std.to(x.device)
            return (x[:, :3] - self.x_mean) / self.x_std

        def forward(self, x):
            x = self._prepare_x(x)
            iw = x.shape[3]
            block_w = int(iw / self.step_cols)
            conv_list = self.feature_extractor(x)
            down_list = []

            for x, f in zip(conv_list, self.stage1):
                tmp = f(x, block_w)  # [b, c, h, w]
                flat = tmp.view(tmp.shape[0], -1, tmp.shape[3])  # [b, c*h, w]
                down_list.append(flat)
            feature = torch.cat(down_list, dim=1)  # [b, c*h, w]


            # rnn
            if self.use_rnn:
                feature = feature.permute(2, 0, 1)  # [w, b, c*h]
                output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
                output = self.drop_out(output)
                output = self.linear(output)  # [seq_len, b, 3 * step_cols]
                output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
                output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
                output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
            else:
                feature = feature.permute(0, 2, 1)  # [b, w, c*h]
                output = self.linear(feature)  # [b, w, 3 * step_cols]
                output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
                output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
                output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

            cor = output[:, :1]
            bon = output[:, 1:]

            return bon, cor

model = HorizonNet(backbone='mobilenet', use_rnn=True)
# (batch, channels, height, width)
summary(model)