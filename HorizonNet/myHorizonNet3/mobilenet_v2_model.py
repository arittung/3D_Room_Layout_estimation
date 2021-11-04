import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchinfo import summary

from torch import nn



__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def lr_pad(x, padding=1):

    #print(x)

    #print(x.size())

#    x=x.reshape(1,3,512, 1024)
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=0)

class LR_PAD(nn.Module):
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding
    def forward(self, x):
        return lr_pad(x, self.padding)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['mobilenet_v2']), strict=False)
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenet_v2']), strict=False)
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
        #print(x)
        #print(x.size())
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
                self.feature_extractor = mobilenet_v2(True)
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

            ### 여기 feature_extractor(x)에서 1000으로 바뀌기 시작한다...ㅠㅠㅠ
            conv_list = self.feature_extractor(x)
            down_list = []

            for x, f in zip(conv_list, self.stage1):
                print(x.size())
                #rint(conv_list)
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