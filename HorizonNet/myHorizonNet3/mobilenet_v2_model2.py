import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchinfo import summary
import math
from torch import nn



__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1',
}


def lr_pad(x, padding=1):
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=0)

class LR_PAD(nn.Module):
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding
    def forward(self, x):
        return lr_pad(x, self.padding)


# 첫번째 layer에서 사용될 convolution 함수
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# inverted bottleneck layer 바로 다음에 나오는 convolution에 사용될 함수
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# channel수를 무조건 8로 나누어 떨어지게 만드는 함수
def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)  # expansion channel
        self.use_res_connect = self.stride == 1 and inp == oup  # skip connection이 가능한지 확인 True or False
        '''
            self.stride == 1 ----> 연산 전 후의 feature_map size가 같다는 의미
            inp == oup ----> 채널수도 동일하게 유지된다는 의미
            즉 skip connection 가능
        '''
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                # 확장시킬 필요가 없기 때문에 바로 depth wise conv
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw(확장)
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear(축소)
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            # t : expand ratio
            # c : channel
            # n : Number of iterations
            # s : stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)] # feature들을 담을 리스트에 first layer 추가

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))  # 반복되는 부분에서 skip connection 가능
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # Average pooling layer
        self.avg = nn.AvgPool2d(7, 7)
        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        # pdb.set_trace()
        x = self.features(x)
        x = self.avg(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    # 초기 weight 설정
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



def mobilenet_v2(pretrained=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2( **kwargs)
    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls['mobilenet_v2']))
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
            if backbone == 'mobilenet_v2':
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
                print(x.shape)

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

model = HorizonNet(backbone='mobilenet_v2', use_rnn=True)
# (batch, channels, height, width)
summary(model)