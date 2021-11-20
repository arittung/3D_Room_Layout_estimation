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

class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()


        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1)

            #nn.AdaptiveAvgPool2d(1)
        )
        #self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)

        #x = x.view(-1, 1024)
       # x = self.fc(x)
        #print(x.size())
        return x


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(

            LR_PAD(),
            nn.Conv2d(in_c, in_c//2, kernel_size=4, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//2, in_c//2, kernel_size=4, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//2),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//2, in_c//4, kernel_size=4, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_c//4),
            nn.ReLU(inplace=True),
            LR_PAD(),
            nn.Conv2d(in_c//4, out_c, kernel_size=4, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        #print(self.layer)

    def forward(self, x, out_w):

        ##############
        print("before")

        print(x.size())
        x = self.layer(x)
        print("after")
        print(x.size())
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
        if backbone == 'mobilenet_v1':
            self.feature_extractor = MobileNetV1(ch_in=3, n_classes=1000)
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
        # print(self.stage1)
        self.step_cols = 4
        self.rnn_hidden_size = 512

        if self.use_rnn:
            # self.bi_rnn = nn.LSTM(input_size=_exp * 256,
            #                      hidden_size=self.rnn_hidden_size,
            #                      num_layers=2,
            #                      dropout=0.5,
            #                      batch_first=False,
            #                      bidirectional=True)
            self.bi_rnn = nn.GRU(input_size=_exp * 256,
                                    hidden_size=self.rnn_hidden_size,
                                    num_layers=2,
                                    dropout=0.5,
                                    batch_first=False,
                                    bidirectional=True)
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
            #print(x.size())
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



if __name__ == '__main__':
    model = MobileNetV1(ch_in=3, n_classes=1000)
    model = model.to("cuda:0")

    summary(model, input_size=(1, 3, 512, 1024))


    model2 = HorizonNet(backbone='mobilenet_v1', use_rnn=True)
    model2 = model2.to("cuda:0")
    summary(model2, input_size=(1, 3, 512, 1024))
# (batch, channels, height, width)
