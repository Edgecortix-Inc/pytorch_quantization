import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d,
            norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d,
                norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        relu6 = self.relu6(self.float_op.add_scalar(x, 3.))
        return self.float_op.mul(x, self.float_op.mul_scalar(relu6, 1/6.))


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        relu6 = self.relu6(self.float_op.add_scalar(x, 3.))
        return self.float_op.mul_scalar(relu6, 1/6.)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )
        self.fmul = nn.quantized.FloatFunctional()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.fmul.mul(x, y.expand_as(x))


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp),
            nlin_layer(inplace=True),
            # dw
            nn.Conv2d(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224,
                 dropout=0.2, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0

        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(input_channel),
                        Hswish(inplace=True))]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(nn.Conv2d(input_channel, last_conv, 1, 1, 0))
            self.features.append(nn.BatchNorm2d(last_conv))
            self.features.append(Hswish(inplace=True))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(nn.Sequential(nn.Conv2d(input_channel, last_conv, 1, 1, 0, bias=False),
                                 nn.BatchNorm2d(last_conv),
                                 Hswish(inplace=True)))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # seq = list(list(raw_model.children())[0].children())[-4]
    def fuse_model(self):
        for m in self.modules():
            if type(m) == SEModule:
               for idx in range(len(m.fc)):
                   if type(m.fc[idx]) == nn.Linear:
                       fuse_modules(m.fc, [str(idx), str(idx+1)], inplace=True)
                       break
            if type(m) == MobileBottleneck:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        fuse_modules(m.conv, [str(idx), str(idx+1)], inplace=True)

                    if type(m.conv[idx]) == nn.BatchNorm2d:
                        fuse_modules(m.conv, [str(idx-1), str(idx)], inplace=True)
            if type(m) == nn.Sequential and len(list(m.children())) == 3:
                fuse_modules(m, ['0', '1'], inplace=True)


def load_model(model_file):
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    model = MobileNetV3()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace(".module", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to('cpu')
    return model
