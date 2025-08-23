import torch.nn as nn
import math
import numpy as np
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F

class FractionalGaborFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super(FractionalGaborFilter, self).__init__()

        self.real_weights = nn.ParameterList()
        self.imag_weights = nn.ParameterList()

        for angle in angles:
            for scale in scales:
                # real_weight, imag_weight = self.generate_fractional_gabor(in_channels, out_channels, kernel_size, order, angle, scale)
                real_weight = self.generate_fractional_gabor(
                    in_channels, out_channels, kernel_size, order, angle, scale
                )
                self.real_weights.append(nn.Parameter(real_weight))
                # self.imag_weights.append(nn.Parameter(imag_weight))

    def generate_fractional_gabor(
            self, in_channels, out_channels, size, order, angle, scale
    ):
        x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
        x_theta = x * np.cos(angle) + y * np.sin(angle)
        y_theta = -x * np.sin(angle) + y * np.cos(angle)

        real_part = np.exp(
            -((x_theta ** 2 + (y_theta / scale) ** 2) ** order)
        ) * np.cos(2 * np.pi * x_theta / scale)
        # imag_part = np.exp(-((x_theta ** 2 + (y_theta / scale) ** 2) ** order)) * np.sin(2 * np.pi * x_theta / scale)

        # Reshape to match the specified out_channels and size
        real_weight = torch.tensor(real_part, dtype=torch.float32).view(
            1, 1, size[0], size[1]
        )
        # imag_weight = torch.tensor(imag_part, dtype=torch.float32).view(1, 1, size[0], size[1])

        # Repeat along the out_channels dimension
        real_weight = real_weight.repeat(out_channels, 1, 1, 1)
        # imag_weight = imag_weight.repeat(out_channels, 1, 1, 1)

        return real_weight  # , imag_weight

    def forward(self, x):
        real_weights = [weight for weight in self.real_weights]
        # imag_weights = [weight for weight in self.imag_weights]

        real_result = sum(weight * x for weight in real_weights)
        # imag_result = sum(weight * x for weight in imag_weights)

        return real_result  # - imag_result


class GaborSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super(GaborSingle, self).__init__()
        self.gabor = FractionalGaborFilter(
            in_channels, out_channels, kernel_size, order, angles, scales
        )
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.gabor(self.t)
        out = F.conv2d(x, out, stride=1, padding=1)
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(
            out, (1, 0, 1, 0), mode="constant", value=0
        )  # Padding on the left and top
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class GaborFPU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            order=0.25,
            angles=[0, 45, 90, 135],
            scales=[1, 2, 3, 4],
    ):
        super(GaborFPU, self).__init__()
        self.gabor1 = GaborSingle(
            in_channels // 4, out_channels // 4, (3, 3), order, angles, scales
        )
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat(
            [self.gabor1(x1), self.gabor1(x2), self.gabor1(x3), self.gabor1(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class FrFTFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super(FrFTFilter, self).__init__()

        self.register_buffer(
            "weight",
            self.generate_FrFT_filter(in_channels, out_channels, kernel_size, f, order),
        )

    def generate_FrFT_filter(self, in_channels, out_channels, kernel, f, p):
        N = out_channels
        d_x = kernel[0]
        d_y = kernel[1]
        x = np.linspace(1, d_x, d_x)
        y = np.linspace(1, d_y, d_y)
        [X, Y] = np.meshgrid(x, y)

        real_FrFT_filterX = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filterY = np.zeros([d_x, d_y, out_channels])
        real_FrFT_filter = np.zeros([d_x, d_y, out_channels])
        for i in range(N):
            real_FrFT_filterX[:, :, i] = np.cos(
                -f * (X) / math.sin(p) + (f * f + X * X) / (2 * math.tan(p))
            )
            real_FrFT_filterY[:, :, i] = np.cos(
                -f * (Y) / math.sin(p) + (f * f + Y * Y) / (2 * math.tan(p))
            )
            real_FrFT_filter[:, :, i] = (
                    real_FrFT_filterY[:, :, i] * real_FrFT_filterX[:, :, i]
            )
        g_f = np.zeros((kernel[0], kernel[1], in_channels, out_channels))
        for i in range(N):
            g_f[:, :, :, i] = np.repeat(
                real_FrFT_filter[:, :, i: i + 1], in_channels, axis=2
            )
        g_f = np.array(g_f)
        g_f_real = g_f.reshape((out_channels, in_channels, kernel[0], kernel[1]))
        return torch.tensor(g_f_real).type(torch.FloatTensor)

    def forward(self, x):
        x = x * self.weight
        return x

    def generate_FrFT_list(self, in_channels, out_channels, kernel, f_list, p):
        FrFT_list = []
        for f in f_list:
            FrFT_list.append(
                self.generate_FrFT_filter(in_channels, out_channels, kernel, f, p)
            )
        return FrFT_list


class FrFTSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.fft = FrFTFilter(in_channels, out_channels, kernel_size, f, order)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]),
            requires_grad=True,
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fft(self.t)
        out = F.conv2d(x, out, stride=1, padding=(1))
        out = self.relu(out)
        out = F.dropout(out, 0.3)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class FourierFPU(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25):
        super().__init__()
        self.fft1 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.25, order)
        self.fft2 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.50, order)
        self.fft3 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.75, order)
        self.fft4 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 1.00, order)
        self.fc = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, 1)
        x_out = torch.cat(
            [self.fft1(x1), self.fft2(x2), self.fft3(x3), self.fft4(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        out1 = self.act(self.pool(x))
        out2 = self.act(self.pool1(out1))
        out3 = self.act(out2+out1)
        return out3

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
#         max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
#         out = self.sigmoid(avg_out + max_out)
#         return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(
            self.cv1(
                torch.cat(
                    [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]],
                    1,
                )
            )
        )


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        c = self.conv(x)
        c = self.bn(c)
        c = self.act(c)
        return c


class SPU(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = Conv(c1 // 2, c2 // 2, 3)
        self.c2 = Conv(c1 // 2, c2 // 2, 5)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.c1(x1)
        x2 = self.c2(x2)
        x_out = torch.cat([x1, x1 + x2], dim=1)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class FrGTConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, order=0.25, filter="FrFT", attention="ca"
    ):
        super().__init__()
        self.PWC0 = Conv(in_channels, in_channels // 2, 1)
        self.PWC1 = Conv(in_channels, in_channels // 2, 1)
        self.SPU = SPU(in_channels // 2, out_channels)

        assert filter in (
            "FrFT",
            "FrGT",
        ), "The filter type must be either Fractional Fourier Transform(FrFT) or Fractional Gabor Transform(FrGT)."
        if filter == "FrFT":
            self.FPU = FourierFPU(in_channels // 2, out_channels, order)
        elif filter == "FrGT":
            self.FPU = GaborFPU(in_channels // 2, out_channels, order)
        self.c3 = Conv(out_channels, out_channels, 1)

        assert attention in (
            "sa",
            "ca",
        ), "The attention type must be either spatial attention (sa) or channel attention (ca)."
        self.attention = attention
        # Spatial Attention
        if self.attention == "sa":
            self.sa = SpatialAttention(kernel_size=3)
            self.fc1 = Conv(1, 1, 1, act=nn.ReLU())
            self.fc2 = Conv(1, 1, 1, act=nn.ReLU())
        # Channel Attention
        elif self.attention == "ca":
            self.ca = ChannelAttention()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x1 = self.PWC0(x)
        x2 = self.PWC1(x)
        feature1 = self.SPU(x1)
        feature2 = self.FPU(x2)
        # Spatial Attention
        if self.attention == "sa":
            feature_U = feature1 + feature2
            feature_sa = self.sa(feature_U)
            t = torch.stack((self.fc1(feature_sa), self.fc2(feature_sa)))
        # Channel Attention
        elif self.attention == "ca":
            t = torch.stack((self.ca(feature1), self.ca(feature2)))
        a, b = self.softmax(t)
        feature_out = a * feature1 + b * feature2
        feature_out = self.c3(feature_out)
        if feature_out.shape[1] == x.shape[1]:
            feature_out = feature_out + x
        return feature_out


class FT_Conv_Concat(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25, filter="FrFT", attention="ca"):
        super().__init__()

        # 使用您现有的 FT_Conv 模块
        self.ft_conv = FrGTConv(in_channels, out_channels, order=0.25, filter="FrFT", attention="ca")

        # 串联的卷积层
        self.conv2 = Conv(out_channels, out_channels, k=1)  # 此处参数 (out_channels, out_channels, 3) 仅作示例

    def forward(self, x):
        out_ft = self.ft_conv(x)  # FT_Conv 模块的输出
        out = self.conv2(out_ft)  # 串联的卷积层的输出
        return out
class FTP(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25, filter="FrFT", attention="ca"):
        super().__init__()

        # 使用您现有的 FT_Conv 模块
        self.ft_conv = FrGTConv(in_channels, out_channels, order=0.25, filter="FrFT", attention="ca")
        self.conv1 = PConv(in_channels, out_channels, 1)
        # 串联的卷积层
        self.conv2 = Conv(out_channels*2, out_channels, k=1)  # 此处参数 (out_channels, out_channels, 3) 仅作示例

    def forward(self, x):
        out_ft = self.ft_conv(x)  # FT_Conv 模块的输出
        out1 = self.conv1(x)
        # 串联的卷积层的输出torch.cat((x4, x3)
        return self.conv2(torch.cat((out_ft,out1), 1))
class FTP1(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25, filter="FrFT", attention="ca"):
        super().__init__()

        # 使用您现有的 FT_Conv 模块
        self.ft_conv = FrGTConv(in_channels, out_channels, order=0.25, filter="FrFT", attention="ca")
        self.conv1 = PConv(in_channels, out_channels, 1)
        # 串联的卷积层
        # self.conv2 = Conv(out_channels*2, out_channels, k=1)  # 此处参数 (out_channels, out_channels, 3) 仅作示例

    def forward(self, x):
        out_ft = self.ft_conv(x)  # FT_Conv 模块的输出
        out1 = self.conv1(x)
        # 串联的卷积层的输出torch.cat((x4, x3)
        return torch.cat((out_ft,out1), 1)

class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x
class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret
class CoordFrGTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
             in_channels += 1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)
        self.FT_Conv = FrGTConv(out_channels, out_channels, order=64, filter="FrFT", attention="ca")
        self.FT_Conv1 = FrGTConv(out_channels, out_channels, order=64, filter="FrGT", attention="ca")
        self.conv1 = Conv(out_channels, out_channels, k=1, s=stride)
        self.conv2 = Conv(out_channels, out_channels, k=3)
    def forward(self, x):
        x1 = self.addcoords(x)
        x2 = self.conv(x1)
        #
        x3 = self.FT_Conv(x2)
        x4 = self.conv1(x3)

        x5 = self.FT_Conv1(x2)
        x6 = self.conv1(x5)

        # x8 = self.conv2(x2)

        x7 = torch.cat([x6 + x4 + x2], dim=1)

        return x7





