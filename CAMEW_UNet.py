import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn import Softmax



class Conv2dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )


class Conv1dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.GroupNorm(4, out_channel),
            nn.GELU()
        )


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1, norm_type='gn', gn_num=4):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        if norm_type == 'bn':
            self.norm_layer = nn.BatchNorm2d(dim_in)
        elif norm_type == 'in':
            self.norm_layer = nn.InstanceNorm2d(dim_in)
        elif norm_type == 'gn':
            self.norm_layer = nn.GroupNorm(gn_num, dim_in)
        else:
            raise ('Error norm_type')
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class InvertedDepthWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio  # 扩展通道数
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 添加 1x1 点卷积层
        layers.append(Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 添加 3x3 深度卷积层
            Conv2dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 添加 1x1 点卷积层（线性）
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # 添加组归一化层
            nn.GroupNorm(4, out_channel),
        ])

        self.conv = nn.Sequential(*layers)  # 将所有层组合成一个顺序容器

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedDepthWiseConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 1x1 pointwise conv
        layers.append(Conv1dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            Conv1dGNGELU(hidden_channel, hidden_channel, kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MEW(nn.Module):
    def __init__(self, dim, bias=False, a=16, b=16, c_h=16, c_w=16):
        super().__init__()

        # 使用register_buffer注册常数值
        self.register_buffer("dim", torch.as_tensor(dim))
        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))
        self.register_buffer("c_h", torch.as_tensor(c_h))
        self.register_buffer("c_w", torch.as_tensor(c_w))

        # 初始化可学习参数
        self.a_weight = nn.Parameter(torch.Tensor(2, 1, dim // 4, a))
        nn.init.ones_(self.a_weight)
        self.b_weight = nn.Parameter(torch.Tensor(2, 1, dim // 4, b))
        nn.init.ones_(self.b_weight)
        self.c_weight = nn.Parameter(torch.Tensor(2, dim // 4, c_h, c_w))
        nn.init.ones_(self.c_weight)
        self.dw_conv = InvertedDepthWiseConv2d(dim // 4, dim // 4)

        # 权重生成器，使用反向深度卷积（inverted depth-wise convolution）
        self.wg_a = nn.Sequential(
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )

        self.wg_b = nn.Sequential(
            InvertedDepthWiseConv1d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv1d(2 * dim // 4, dim // 4),
        )

        self.wg_c = nn.Sequential(
            InvertedDepthWiseConv2d(dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, 2 * dim // 4),
            InvertedDepthWiseConv2d(2 * dim // 4, dim // 4),
        )

    def forward(self, x):
        # 将输入张量x分成四个部分
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, c, a, b = x1.size()
        # ----- a convlution -----#
        x1 = x1.permute(0, 2, 1, 3)  # 调整维度顺序，B, a, c, b
        x1 = torch.fft.rfft2(x1, dim=(2, 3), norm='ortho')  # 二维实部-虚部FFT
        a_weight = self.a_weight
        a_weight = self.wg_a(F.interpolate(a_weight, size=x1.shape[2:4],  #使用双线性插值方法，对 a_weight 进行大小调整，为了保证 a_weight 的大小与 x1 的特征图大小一致，以便进行后续的加权操作
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        a_weight = torch.view_as_complex(a_weight.contiguous())  # 转换为复数形式
        x1 = x1 * a_weight  # 权重乘以FFT结果
        x1 = torch.fft.irfft2(x1, s=(c, b), dim=(2, 3), norm='ortho').permute(0, 2, 1, 3)  # 逆FFT。s=(c, b) 指定逆变换后的输出形状为 (c, b)，这里 c 是通道数，b 是长度

        # ----- b convlution -----#
        x2 = x2.permute(0, 3, 1, 2)  # 调整维度顺序，B, b, c, a
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 二维实部-虚部FFT
        b_weight = self.b_weight
        b_weight = self.wg_b(F.interpolate(b_weight, size=x2.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        b_weight = torch.view_as_complex(b_weight.contiguous())  # 转换为复数形式
        x2 = x2 * b_weight  # 权重乘以FFT结果
        x2 = torch.fft.irfft2(x2, s=(c, a), dim=(2, 3), norm='ortho').permute(0, 2, 3, 1)  # 逆FFT

        # ----- c convlution -----#
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # 二维实部-虚部FFT
        c_weight = self.c_weight
        c_weight = self.wg_c(F.interpolate(c_weight, size=x3.shape[2:4],
                                           mode='bilinear', align_corners=True)).permute(1, 2, 3, 0)
        c_weight = torch.view_as_complex(c_weight.contiguous())  # 转换为复数形式
        x3 = x3 * c_weight  # 权重乘以FFT结果
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')  # 逆FFT

        # ----- dw convlution -----#
        x4 = self.dw_conv(x4)

        # ----- concat -----#
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.mlp = nn.Sequential(InvertedDepthWiseConv2d(dim, mlp_ratio * dim),
                                 InvertedDepthWiseConv2d(mlp_ratio * dim, dim),
                                 nn.GELU()
                                 )

    def forward(self, x):
        return self.mlp(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(4, dim)
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class MEWB(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MEW(dim)),
                PreNorm(dim, MLP(dim, mlp_ratio))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class CAMEWUNet(nn.Module):
    def __init__(self, in_c=3, dim=[32, 64, 128, 256, 512], depth=[1, 2, 2, 4], mlp_ratio=4):
        super().__init__()

        self.e0 = nn.Sequential(
            DepthWiseConv2d(in_c, dim[0], norm_type='bn'),
        )
        self.e1 = nn.Sequential(
            DepthWiseConv2d(dim[0], dim[1]),
            MEWB(dim[1], depth[0], mlp_ratio)
        )
        self.e2 = nn.Sequential(
            DepthWiseConv2d(dim[1], dim[2]),
            MEWB(dim[2], depth[1], mlp_ratio)
        )
        self.e3 = nn.Sequential(
            DepthWiseConv2d(dim[2], dim[3]),
            MEWB(dim[3], depth[2], mlp_ratio)
        )
        self.e4 = nn.Sequential(
            DepthWiseConv2d(dim[3], dim[4]),
            MEWB(dim[4], depth[3], mlp_ratio)
        )

        self.d4 = nn.Sequential(
            MEWB(dim[4], depth[3], mlp_ratio),
            DepthWiseConv2d(dim[4], dim[3])
        )
        self.Att4 = Attention_block(F_g=dim[3], F_l=dim[3], F_int=dim[2])
        self.d3 = nn.Sequential(
            MEWB(dim[4], depth[2], mlp_ratio),
            DepthWiseConv2d(dim[4], dim[2])
        )
        self.Att3 = Attention_block(F_g=dim[2], F_l=dim[2], F_int=dim[1])
        self.d2 = nn.Sequential(
            MEWB(dim[3], depth[1], mlp_ratio),
            DepthWiseConv2d(dim[3], dim[1])
        )
        self.Att2 = Attention_block(F_g=dim[1], F_l=dim[1], F_int=dim[0])
        self.d1 = nn.Sequential(
            MEWB(dim[2], depth[0], mlp_ratio),
            DepthWiseConv2d(dim[2], dim[0])
        )
        self.Att1 = Attention_block(F_g=dim[0], F_l=dim[0], F_int=16)
        self.d0 = nn.Sequential(
            nn.Conv2d(dim[1], dim[0], 1)
        )
        self.conv1 = nn.Conv2d(dim[0], 16, 3, 1, 1)  # 添加
        self.cca1 = CrissCrossAttention(16)
        self.cca2 = CrissCrossAttention(16)
        self.conv2 = nn.Conv2d(16, 1, 3, 1, 1)  # 添加


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # ------encoder------#
        x0 = F.max_pool2d(self.e0(x), 2, 2)  # b, c0, h/2, w/2
        # print(x0.shape)  #torch.Size([1, 32, 128, 128])
        x1 = F.max_pool2d(self.e1(x0), 2, 2)  # b, c1, h/4, w/4
        # print(x1.shape)  #torch.Size([1, 64, 64, 64])
        x2 = F.max_pool2d(self.e2(x1), 2, 2)  # b, c2, h/8, w/8
        # print(x2.shape)  #torch.Size([1, 128, 32, 32])
        x3 = F.max_pool2d(self.e3(x2), 2, 2)  # b, c3, h/16, w/16
        # print(x3.shape)  #torch.Size([1, 256, 16, 16])
        x4 = F.max_pool2d(self.e4(x3), 2, 2)  # b, c4, h/32, w/32
        # print(x4.shape)  #torch.Size([1, 512, 8, 8])
        # ------decoder------#
        out4 = F.interpolate(self.d4(x4), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c3, h/16, w/16
        # print(self.d4(x4).shape)  #torch.Size([1, 256, 8, 8])
        # print(out4.shape)  #torch.Size([1, 256, 16, 16])
        out4 = torch.cat((self.Att4(out4, x3), out4), dim=1)
        # print(out4.shape)  #torch.Size([1, 512, 16, 16])
        out3 = F.interpolate(self.d3(out4), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c2, h/8, w/8
        out3 = torch.cat((self.Att3(out3, x2), out3), dim=1)
        out2 = F.interpolate(self.d2(out3), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c1, h/4, w/4
        out2 = torch.cat((self.Att2(out2, x1), out2), dim=1)
        out1 = F.interpolate(self.d1(out2), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, c0, h/2, w/2
        out1 = torch.cat((self.Att1(out1, x0), out1), dim=1)
        out0 = F.interpolate(self.d0(out1), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, out_c, h, w
        # print(out0.shape)  #torch.Size([1, 32, 256, 256])

        out = self.conv1(out0)
        out = self.cca1(out)
        # print(out.shape)  #torch.Size([1, 16, 256, 256])
        out = self.cca2(out)
        out = self.conv2(out)

        return out


