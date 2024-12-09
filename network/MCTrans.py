from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch
import torch.nn.functional as F
import ml_collections
from einops import rearrange
import numbers
from thop import profile
from utils import *

class Channel_Embeddings(nn.Module):
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        patch_size = _pair(patchsize)


        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = self.dropout(x)
        return x


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        x = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# channel-wise Trans
class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer.num_heads
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)

        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.mhead1 = nn.Conv2d(channel_num[0], channel_num[0] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead2 = nn.Conv2d(channel_num[1], channel_num[1] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead3 = nn.Conv2d(channel_num[2], channel_num[2] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mhead4 = nn.Conv2d(channel_num[3], channel_num[3] * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadk = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)
        self.mheadv = nn.Conv2d(self.KV_size, self.KV_size * self.num_attention_heads, kernel_size=1, bias=False)

        self.q1 = nn.Conv2d(channel_num[0] * self.num_attention_heads, channel_num[0] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[0] * self.num_attention_heads // 2, bias=False)
        self.q2 = nn.Conv2d(channel_num[1] * self.num_attention_heads, channel_num[1] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[1] * self.num_attention_heads // 2, bias=False)
        self.q3 = nn.Conv2d(channel_num[2] * self.num_attention_heads, channel_num[2] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[2] * self.num_attention_heads // 2, bias=False)
        self.q4 = nn.Conv2d(channel_num[3] * self.num_attention_heads, channel_num[3] * self.num_attention_heads, kernel_size=3, stride=1,
                            padding=1,
                            groups=channel_num[3] * self.num_attention_heads // 2, bias=False)
        self.k = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)
        self.v = nn.Conv2d(self.KV_size * self.num_attention_heads, self.KV_size * self.num_attention_heads, kernel_size=3, stride=1,
                           padding=1, groups=self.KV_size * self.num_attention_heads, bias=False)

        self.project_out1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
        self.project_out2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
        self.project_out3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
        self.project_out4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb2, emb3, emb4, emb_all, cross_flag = False):
        b, c, h, w = emb1.shape
        q1 = self.q1(self.mhead1(emb1))
        q2 = self.q2(self.mhead2(emb2))
        q3 = self.q3(self.mhead3(emb3))
        q4 = self.q4(self.mhead4(emb4))
        k = self.k(self.mheadk(emb_all))
        v = self.v(self.mheadv(emb_all))
        # k, v = kv.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q3 = rearrange(q3, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        q4 = rearrange(q4, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_attention_heads)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q3 = torch.nn.functional.normalize(q3, dim=-1)
        q4 = torch.nn.functional.normalize(q4, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, c1, _ = q1.shape
        _, _, c2, _ = q2.shape
        _, _, c3, _ = q3.shape
        _, _, c4, _ = q4.shape
        _, _, c, _ = k.shape

        attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)
        attn4 = (q4 @ k.transpose(-2, -1)) / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attn1))
        attention_probs2 = self.softmax(self.psi(attn2))
        attention_probs3 = self.softmax(self.psi(attn3))
        attention_probs4 = self.softmax(self.psi(attn4))

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None

        out1 = (attention_probs1 @ v)
        out2 = (attention_probs2 @ v)
        out3 = (attention_probs3 @ v)
        out4 = (attention_probs4 @ v)

        out_1 = out1.mean(dim=1)
        out_2 = out2.mean(dim=1)
        out_3 = out3.mean(dim=1)
        out_4 = out4.mean(dim=1)

        out_1 = rearrange(out_1, 'b  c (h w) -> b c h w', h=h, w=w)
        out_2 = rearrange(out_2, 'b  c (h w) -> b c h w', h=h, w=w)
        out_3 = rearrange(out_3, 'b  c (h w) -> b c h w', h=h, w=w)
        out_4 = rearrange(out_4, 'b  c (h w) -> b c h w', h=h, w=w)

        O1 = self.project_out1(out_1)
        O2 = self.project_out2(out_2)
        O3 = self.project_out3(out_3)
        O4 = self.project_out4(out_4)

        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None

        weights = None

        return O1, O2, O3, O4, weights


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class eca_layer_2d(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer_2d, self).__init__()
        padding = k_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_features * 2), nn.GELU())

        # Depthwise convolution for local feature enhancement
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, groups=hidden_features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dwconv5x5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, groups=hidden_features, kernel_size=5, stride=1, padding=2),
            nn.GELU()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_features, dim))

        self.dim_conv = dim // 4
        self.dim_untouched = dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=bias)
        self.eca = eca_layer_2d(dim)

    def forward(self, x):
        bs, c, h, w = x.size()

        # Feature refinement with partial convolution
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)


        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.linear1(x)
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, 'b (h w) (c) -> b c h w', h=h, w=w)
        x_1 = self.dwconv3x3(x_1)
        x_2 = rearrange(x_2, 'b (h w) (c) -> b c h w', h=h, w=w)
        x_2 = self.dwconv5x5(x_2)

        x = x_1 * x_2

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.linear2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.eca(x)
        return x



# CCTB
class Block_CC(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_CC, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.cc_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)

    def forward(self, vi_ct, ir_ct):
        embcat_vi = []
        org_v1, org_v2, org_v3, org_v4 = vi_ct
        embv1, embv2, embv3, embv4 = vi_ct
        for i in range(4):
            var_name = "embv" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat_vi.append(tmp_var)
        emb_all_vi = torch.cat(embcat_vi, dim=1)

        embcat_ir = []
        org_r1, org_r2, org_r3, org_r4 = ir_ct
        embr1, embr2, embr3, embr4 = ir_ct
        for i in range(4):
            var_name = "embr" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat_ir.append(tmp_var)
        emb_all_ir = torch.cat(embcat_ir, dim=1)

        cv1 = self.attn_norm1(embv1) if embv1 is not None else None
        cv2 = self.attn_norm2(embv2) if embv2 is not None else None
        cv3 = self.attn_norm3(embv3) if embv3 is not None else None
        cv4 = self.attn_norm4(embv4) if embv4 is not None else None
        emb_all_vi = self.attn_norm(emb_all_vi)

        cr1 = self.attn_norm1(embr1) if embr1 is not None else None
        cr2 = self.attn_norm2(embr2) if embr2 is not None else None
        cr3 = self.attn_norm3(embr3) if embr3 is not None else None
        cr4 = self.attn_norm4(embr4) if embr4 is not None else None
        emb_all_ir = self.attn_norm(emb_all_ir)

        cross_flag = True
        cv1, cv2, cv3, cv4, weight = self.cc_attn(cv1, cv2, cv3, cv4, emb_all_ir, cross_flag)
        cr1, cr2, cr3, cr4, _ = self.cc_attn(cr1, cr2, cr3, cr4, emb_all_vi, cross_flag)

        cv1 = org_v1 + cv1
        cv2 = org_v2 + cv2
        cv3 = org_v3 + cv3
        cv4 = org_v4 + cv4
        org1 = cv1
        org2 = cv2
        org3 = cv3
        org4 = cv4
        vx1 = self.ffn_norm1(cv1) if cv1 is not None else None
        vx2 = self.ffn_norm2(cv2) if cv2 is not None else None
        vx3 = self.ffn_norm3(cv3) if cv3 is not None else None
        vx4 = self.ffn_norm4(cv4) if cv4 is not None else None
        vx1 = self.ffn1(vx1) if vx1 is not None else None
        vx2 = self.ffn2(vx2) if vx2 is not None else None
        vx3 = self.ffn3(vx3) if vx3 is not None else None
        vx4 = self.ffn4(vx4) if vx4 is not None else None
        vx1 = vx1 + org1 if vx1 is not None else None
        vx2 = vx2 + org2 if vx2 is not None else None
        vx3 = vx3 + org3 if vx3 is not None else None
        vx4 = vx4 + org4 if vx4 is not None else None

        cr1 = org_r1 + cr1
        cr2 = org_r2 + cr2
        cr3 = org_r3 + cr3
        cr4 = org_r4 + cr4
        org1 = cr1
        org2 = cr2
        org3 = cr3
        org4 = cr4
        rx1 = self.ffn_norm1(cr1) if cr1 is not None else None
        rx2 = self.ffn_norm2(cr2) if cr2 is not None else None
        rx3 = self.ffn_norm3(cr3) if cr3 is not None else None
        rx4 = self.ffn_norm4(cr4) if cr4 is not None else None
        rx1 = self.ffn1(rx1) if rx1 is not None else None
        rx2 = self.ffn2(rx2) if rx2 is not None else None
        rx3 = self.ffn3(rx3) if rx3 is not None else None
        rx4 = self.ffn4(rx4) if rx4 is not None else None
        rx1 = rx1 + org1
        rx2 = rx2 + org2
        rx3 = rx3 + org3
        rx4 = rx4 + org4

        # element-wise add
        f1 = vx1 + rx1
        f2 = vx2 + rx2
        f3 = vx3 + rx3
        f4 = vx4 + rx4

        return f1, f2, f3, f4, weight



#  ICTB
class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(config.KV_size, LayerNorm_type='WithBias')

        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)


    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        emb_all = torch.cat(embcat, dim=1)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)  # 1 480 16 16
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None



        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.cclayer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.encoder_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.encoder_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.encoder_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))
        for _ in range(config.transformer["num_cc_layers"]):
            layer = Block_CC(config, vis, channel_num)
            self.cclayer.append(copy.deepcopy(layer))

    def forward(self, vi_emb, ir_emb):
        attn_weights = []
        emb1, emb2, emb3, emb4 = vi_emb
        emr1, emr2, emr3, emr4 = ir_emb
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            emr1, emr2, emr3, emr4, _ = layer_block(emr1, emr2, emr3, emr4)
            if self.vis:
                attn_weights.append(weights)
        vi_ct = (emb1, emb2, emb3, emb4)
        ir_ct = (emr1, emr2, emr3, emr4)
        for cc_block in self.cclayer:
            emf1, emf2, emf3, emf4, _ = cc_block(vi_ct, ir_ct)

        emf1 = self.encoder_norm1(emf1) if emf1 is not None else None
        emf2 = self.encoder_norm2(emf2) if emf2 is not None else None
        emf3 = self.encoder_norm3(emf3) if emf3 is not None else None
        emf4 = self.encoder_norm4(emf4) if emf4 is not None else None
        return emf1,emf2,emf3,emf4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[32, 64, 128, 256], patchSize=[8, 4, 2, 1], train_flag = True):
        super().__init__()

        self.train_flag = train_flag
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config, self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config, self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1, scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1, scale_factor=(self.patchSize_3, self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1, scale_factor=(self.patchSize_4, self.patchSize_4))

        self.shape_adjust = UpsampleReshape_eval()
    def forward(self, vi, ir):
        emb1 = self.embeddings_1(vi[0])
        emb2 = self.embeddings_2(vi[1])
        emb3 = self.embeddings_3(vi[2])
        emb4 = self.embeddings_4(vi[3])

        emr1 = self.embeddings_1(ir[0])
        emr2 = self.embeddings_2(ir[1])
        emr3 = self.embeddings_3(ir[2])
        emr4 = self.embeddings_4(ir[3])

        vi_emb = (emb1, emb2, emb3, emb4)
        ir_emb = (emr1, emr2, emr3, emr4)
        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(vi_emb, ir_emb)

        x1 = self.reconstruct_1(encoded1)
        x2 = self.reconstruct_2(encoded2)
        x3 = self.reconstruct_3(encoded3)
        x4 = self.reconstruct_4(encoded4)

        if not self.train_flag:
            x1 = self.shape_adjust(vi[0], x1)
            x2 = self.shape_adjust(vi[1], x2)
            x3 = self.shape_adjust(vi[2], x3)
            x4 = self.shape_adjust(vi[3], x4)

        w = [0.5, 0.5]
        x1 = x1 + w[0] * vi[0] + w[1] * ir[0]
        x2 = x2 + w[0] * vi[1] + w[1] * ir[1]
        x3 = x3 + w[0] * vi[2] + w[1] * ir[2]
        x4 = x4 + w[0] * vi[3] + w[1] * ir[3]

        return x1, x2, x3, x4, attn_weights




if __name__ == '__main__':
    config_vit = get_CTranS_config()
    x1 = torch.rand(1, 32, 128, 128)
    x2 = torch.rand(1, 64, 64, 64)
    x3 = torch.rand(1, 128, 32, 32)
    x4 = torch.rand(1, 256, 16, 16)



