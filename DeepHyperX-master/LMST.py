import math
import torch
import torch.nn as nn

class lmst(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, num_stages=3,
                n_groups=[2, 2, 2], embed_dims=[256, 128, 64], num_heads=[8, 8, 2], mlp_ratios=[1, 1, 1], depths=[1, 1, 1],drop_path_rate=0.1):
        super().__init__()

        self.num_stages = num_stages
        
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]

        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))

        for i in range(num_stages):

            patch_embed = LocalPixelEmbedding(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )

            multi_scal2 = Multi_Scale2(i, embed_dim=embed_dims[i],)  #proposed
            # multi_scal3 = Multi_Scale3(i, embed_dim=embed_dims[i],)  #For Ablation Study

            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop=0.,
                attn_drop=0.) for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)

            setattr(self, f"multi_scal2{i + 1}", multi_scal2)
            # setattr(self, f"multi_scal3{i + 1}", multi_scal3)


            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        
        self.head = nn.Linear(embed_dims[-2],num_classes)  #if num_stages=3,  embed_dims[-1]


    def forward_features(self, x):
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            multi_scal2 = getattr(self, f"multi_scal2{i + 1}")
            # multi_scal3 = getattr(self, f"multi_scal3{i + 1}")

            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")


            x, s = patch_embed(x)  # s = feature map size after patch embedding
            x = multi_scal2(x)
            # x = multi_scal3(x)


            for blk in block:

                x = blk(x)

            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension hidden_features and out_features
        output: (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = H * W, C = dimension for attention
        output: (B, N, C)
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LocalPixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        after_feature_map_size = self.ifm_size

        return x, after_feature_map_size


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Multi_Scale2(nn.Module):
    def __init__(self,  i,embed_dim,):
        super().__init__()
        self.i = i
        self.Multi_Scale2_1 =Multi_Scale2_1(embed_dim,)#
        self.Multi_Scale2_2 =Multi_Scale2_2(embed_dim,)
        self.ms = [self.Multi_Scale2_1,self.Multi_Scale2_2]

    def forward(self, x):
        """
         input: (B, C, H, W)
         output: (B, (H x W), C)
        """
        x = self.ms[self.i](x)
        x = x.flatten(2).transpose(1, 2)

        return x



class Eca_layer(nn.Module):
    """Constructs a ECA module.*
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self,):
        super(Eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel = x.shape[1]
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  #y  64,40,1,1

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)
        values, indices = torch.topk(y, channel, dim=1, largest=True, sorted=True)
        b, c, h, w = x.shape
        # output = x * y.expand_as(x)
        # output = torch.sum(output, dim=1).unsqueeze(1)
        out = []
        for i in range(b):
            m = x[i, :, :, :]
            j = indices[i]
            j = torch.squeeze(j)
            t = m.index_select(0, j)
            t = torch.unsqueeze(t, 0)
            out.append(t)
        out = torch.cat(out, dim=0)
        # z = torch.cat([output, out], dim=1)
        return out



class PConv2_1(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=2, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_conv], dim=1)
        x1 = self.conv(x1)
        x = torch.concat([x1, x2], dim=1)

        return x

class PConv2_2(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=2, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_conv], dim=1)
        x2 = self.conv(x2)
        x = torch.concat([x1, x2], dim=1)

        return x

class Multi_Scale2_1(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0 ,dropout=0.1):
        super().__init__()
        self.pconv1 = PConv2_1(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.eca_layer = Eca_layer()
        self.do1 = nn.Dropout(dropout)

    def forward_ESA1(self, x):#ESA in the stage1
        x = self.pconv1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.forward_ESA1(x)
        x = self.do1(x) + residual
        return x

class Multi_Scale2_2(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0 , dropout=0.1):
        super().__init__()
        self.pconv2 = PConv2_2(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.eca_layer = Eca_layer()
        self.do1 = nn.Dropout(dropout)

    def forward_ESA2(self, x):  #ESA in the stage2
        x = self.pconv2(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.forward_ESA2(x)
        x = self.do1(x) + residual
        return x


class Multi_Scale3(nn.Module):
    def __init__(self, i, embed_dim, ):
        super().__init__()

        self.i = i
        self.Multi_Scale3_1 = Multi_Scale3_1(embed_dim, )  #
        self.Multi_Scale3_2 = Multi_Scale3_2(embed_dim, )
        self.Multi_Scale3_3 = Multi_Scale3_3(embed_dim, )
        self.ms = [self.Multi_Scale3_1, self.Multi_Scale3_2, self.Multi_Scale3_3]

    def forward(self, x):
        x = self.ms[self.i](x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PConv3_1(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_last = dim - self.dim_conv * 2
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_last], dim=1)
        x1 = self.conv(x1)
        x = torch.concat([x1, x2, x3], dim=1)

        return x


class PConv3_2(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_last = dim - self.dim_conv * 2

        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_last], dim=1)
        x2 = self.conv(x2)
        x = torch.concat([x1, x2, x3], dim=1)

        return x


class PConv3_3(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_last = dim - self.dim_conv * 2

        self.conv = nn.Conv2d(self.dim_last, self.dim_last, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_last], dim=1)
        x3 = self.conv(x3)
        x = torch.concat([x1, x2, x3], dim=1)
        return x


class Multi_Scale3_1(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, dropout=0.1, ):
        super().__init__()
        self.pconv1 = PConv3_1(dim)
        # print(".....dim {}".format(dim))
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, stride=1, bias=False)
        self.eca_layer = Eca_layer()
        self.do1 = nn.Dropout(dropout)

    def forward_ESA3_1(self, x):
        x = self.pconv1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.forward_ESA3_1(x)
        x = self.do1(x) + residual
        return x


class Multi_Scale3_2(nn.Module):
    def __init__(self, dim=80, expand_ratio=2, act_layer=nn.ReLU, dropout=0.1, ):
        super().__init__()
        self.pconv2 = PConv3_2(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.eca_layer = Eca_layer()
        self.do1 = nn.Dropout(dropout)

    def forward_ESA3_2(self, x):
        x = self.pconv2(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.forward_ESA3_2(x)
        x = self.do1(x) + residual
        return x


class Multi_Scale3_3(nn.Module):
    def __init__(self, dim=80, expand_ratio=2, act_layer=nn.ReLU, dropout=0.1, ):
        super().__init__()
        self.pconv3 = PConv3_3(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.eca_layer = Eca_layer()
        self.do1 = nn.Dropout(dropout)

    def forward_ESA3_3(self, x):
        x = self.pconv3(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.forward_ESA3_3(x)
        x = self.do1(x) + residual
        return x
