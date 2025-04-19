from math import exp
import torch
import torch.nn as nn
from denseblock import Dense
import config as c


class INV_block_addition(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()

        if harr:
            self.split_len1 = in_1 * 4  # 因为是小波域所以*4
            self.split_len2 = in_2 * 4
        self.clamp = clamp

        # ρ
        # self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # self.s2 = subnet_constructor(self.split_len2, self.split_len1)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            t1 = self.y(y1)
            y2 = x2 + t1

        else:  # names of x and y are swapped!

            t1 = self.y(x1)
            y2 = (x2 - t1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3, imp_map=True):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4  # 12
            self.split_len2 = in_2 * 4  # 12
        self.clamp = clamp
        if imp_map:
            self.imp = 12
        else:
            self.imp = 0    #  imp_map=False时imp=0

        ### 构建一个IHNN中的invertible hiding module部分，四个稠密函数
        # ρ
        self.r = subnet_constructor(self.split_len1 + self.imp, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1 + self.imp, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1 + self.imp)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1 + self.imp)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = (x.narrow(1, 0, self.split_len1 + self.imp),       # 取出列，从第0列，取self.split_len1 + self.imp+1列
                  x.narrow(1, self.split_len1 + self.imp, self.split_len2))  # 对列进行操作，从self.split_len1 + self.imp列

        if not rev:     # rev为false表示前向求解

            t2 = self.f(x2)
            s2 = self.p(x2)
            y1 = self.e(s2) * x1 + t2   # y1为stego
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1   # y2为信息损失R

        else:  # names of x and y are swapped!  # rev为true表示后向求解

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)  # 还原过程中的sr(secret)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2) # 还原过程中的cr(cover)

        return torch.cat((y1, y2), 1)       # 拼接在一起，dim=1

class INV_block_affine_new(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4  # 12
            self.split_len2 = in_2 * 4  # 12
        self.clamp = clamp
        self.map = 8

        ### 构建一个IHNN中的invertible hiding module部分，四个稠密函数
        # ρ
        self.r = subnet_constructor(self.split_len1 + self.map, self.split_len2)    # input=20, output=12
        # η
        self.y = subnet_constructor(self.split_len1 + self.map, self.split_len2)    # 20, 12
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1 + self.map)    # 12, 20
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1 + self.map)    # 12, 20

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = (x.narrow(1, 0, self.split_len1 + self.map),       # channels = 20
                  x.narrow(1, self.split_len1 + self.map, self.split_len2))  # channels = 12

        if not rev:     # rev为false表示前向求解

            t2 = self.f(x2)      # channels = 20
            s2 = self.p(x2)      # channels = 20
            y1 = self.e(s2) * x1 + t2   # y1为stego
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1   # y2为信息损失R

        else:  # names of x and y are swapped!  # rev为true表示后向求解

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)  # 还原过程中的sr(secret)
            t2 = self.f(y2)
            s2 = self.p(y2)
            y1 = (x1 - t2) / self.e(s2) # 还原过程中的cr(cover)

        return torch.cat((y1, y2), 1)       # 拼接在一起，dim=1