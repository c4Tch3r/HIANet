import torch
import torch.nn as nn
import audio_config as ac
from att_denselayer import ResidualDenseBlock_out


class AAI_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=ac.clamp, harr=True):
        super().__init__()
        if harr:
            self.split_len1 = 4
            self.split_len2 = 4
        self.clamp = clamp
        # ψ
        self.p = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len1, self.split_len2)
        # ρ
        self.r = subnet_constructor(self.split_len2, self.split_len1)
        # η
        self.y = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        
        if not rev:
            s1, t1 = self.p(x1), self.f(x1)
            y2 = self.e(s1) * x2 + t1
            s2, t2 = self.r(y2), self.y(y2)
            y1 = self.e(s2) * x1 + t2
        else:
            s2, t2 = self.r(x2), self.y(x2)
            y1 = (x1 - t2) / self.e(s2)
            s1, t1 = self.p(y1), self.f(y1)
            y2 = (x2 - t1) / self.e(s1)


        return torch.cat((y1, y2), 1)

