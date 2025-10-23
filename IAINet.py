import torch.nn as nn
from IAIBlock import IAI_block


class IAINet(nn.Module):

    def __init__(self):
        super(IAINet, self).__init__()

        self.inv1 = IAI_block()
        self.inv2 = IAI_block()
        self.inv3 = IAI_block()
        self.inv4 = IAI_block()
        self.inv5 = IAI_block()
        self.inv6 = IAI_block()
        self.inv7 = IAI_block()
        self.inv8 = IAI_block()

        self.inv9 = IAI_block()
        self.inv10 = IAI_block()
        self.inv11 = IAI_block()
        self.inv12 = IAI_block()
        self.inv13 = IAI_block()
        self.inv14 = IAI_block()
        self.inv15 = IAI_block()
        self.inv16 = IAI_block()

    def forward(self, x, rev=False):
        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

            out = self.inv9(out)
            out = self.inv10(out)
            out = self.inv11(out)
            out = self.inv12(out)
            out = self.inv13(out)
            out = self.inv14(out)
            out = self.inv15(out)
            out = self.inv16(out)
        else:
            out = self.inv16(x, rev=True)
            out = self.inv15(out, rev=True)
            out = self.inv14(out, rev=True)
            out = self.inv13(out, rev=True)
            out = self.inv12(out, rev=True)
            out = self.inv11(out, rev=True)
            out = self.inv10(out, rev=True)
            out = self.inv9(out, rev=True)
            
            out = self.inv8(out, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out


