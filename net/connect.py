import torch
import torch.nn as nn
import torch.nn.functional as F


class Corr_Up(nn.Module):
    def __init__(self):
        super(Corr_Up, self).__init__()

        self.loc_adjust = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, z_f, x_f):
        pred_loc = self.loc_adjust(F.conv2d(x_f, z_f))
        return pred_loc


