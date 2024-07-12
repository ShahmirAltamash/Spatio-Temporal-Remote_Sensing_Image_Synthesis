
import torch
import config
import utils
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import matplotlib as mpl
from pathlib import Path
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# %matplotlib inline

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
        if down
        else nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2)
    )
    self.use_dropout = use_dropout
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    x = self.conv(x)
    return self.dropout(x) if self.use_dropout else x

class UNet(nn.Module):
  def __init__(self, in_channels=3, features=64):
    super().__init__()
    self.initial_down = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
        nn.LeakyReLU(0.2)
    ) # 128

    # Down
    self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False)   # 64
    self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=False) # 32
    self.down3 = Block(features*4, features*8, down=True, act="leaky", use_dropout=False) # 16
    self.down4 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False) # 8
    self.down5 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False) # 4
    self.down6 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False) # 2
    self.bottleneck = nn.Sequential(
        nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),               # 1
        nn.ReLU()
    )

    # Up
    self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=True)   # 2
    self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 4
    self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 8
    self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 16
    self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=True) # 32
    self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=True) # 64
    self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=True)   # 128
    self.final_up = nn.Sequential(
        nn.ConvTranspose2d(features*2, 1, kernel_size=4, stride=2, padding=1), # 256
        nn.Sigmoid()
    )
  
  def forward(self, x):
    d1 = self.initial_down(x)
    d2 = self.down1(d1)
    d3 = self.down2(d2)
    d4 = self.down3(d3)
    d5 = self.down4(d4)
    d6 = self.down5(d5)
    d7 = self.down6(d6)
    bottleneck = self.bottleneck(d7)
    up1 = self.up1(bottleneck)
    up2 = self.up2(torch.cat([up1, d7], dim=1))
    up3 = self.up3(torch.cat([up2, d6], dim=1))
    up4 = self.up4(torch.cat([up3, d5], dim=1))
    up5 = self.up5(torch.cat([up4, d4], dim=1))
    up6 = self.up6(torch.cat([up5, d3], dim=1))
    up7 = self.up7(torch.cat([up6, d2], dim=1))
    return self.final_up(torch.cat([up7, d1], dim=1))


if __name__ == "__main__":
    model = UNet()
    utils.load_model(config.CHECKPOINT_SEG, model)

