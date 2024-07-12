import torch
import torch.nn as nn


# Discriminator Model
class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride, padding):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=padding, bias=False, padding_mode="reflect"),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.conv(x)

# input is (x, y) concatenated across channels
class Discriminator(nn.Module):
  """
  Input shape  = (n, in_channels, 1024, 1024)
  Output shape = (n, 1, 30, 30)
  """
  def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # 512 -> 30x30
    super().__init__()
    self.initial = nn.Sequential(
        nn.Conv2d(in_channels=in_channels*2, out_channels=features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
        nn.LeakyReLU(0.2)
    )
    layers = []
    in_channels = features[0]
    for feature in features[1:]:
      layers.append(
          CNNBlock(in_channels, feature, stride=2, padding=0 if feature == features[-1] else 1)
      )
      in_channels = feature
    
    layers.append(
        nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
        )
    )

    self.model = nn.Sequential(*layers)

  def forward(self, x, y):
    x = torch.cat([x, y], dim=1)
    x = self.initial(x)
    return self.model(x)


# Generator Model
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

class Generator(nn.Module):
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
        nn.ConvTranspose2d(features*2, 3, kernel_size=4, stride=2, padding=1), # 256
        nn.Tanh()
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
  

# Attention Block
class AttentionBlock(nn.Module):
  def __init__(self,f_g, f_l, f_int):
    super(AttentionBlock, self).__init__()

    self.W_g = nn.Sequential(
      nn.Conv2d(f_g, f_int, kernel_size=1,stride=1,padding=0,bias=True),
      nn.BatchNorm2d(f_int)
    )
    self.W_x = nn.Sequential(
      nn.Conv2d(f_l, f_int, kernel_size=1,stride=1,padding=0,bias=True),
      nn.BatchNorm2d(f_int)
    )
    self.psi = nn.Sequential(
      nn.Conv2d(f_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
      nn.BatchNorm2d(1),
      nn.Sigmoid()
    )

    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, g, x):
    g1 = self.W_g(g)
    x1 = self.W_x(x)
    psi = self.relu(g1 + x1)
    psi = self.psi(psi)
    return x * psi
  
# Att_Generator (input shape = (n, 4, 512, 512) , (output shape = (n, 3, 512, 512) 
class Att_Generator(nn.Module):
  def __init__(self, in_channels=4, features=64):
    super().__init__()
    self.initial_down = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
        nn.LeakyReLU(0.2)
    ) # 128
    
    # Down  
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
    self.att1 = AttentionBlock(features*8, features*8, features*4)
    self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 4
    self.att2 = AttentionBlock(features*8, features*8, features*4)
    self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 8
    self.att3 = AttentionBlock(features*8, features*8, features*4)
    self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) # 16
    self.att4 = AttentionBlock(features*8, features*8, features*4)
    self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=True) # 32
    self.att5 = AttentionBlock(features*4, features*4, features*2)
    self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=True) # 64
    self.att6 = AttentionBlock(features*2, features*2, features)
    self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=True)   # 128
    self.att7 = AttentionBlock(features, features, features//2)
    self.final_up = nn.Sequential(
        nn.ConvTranspose2d(features*2, 3, kernel_size=4, stride=2, padding=1),
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
    att1 = self.att1(g=up1 , x=d7)
    up1 = torch.cat([up1,att1] , dim=1)
    
    up2 = self.up2(up1)
    att2 = self.att2(g=up2 , x=d6)
    up2 = torch.cat([up2,att2] , dim=1)
  
    up3 = self.up3(up2)
    att3 = self.att3(g=up3 , x=d5)
    up3 = torch.cat([up3,att3] , dim=1)

    up4 = self.up4(up3)
    att4 = self.att4(g=up4 , x=d4)
    up4 = torch.cat([up4,att4] , dim=1)

    up5 = self.up5(up4)
    att5 = self.att5(g=up5 , x=d3)
    up5 = torch.cat([up5,att5] , dim=1)

    up6 = self.up6(up5)
    att6 = self.att6(g=up6 , x=d2)
    up6 = torch.cat([up6,att6] , dim=1)

    up7 = self.up7(up6)
    att7 = self.att7(g=up7 , x=d1)
    up7 = torch.cat([up7,att7] , dim=1)

    final = self.final_up(up7)

    return final
  

def test_disc():
  x = torch.randn((1, 3, 512, 512))
  y = torch.randn((1, 3, 512, 512))
  model = Discriminator()
  preds = model(x, y)
  print(preds.shape)

def test_gen():
  x = torch.randn((1, 4, 512, 512))
  model = Generator(in_channels=4, features=64)
  preds = model(x)
  print(preds.shape)

def test_att_gen():
  x = torch.randn((1, 4, 512, 512))
  model = Att_Generator(in_channels=4, features=64)
  preds = model(x)
  print(preds.shape)


if __name__ == "__main__":
  print(f"Testing Discriminator:")
  test_disc()
  print(f"Testing Generator:")
  test_gen()
  print(f"Testing Att_Generator:")
  test_att_gen()