import torch
import torch.nn as nn


class DBLUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layers = nn.Sequential(
            DBLUnit(in_channels, in_channels // 2, 1, 1),
            DBLUnit(in_channels // 2, in_channels, 3, 1)
        )
    
    def forward(self, x):
        x = self.layers(x) + x
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = DBLUnit(in_channels, out_channels, 3, 2)
    
    def forward(self, x):
        x = self.layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.UpsamplingNearest2d(scale_factor=2)
    
    def forward(self, x):
        x = self.layer(x)
        return x


class YOLOv3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 25~31
        self.dbl1 = DBLUnit(3, 32, 3, 1)

        # 33~61
        self.res1 = nn.Sequential(
            DownSample(32, 64),
            ResBlock(64),
        )

        # 63~91
        self.res2 = nn.Sequential(
            DownSample(64, 128),
            ResBlock(128),
            ResBlock(128),
        )

        # 113~141
        self.res8_big = nn.Sequential(
            DownSample(128, 256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )

        # 286~312
        self.res8_mid = nn.Sequential(
            DownSample(256, 512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
        )

        # 461~487
        self.res4_sml = nn.Sequential(
            DownSample(512, 1024),
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),

        )

        # 551~590
        self.dbl5_sml = nn.Sequential(
            DBLUnit(1024, 512, 1, 1),
            DBLUnit(512, 1024, 3, 1),
            DBLUnit(1024, 512, 1, 1),
            DBLUnit(512, 1024, 3, 1),
            DBLUnit(1024, 512, 1, 1),
        )

        # 592~605
        self.feat_sml = nn.Sequential(
            DBLUnit(512, 1024, 3, 1),
            nn.Conv2d(1024, 255, 1, 1),
        )

        # 622~631
        self.upsample_mid = nn.Sequential(
            DBLUnit(512, 256, 1, 1),
            Upsample()
        )

        # 638~676
        self.dbl5_mid = nn.Sequential(
            DBLUnit(512+256, 256, 1, 1),
            DBLUnit(256, 512, 3, 1),
            DBLUnit(512, 256, 1, 1),
            DBLUnit(256, 512, 3, 1),
            DBLUnit(512, 256, 1, 1),
        )

        # 678~691
        self.feat_mid = nn.Sequential(
            DBLUnit(256, 512, 3, 1),
            nn.Conv2d(512, 255, 1, 1),
        )

        # 709~718
        self.upsample_big = nn.Sequential(
            DBLUnit(256, 128, 1, 1),
            Upsample()
        )

        # 725~763
        self.dbl5_big = nn.Sequential(
            DBLUnit(256+128, 128, 1, 1),
            DBLUnit(128, 256, 3, 1),
            DBLUnit(256, 128, 1, 1),
            DBLUnit(128, 256, 3, 1),
            DBLUnit(256, 128, 1, 1),
        )

        # 765~778
        self.feat_big = nn.Sequential(
            DBLUnit(128, 256, 3, 1),
            nn.Conv2d(256, 255, 1, 1),
        )

    
    def forward(self, x):
        x = self.dbl1(x)
        x = self.res1(x)
        x = self.res2(x)

        x_big = self.res8_big(x)
        x_mid = self.res8_mid(x_big)

        # get small first
        x_sml = self.res4_sml(x_mid)
        x_sml = self.dbl5_sml(x_sml)

        # get mid second
        x_mid = torch.concat([x_mid, self.upsample_mid(x_sml)], dim=1)
        x_mid = self.dbl5_mid(x_mid)

        # get last last
        x_big = torch.concat([x_big, self.upsample_big(x_mid)], dim=1)
        x_big = self.dbl5_big(x_big)

        x_sml = self.feat_sml(x_sml)
        x_mid = self.feat_mid(x_mid)
        x_big = self.feat_big(x_big)
        
        return x_sml, x_mid, x_big


if __name__ == "__main__":
    import torch
    x = torch.randn((10, 3, 64, 64))

    yolov3 = YOLOv3()
    x_sml, x_mid, x_big = yolov3(x)

    print(x_sml.shape)
    print(x_mid.shape)
    print(x_big.shape)

