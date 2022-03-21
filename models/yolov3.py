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


class YOLOv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_class = 80
        self.anchors = {
            'big': [(116, 90), (156, 198), (373, 326)],
            'mid': [(30, 61), (62, 45), (59, 119)],
            'sml': [(10, 13), (16, 30), (33, 23)],
        }

        self.dbl1 = DBLUnit(3, 32, 3, 1)

        self.res1 = nn.Sequential(
            DBLUnit(32, 64, 3, 2), # Downsample
            ResBlock(64),
        )

        self.res2 = nn.Sequential(
            DBLUnit(64, 128, 3, 2), # Downsample
            ResBlock(128),
            ResBlock(128),
        )

        self.res8_big = nn.Sequential(
            DBLUnit(128, 256, 3, 2), # Downsample
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )

        self.res8_mid = nn.Sequential(
            DBLUnit(256, 512, 3, 2), # Downsample
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
        )

        self.res4_sml = nn.Sequential(
            DBLUnit(512, 1024, 3, 2), # Downsample
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
        )

        self.dbl5_sml = nn.Sequential(
            DBLUnit(1024, 512, 1, 1),
            DBLUnit(512, 1024, 3, 1),
            DBLUnit(1024, 512, 1, 1),
            DBLUnit(512, 1024, 3, 1),
            DBLUnit(1024, 512, 1, 1),
        )

        self.upsample_mid = nn.Sequential(
            DBLUnit(512, 256, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.dbl5_mid = nn.Sequential(
            DBLUnit(512+256, 256, 1, 1),
            DBLUnit(256, 512, 3, 1),
            DBLUnit(512, 256, 1, 1),
            DBLUnit(256, 512, 3, 1),
            DBLUnit(512, 256, 1, 1),
        )

        self.upsample_big = nn.Sequential(
            DBLUnit(256, 128, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.dbl5_big = nn.Sequential(
            DBLUnit(256+128, 128, 1, 1),
            DBLUnit(128, 256, 3, 1),
            DBLUnit(256, 128, 1, 1),
            DBLUnit(128, 256, 3, 1),
            DBLUnit(256, 128, 1, 1),
        )

        self.feat_sml = nn.Sequential(
            DBLUnit(512, 1024, 3, 1),
            nn.Conv2d(1024, 3 * (5 + self.n_class), 1, 1),
        )

        self.feat_mid = nn.Sequential(
            DBLUnit(256, 512, 3, 1),
            nn.Conv2d(512, 3 * (5 + self.n_class), 1, 1),
        )

        self.feat_big = nn.Sequential(
            DBLUnit(128, 256, 3, 1),
            nn.Conv2d(256, 3 * (5 + self.n_class), 1, 1),
        )

    def forward(self, x):
        # common layers
        x = self.dbl1(x)
        x = self.res1(x)
        x = self.res2(x)

        # features with three different scales
        x_big = self.res8_big(x)
        x_mid = self.res8_mid(x_big)
        x_sml = self.res4_sml(x_mid)

        x_sml = self.dbl5_sml(x_sml)

        x_mid = torch.concat([x_mid, self.upsample_mid(x_sml)], dim=1)
        x_mid = self.dbl5_mid(x_mid)

        x_big = torch.concat([x_big, self.upsample_big(x_mid)], dim=1)
        x_big = self.dbl5_big(x_big)

        # get feature maps
        x_sml = self.feat_sml(x_sml)
        x_mid = self.feat_mid(x_mid)
        x_big = self.feat_big(x_big)

        # calculate offsets
        x_sml = calculate_offset(x_sml, x.shape, self.anchors['sml'])
        x_mid = calculate_offset(x_mid, x.shape, self.anchors['mid'])
        x_big = calculate_offset(x_big, x.shape, self.anchors['big'])

        return x_sml, x_mid, x_big


def calculate_offset(x_map, img_shape, anchors):
    # x: [n_batch, 3*(80+5), h, w] -> [n_batch, 3, h, w, 80+5]
    n_batch, _, h_map, w_map = x_map.shape
    x_map = x_map.view(n_batch, 3, -1, h_map, w_map).permute(0, 1, 3, 4, 2).contiguous()

    # Add offset and scale to the original size
    _, _, _, w_img = img_shape
    r = w_img/w_map
    grid_h, grid_w = torch.meshgrid(torch.arange(h_map), torch.arange(w_map), indexing='ij')
    x_map[..., 0] = r * (torch.sigmoid(x_map[..., 0]) + grid_w)
    x_map[..., 1] = r * (torch.sigmoid(x_map[..., 1]) + grid_h)

    # Tune anchors (anchors are already scaled to the original size)
    w_anchor, h_anchor = torch.tensor(anchors).t().view(2, 1, 3, 1, 1)
    x_map[..., 2] = torch.exp(x_map[..., 2]) * w_anchor
    x_map[..., 3] = torch.exp(x_map[..., 3]) * h_anchor

    # object score(4), class probability(5:)
    x_map[..., 4:] = torch.sigmoid(x_map[..., 4:])

    return x_map


if __name__ == "__main__":
    x = torch.randn((10, 3, 416, 416))

    yolov3 = YOLOv3()
    x_sml, x_mid, x_big = yolov3(x)

    print(x_sml.shape)
    print(x_mid.shape)
    print(x_big.shape)
    print(x_sml[0, 0, 0, 0, :4])

