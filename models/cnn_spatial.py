"""
CNN Spatial Encoder for topographic EEG images

Accepts input tensor of shape (batch, seq_len, channels, H, W)
and outputs per-time embeddings (batch, seq_len, out_dim).
"""
import torch
import torch.nn as nn


class CNNSpatialEncoder(nn.Module):
    def __init__(self, in_channels=1, out_dim=256, image_size=64, channels=(32, 64, 128), kernel_size=3, dropout=0.1):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.image_size = image_size

        layers = []
        cur_in = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(cur_in, out_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(p=dropout)
            ])
            cur_in = out_ch

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # projection to out_dim
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], out_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, config):
        params = getattr(config, 'CNN_PARAMS', None) or {}
        channels = params.get('channels', [32, 64, 128])
        kernel_size = params.get('kernel_size', 3)
        dropout = params.get('dropout', 0.1)
        image_size = getattr(config, 'TOPO_IMAGE_SIZE', 64)
        out_dim = getattr(config, 'CNN_OUT_DIM', None) or config.D_MODEL
        return cls(in_channels=1, out_dim=out_dim, image_size=image_size, channels=channels, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, SEQ, C, H, W)
        returns: (B, SEQ, out_dim)
        """
        B, SEQ, C, H, W = x.shape
        # reshape to process frames together
        x = x.reshape(B * SEQ, C, H, W)
        x = self.cnn(x)
        x = self.global_pool(x)  # (B*SEQ, channels[-1], 1, 1)
        x = x.view(B * SEQ, -1)
        x = self.projection(x)  # (B*SEQ, out_dim)
        x = x.view(B, SEQ, self.out_dim)
        return x


def _test():
    import torch
    enc = CNNSpatialEncoder(in_channels=1, out_dim=256, image_size=64)
    x = torch.randn(2, 16, 1, 64, 64)
    out = enc(x)
    print(out.shape)


if __name__ == '__main__':
    _test()
