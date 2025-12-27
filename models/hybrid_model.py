"""
Hybrid CNN -> Topological Converter -> Transformer

Provides `CNNTransformer` which converts raw EEG (B,SEQ,CH) to topographic
images, encodes them with a CNN per-frame, and feeds the temporal embeddings
into the existing transformer backbone to produce source predictions.
"""
from typing import Optional
import torch
import torch.nn as nn

from utils.topological_converter import TopologicalConverter
from models.cnn_spatial import CNNSpatialEncoder
from models.transformer_model import EEGSourceTransformerV2


class CNNTransformer(nn.Module):
    """Hybrid model: Topological convert -> CNN spatial encoder -> Transformer

    Args:
        config: configuration object with model hyperparameters
        topo_converter: optional TopologicalConverter instance
        cnn_encoder: optional CNNSpatialEncoder instance
        fusion: unused for now (kept for API compatibility)
    """

    def __init__(self, config, topo_converter: Optional[TopologicalConverter] = None, cnn_encoder: Optional[CNNSpatialEncoder] = None, fusion: str = 'add'):
        super().__init__()
        self.config = config
        image_size = getattr(config, 'TOPO_IMAGE_SIZE', 64)

        # Topological converter
        self.topo = topo_converter if topo_converter is not None else TopologicalConverter(electrode_file=getattr(config, 'ELECTRODE_FILE', 'anatomy/electrode_75.mat'), image_size=image_size)

        # CNN encoder
        self.cnn = cnn_encoder if cnn_encoder is not None else CNNSpatialEncoder.from_config(config)

        # Transformer core: create a V2 transformer and make it accept precomputed embeddings
        self.transformer = EEGSourceTransformerV2(
            eeg_channels=self.cnn.out_dim,
            source_regions=config.SOURCE_REGIONS,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT
        )

        # Replace input projection so transformer accepts precomputed embeddings
        self.transformer.input_projection = nn.Identity()

        # If cnn output dim doesn't match d_model, add projection
        if self.cnn.out_dim != config.D_MODEL:
            self.cnn_to_model = nn.Linear(self.cnn.out_dim, config.D_MODEL)
        else:
            self.cnn_to_model = nn.Identity()

    def forward(self, eeg_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            eeg_batch: (B, SEQ, CH)

        Returns:
            (B, SEQ, SOURCE_REGIONS)
        """
        device = eeg_batch.device

        # Convert to topo images (returns tensor on device)
        topo_images = self.topo.to_image_tensor(eeg_batch, device=device)
        # topo_images: (B, SEQ, 1, H, W)

        # CNN encode
        cnn_emb = self.cnn(topo_images)  # (B, SEQ, cnn_out_dim)

        # Project cnn embedding to d_model if needed
        emb = self.cnn_to_model(cnn_emb)

        # Pass embeddings into transformer; transformer expects (B,SEQ,d_model)
        out = self.transformer(emb)
        return out


def _test():
    import torch
    from configs.config import Config
    cfg = Config
    # small test settings
    cfg.SOURCE_REGIONS = 10
    cfg.D_MODEL = 64
    cfg.NHEAD = 8
    cfg.NUM_LAYERS = 2
    cfg.DIM_FEEDFORWARD = 128
    cfg.DROPOUT = 0.1
    cfg.TOPO_IMAGE_SIZE = 64

    model = CNNTransformer(cfg)
    x = torch.randn(2, 16, cfg.EEG_CHANNELS)
    out = model(x)
    print('out', out.shape)


if __name__ == '__main__':
    _test()
