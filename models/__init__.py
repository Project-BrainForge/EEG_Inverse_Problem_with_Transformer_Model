"""
Model architectures for EEG Source Localization
"""
from .transformer_model import EEGSourceTransformer, EEGSourceTransformerV2
from .hybrid_model import CNNTransformer
from .cnn_spatial import CNNSpatialEncoder

__all__ = ['EEGSourceTransformer', 'EEGSourceTransformerV2', 'CNNTransformer', 'CNNSpatialEncoder']


def create_model(model_type: str, config, **kwargs):
	"""Factory to create models by name.

	model_type: 'transformer' or 'hybrid'
	"""
	if model_type == 'hybrid':
		return CNNTransformer(config, **kwargs)
	else:
		# default: transformer
		return EEGSourceTransformerV2(
			eeg_channels=getattr(config, 'EEG_CHANNELS', 75),
			source_regions=getattr(config, 'SOURCE_REGIONS', 994),
			d_model=getattr(config, 'D_MODEL', 256),
			nhead=getattr(config, 'NHEAD', 8),
			num_layers=getattr(config, 'NUM_LAYERS', 6),
			dim_feedforward=getattr(config, 'DIM_FEEDFORWARD', 1024),
			dropout=getattr(config, 'DROPOUT', 0.1)
		)

