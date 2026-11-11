from .single_edge_image_unet import SingleEdgeImageUNet

def create_model(config):
    return SingleEdgeImageUNet(input_size=config.get('input_size', config.get('input_grid_zyx', [112, 160, 160])), filters=config.get('filters', [16, 32, 64, 128, 256]), adain=False, fusion_type='distance', fusion_temperature=config.get('fusion_temperature', 0.1))
