import torch

from skill_jepa.modules import StateProjector


def test_state_projector_shapes():
    projector = StateProjector(input_dim=1024, output_dim=384, pool_grid=4)
    patch_tokens = torch.randn(2, 256, 1024)
    states = projector(patch_tokens)
    assert states["global_state"].shape == (2, 384)
    assert states["spatial_tokens"].shape == (2, 16, 384)

