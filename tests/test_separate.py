import torch

from bsro.model import load_config, residual_name, stem_names
from bsro.separate import separate_track, window_array


class ScaleModel(torch.nn.Module):
    """Stand-in for BS-RoFormer: returns two stems, 1x and 0.5x the input."""

    def forward(self, x):
        return torch.stack([x, 0.5 * x], dim=1)


def test_overlap_add_reconstructs_input():
    torch.manual_seed(0)
    mix = torch.randn(2, 10_000)
    out = separate_track(ScaleModel(), mix, chunk_size=1000, num_overlap=4, batch_size=3, device=torch.device("cpu"), amp_dtype=None)
    assert out.shape == (2, 2, 10_000)
    assert torch.allclose(out[0], mix, atol=1e-5)
    assert torch.allclose(out[1], 0.5 * mix, atol=1e-5)


def test_short_input_without_border_padding():
    mix = torch.randn(2, 300)
    out = separate_track(ScaleModel(), mix, chunk_size=1000, num_overlap=2, batch_size=1, device=torch.device("cpu"), amp_dtype=None)
    assert out.shape == (2, 2, 300)
    assert torch.allclose(out[0], mix, atol=1e-5)


def test_window_edges():
    w = window_array(100, 10)
    assert w[0] == 0 and w[-1] == 0 and torch.all(w[10:-10] == 1)


def test_config_stem_names():
    vocals = load_config("configs/model_bs_roformer_ep_317_sdr_12.9755.yaml")
    assert stem_names(vocals) == ["vocals"]
    assert residual_name(vocals) == "other"
    assert isinstance(vocals["model"]["freqs_per_bands"], tuple)
