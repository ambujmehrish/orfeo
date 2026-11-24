import numpy as np

from bsro.metrics import active_frames, low_frequency_fraction, sdr, si_sdr


def test_sdr_identity_and_noise():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((2, 44100)).astype(np.float32)
    assert sdr(ref, ref) >= 90
    noisy = ref + 0.1 * rng.standard_normal(ref.shape).astype(np.float32)
    assert 19 < sdr(ref, noisy) < 21


def test_si_sdr_scale_invariant():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((2, 20000)).astype(np.float32)
    noisy = ref + 0.1 * rng.standard_normal(ref.shape).astype(np.float32)
    assert abs(si_sdr(ref, 0.3 * noisy) - si_sdr(ref, noisy)) < 1e-3
    assert si_sdr(ref, ref) > 90


def test_active_frames_detects_silent_region():
    sr = 1000
    mix = np.ones((2, 10 * sr), dtype=np.float32) * 0.5
    vocals = np.zeros_like(mix)
    vocals[:, : 4 * sr] = 0.4
    active = active_frames(vocals, mix, frame=sr, threshold_db=-25.0)
    assert active.tolist() == [True] * 4 + [False] * 6


def test_low_frequency_fraction():
    sr = 8000
    t = np.arange(sr) / sr
    low = np.sin(2 * np.pi * 50 * t)[None]
    high = np.sin(2 * np.pi * 1000 * t)[None]
    assert low_frequency_fraction(low, sr, 100.0) > 0.99
    assert low_frequency_fraction(high, sr, 100.0) < 0.01
    assert abs(low_frequency_fraction(low + high, sr, 100.0) - 0.5) < 0.02
