from __future__ import annotations

import numpy as np

EPS = 1e-10


def energy(x: np.ndarray) -> float:
    return float(np.sum(x.astype(np.float64) ** 2))


def db(ratio: float) -> float:
    return float(10.0 * np.log10(max(ratio, EPS)))


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Plain SDR as used by MSST for its reported numbers: 10log10(|ref|^2 / |ref-est|^2)."""
    return db(energy(reference) / max(energy(reference - estimate), EPS))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.astype(np.float64).ravel()
    est = estimate.astype(np.float64).ravel()
    ref = ref - ref.mean()
    est = est - est.mean()
    scale = np.dot(est, ref) / max(np.dot(ref, ref), EPS)
    target = scale * ref
    return db(energy(target) / max(energy(est - target), EPS))


def frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    """RMS over non-overlapping frames of `frame` samples; x is (channels, time)."""
    n = x.shape[-1] // frame
    if n == 0:
        return np.zeros(0)
    y = x[..., : n * frame].reshape(x.shape[0], n, frame).astype(np.float64)
    return np.sqrt(np.mean(y**2, axis=(0, 2)))


def active_frames(stem: np.ndarray, mix: np.ndarray, frame: int, threshold_db: float) -> np.ndarray:
    """Frames where the stem is within `threshold_db` of the mix level (stem considered present)."""
    stem_rms = frame_rms(stem, frame)
    mix_rms = frame_rms(mix, frame)
    ratio_db = 20.0 * np.log10((stem_rms + EPS) / (mix_rms + EPS))
    return (ratio_db > threshold_db) & (mix_rms > 1e-4)


def low_frequency_fraction(x: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
    """Fraction of a signal's energy below cutoff_hz (vocals carry little energy below ~100 Hz)."""
    spec = np.fft.rfft(x.astype(np.float64), axis=-1)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate)
    total = power.sum()
    if total <= EPS:
        return 0.0
    return float(power[..., freqs < cutoff_hz].sum() / total)


def energy_in_frames(x: np.ndarray, mask: np.ndarray, frame: int) -> float:
    n = mask.shape[0]
    y = x[..., : n * frame].reshape(x.shape[0], n, frame).astype(np.float64)
    return float(np.sum(y[:, mask, :] ** 2))
