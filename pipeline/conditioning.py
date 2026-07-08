"""Signal conditioning: de-noise, remove baseline drift, reject motion artifacts.

Pure NumPy so the demo runs with no heavy DSP dependency. In a real deployment
these become SciPy Butterworth / wavelet stages, but the interface is identical.
"""

from __future__ import annotations

import numpy as np


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def remove_baseline(ppg: np.ndarray, fs: int = 64) -> np.ndarray:
    """High-pass by subtracting a slow moving-average trend (drift/respiration)."""
    trend = moving_average(ppg, k=int(fs * 1.5))
    return ppg - trend


def bandlimit(ppg: np.ndarray, fs: int = 64) -> np.ndarray:
    """Light low-pass to suppress high-frequency white noise."""
    return moving_average(ppg, k=max(2, int(fs * 0.03)))


def motion_mask(acc: np.ndarray, z_thresh: float = 2.5) -> np.ndarray:
    """Flag samples where accelerometer energy is an outlier -> motion corruption."""
    energy = np.abs(acc)
    mu, sd = energy.mean(), energy.std() + 1e-9
    return (energy - mu) / sd > z_thresh


def condition_window(ppg: np.ndarray, acc: np.ndarray, fs: int = 64) -> np.ndarray:
    """Full per-window conditioning: drift removal, band-limiting, motion masking."""
    x = remove_baseline(ppg, fs)
    x = bandlimit(x, fs)
    mask = motion_mask(acc)
    if mask.any() and not mask.all():
        # replace motion-corrupted samples with the window mean (simple, honest)
        x = x.copy()
        x[mask] = x[~mask].mean()
    return x
