"""Physiologically-grounded biomarker extraction from conditioned windows.

Every feature is interpretable (heart rate, HRV proxy, respiratory band power,
motion level, temperature) rather than an opaque embedding. At small n,
interpretable + few features generalises better and is defensible to a reviewer.
"""

from __future__ import annotations

import numpy as np

from .conditioning import condition_window


def estimate_hr(ppg: np.ndarray, fs: int = 64) -> float:
    """Heart rate from the dominant spectral peak in the cardiac band (0.7-3 Hz)."""
    n = len(ppg)
    spec = np.abs(np.fft.rfft(ppg - ppg.mean()))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not band.any():
        return float("nan")
    peak = freqs[band][np.argmax(spec[band])]
    return peak * 60.0


def hrv_proxy(ppg: np.ndarray, fs: int = 64) -> float:
    """Crude HRV proxy: std of inter-peak intervals from a simple peak finder."""
    x = ppg - ppg.mean()
    peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0))[0] + 1
    if len(peaks) < 3:
        return 0.0
    ibi = np.diff(peaks) / fs
    return float(np.std(ibi) * 1000.0)  # ms


def resp_band_power(ppg: np.ndarray, fs: int = 64) -> float:
    """Power in the respiratory band (0.1-0.4 Hz) of the raw (pre-drift-removal) PPG."""
    spec = np.abs(np.fft.rfft(ppg - ppg.mean())) ** 2
    freqs = np.fft.rfftfreq(len(ppg), d=1.0 / fs)
    band = (freqs >= 0.1) & (freqs <= 0.4)
    return float(spec[band].mean()) if band.any() else 0.0


def window_features(ppg: np.ndarray, acc: np.ndarray, temp: np.ndarray,
                    fs: int = 64) -> dict:
    resp = resp_band_power(ppg, fs)          # from raw ppg, before conditioning
    cond = condition_window(ppg, acc, fs)
    return {
        "hr_bpm": estimate_hr(cond, fs),
        "hrv_ms": hrv_proxy(cond, fs),
        "resp_power": resp,
        "motion_level": float(np.abs(acc).mean()),
        "temp_mean": float(np.mean(temp)),
    }


FEATURE_ORDER = ["hr_bpm", "hrv_ms", "resp_power", "motion_level", "temp_mean"]


def subject_feature_matrix(subject: dict, fs: int = 64):
    """Return (X, y, subject_ids) for one subject."""
    rows = []
    for i in range(len(subject["y"])):
        f = window_features(subject["ppg"][i], subject["acc"][i],
                            subject["temp"][i], fs)
        rows.append([f[k] for k in FEATURE_ORDER])
    X = np.array(rows)
    y = subject["y"]
    sid = np.full(len(y), subject["subject_id"])
    return X, y, sid
