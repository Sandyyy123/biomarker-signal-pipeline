"""Synthetic multi-sensor wearable data generator.

Produces per-subject PPG + accelerometer + skin-temperature windows with a known
ground-truth label (e.g. an elevated-stress / arousal state) so the rest of the
pipeline can be exercised end-to-end without any real patient data.

The point of the synthetic generator is that each *subject* has their own baseline
and noise profile. That is what makes subject-wise (leave-one-subject-out)
validation meaningful and row-wise validation optimistic.
"""

from __future__ import annotations

import numpy as np

FS = 64  # Hz, typical wearable PPG sampling rate
WINDOW_SEC = 30


def _ppg_window(hr_bpm: float, fs: int, seconds: int, rng: np.random.Generator,
                motion: float) -> np.ndarray:
    """A crude but physiologically-shaped PPG: cardiac fundamental + harmonic,
    baseline wander, and motion-correlated corruption."""
    n = fs * seconds
    t = np.arange(n) / fs
    f = hr_bpm / 60.0
    ppg = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
    baseline = 0.6 * np.sin(2 * np.pi * 0.15 * t)          # respiration/drift
    motion_noise = motion * rng.standard_normal(n).cumsum() / np.sqrt(n)
    white = 0.05 * rng.standard_normal(n)
    return ppg + baseline + motion_noise + white


def make_subject(subject_id: int, n_windows: int, rng: np.random.Generator) -> dict:
    """One subject, n_windows labelled 30-second windows across all sensors."""
    # Per-subject baseline heart rate and noise character -> the 'subject effect'.
    base_hr = rng.uniform(58, 78)
    base_temp = rng.uniform(33.5, 35.5)
    subj_motion = rng.uniform(0.02, 0.12)

    X_ppg, X_acc, X_temp, y = [], [], [], []
    for _ in range(n_windows):
        label = int(rng.random() < 0.4)                     # arousal state present?
        hr = base_hr + (14 if label else 0) + rng.normal(0, 3)
        motion = subj_motion * (1.6 if label else 1.0)

        ppg = _ppg_window(hr, FS, WINDOW_SEC, rng, motion)
        # accelerometer magnitude: higher & burstier during arousal/motion
        acc = np.abs(rng.normal(motion, motion, FS * WINDOW_SEC))
        temp = base_temp + (0.3 if label else 0.0) + rng.normal(0, 0.15,
                                                                FS * WINDOW_SEC)

        X_ppg.append(ppg)
        X_acc.append(acc)
        X_temp.append(temp)
        y.append(label)

    return {
        "subject_id": subject_id,
        "ppg": np.array(X_ppg),
        "acc": np.array(X_acc),
        "temp": np.array(X_temp),
        "y": np.array(y),
        "true_hr_offset": 14.0,     # ground truth the pipeline should recover
    }


def make_cohort(n_subjects: int = 12, windows_per_subject: int = 40,
                seed: int = 7) -> list[dict]:
    """A small clinical-style cohort. Deliberately small to make overfitting real."""
    rng = np.random.default_rng(seed)
    return [make_subject(i, windows_per_subject, rng) for i in range(n_subjects)]
