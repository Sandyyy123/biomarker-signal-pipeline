"""Smoke + correctness tests. Run: `pytest -q`"""

import numpy as np

from pipeline import synth, validation
from pipeline.features import subject_feature_matrix, estimate_hr, FEATURE_ORDER
from pipeline.synth import FS, WINDOW_SEC


def test_synth_shapes():
    cohort = synth.make_cohort(n_subjects=3, windows_per_subject=10, seed=1)
    assert len(cohort) == 3
    s = cohort[0]
    assert s["ppg"].shape == (10, FS * WINDOW_SEC)
    assert set(np.unique(s["y"])).issubset({0, 1})


def test_hr_estimation_recovers_frequency():
    # a clean 72 bpm (1.2 Hz) sine should be recovered within a few bpm
    fs, secs = FS, WINDOW_SEC
    t = np.arange(fs * secs) / fs
    ppg = np.sin(2 * np.pi * 1.2 * t)
    assert abs(estimate_hr(ppg, fs) - 72) < 5


def test_feature_matrix_no_nan():
    cohort = synth.make_cohort(n_subjects=2, windows_per_subject=8, seed=2)
    X, y, sid = subject_feature_matrix(cohort[0])
    assert X.shape[1] == len(FEATURE_ORDER)
    assert not np.isnan(X).any()


def test_loso_not_more_optimistic_than_rowwise():
    """The whole thesis: row-wise should not UNDER-report vs LOSO on this data."""
    cohort = synth.make_cohort(n_subjects=10, windows_per_subject=40, seed=7)
    Xs, ys, sids = [], [], []
    for subj in cohort:
        X, y, sid = subject_feature_matrix(subj)
        Xs.append(X); ys.append(y); sids.append(sid)
    X = np.vstack(Xs); y = np.concatenate(ys); sid = np.concatenate(sids)
    loso = validation.loso_scores(X, y, sid).mean()
    row = validation.rowwise_scores(X, y, k=5).mean()
    assert row + 1e-9 >= loso  # row-wise is optimistic (>=), never pessimistic


def test_bland_altman_zero_bias_on_identical():
    x = np.array([60.0, 65, 70, 75, 80])
    ba = validation.bland_altman(x, x)
    assert abs(ba["bias"]) < 1e-9
    assert abs(ba["sd_diff"]) < 1e-9
