"""End-to-end demo: synthetic wearable cohort -> conditioned signals ->
biomarkers -> subject-wise validation, contrasted with the optimistic row-wise
number. Run: `python main.py`
"""

from __future__ import annotations

import numpy as np

from pipeline import synth, validation
from pipeline.features import subject_feature_matrix, FEATURE_ORDER


def build_matrix(cohort):
    Xs, ys, sids = [], [], []
    for subj in cohort:
        X, y, sid = subject_feature_matrix(subj)
        Xs.append(X); ys.append(y); sids.append(sid)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(sids)


def main() -> None:
    print("Biomarker signal pipeline — demo run\n" + "=" * 42)
    cohort = synth.make_cohort(n_subjects=12, windows_per_subject=40, seed=7)
    print(f"Cohort: {len(cohort)} subjects x 40 windows "
          f"= {len(cohort) * 40} labelled 30s windows")

    X, y, sid = build_matrix(cohort)
    print(f"Feature matrix: {X.shape} | features: {FEATURE_ORDER}")
    print(f"Prevalence of arousal label: {y.mean():.2f}\n")

    # --- honest vs optimistic validation ---
    loso = validation.loso_scores(X, y, sid)
    row = validation.rowwise_scores(X, y, k=5)

    loso_ci = validation.bootstrap_ci(loso)
    print("Leave-one-subject-out accuracy (HONEST):")
    print(f"  mean {loso.mean():.3f}  95% CI [{loso_ci[0]:.3f}, {loso_ci[1]:.3f}]")
    print("Row-wise 5-fold accuracy (OPTIMISTIC — leaks subject identity):")
    print(f"  mean {row.mean():.3f}")
    gap = row.mean() - loso.mean()
    print(f"\nOptimism gap (row-wise - LOSO): {gap:+.3f}")
    print("  ^ this gap is exactly what row-wise validation hides on small cohorts.\n")

    # --- agreement of estimated HR vs a reference HR (Bland-Altman/ICC) ---
    # Treat estimated hr_bpm as the wearable measure; construct a reference with
    # small measurement noise to illustrate the agreement report.
    rng = np.random.default_rng(1)
    measured_hr = X[:, FEATURE_ORDER.index("hr_bpm")]
    reference_hr = measured_hr + rng.normal(0, 2.0, len(measured_hr))
    ba = validation.bland_altman(measured_hr, reference_hr)
    icc = validation.icc_2way(measured_hr, reference_hr)
    print("Heart-rate agreement vs reference standard:")
    print(f"  Bland-Altman bias {ba['bias']:+.2f} bpm  "
          f"LoA [{ba['loa_lower']:.2f}, {ba['loa_upper']:.2f}]")
    print(f"  ICC(2,1) = {icc:.3f}")
    print("\nDone. See README.md for how each stage maps to a real deployment.")


if __name__ == "__main__":
    main()
