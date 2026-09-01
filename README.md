> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# biomarker-signal-pipeline

Raw multi-sensor wearable data → **clinically-defensible measures**, validated against a reference standard, with the overfitting guardrails that matter on a small clinical cohort.

This is a compact, fully-runnable demo of how I build biosignal pipelines: physiologically-grounded feature extraction, and — the part that actually decides whether a measure is trustworthy — **subject-wise validation** that reports the honest error, not the optimistic one.

```
raw sensors ──▶ conditioning ──▶ biomarkers ──▶ small-data-safe model ──▶ validation
 PPG / ACC       de-noise         HR, HRV,        L2 logistic,             LOSO + Bland-Altman
 TEMP            drift removal    resp power,      standardised,            + ICC + bootstrap CI
                 motion reject    motion, temp     low capacity
```

## Why this repo exists

A number on a wearable's screen is not a clinical measure until it (a) survives signal conditioning and (b) is shown to agree with a reference standard **on subjects the model never saw**. On a small cohort the real risk is *optimistic error estimates* — a pipeline that looks accurate because it memorised your patients.

The demo makes that failure visible. On the same synthetic cohort:

| Validation scheme | Reported accuracy | What it means |
|---|---|---|
| Row-wise 5-fold | ~0.86 | **Optimistic** — leaks subject identity across folds |
| Leave-one-subject-out | ~0.80 (95% CI ~0.70–0.88) | **Honest** — what you'd see on the next patient |

The gap between them is exactly what row-wise cross-validation hides.

## Run it

```bash
pip install -r requirements.txt
python main.py      # end-to-end demo with printed validation report
pytest -q           # 5 tests, including the LOSO-vs-rowwise guarantee
```

No real patient data is used — `pipeline/synth.py` generates a small multi-sensor
cohort where each subject has their own baseline, which is what makes subject-wise
validation meaningful.

## Module map

| File | Role | Real-deployment equivalent |
|---|---|---|
| `pipeline/synth.py` | Synthetic PPG/accelerometer/temperature cohort with ground truth | Your ingested device streams |
| `pipeline/conditioning.py` | Drift removal, band-limiting, motion-artifact rejection | SciPy Butterworth / wavelet DSP |
| `pipeline/features.py` | HR, HRV proxy, respiratory band power, motion, temperature | Physiologically-grounded biomarkers |
| `pipeline/validation.py` | LOSO vs row-wise CV, Bland-Altman, ICC(2,1), bootstrap CIs | The clinical validation report |
| `main.py` | End-to-end run + printed report | The pipeline entry point |

## The overfitting guardrails (what actually generalises at small n)

- **Subject-wise splits (LOSO)** — never split by row.
- **Low-capacity, regularised model** — standardised features + L2, few interpretable biomarkers over deep capacity.
- **Leakage discipline** — no feature encodes patient/device/time identity.
- **Bootstrap confidence intervals** — a CI, not a single accuracy point.
- **Agreement statistics** — Bland-Altman + ICC against the reference standard, the way a clinician or regulator reads it.

---

Dr. Sandeep Grover — PhD (Data Science), clinical research at Charité Berlin, Lübeck, Tübingen, Bonn/Marburg. Wearable-vs-clinical-reference validation and signal processing on small clinical cohorts.
