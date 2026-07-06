# Spam Detection with String Kernels

**Applied Machine Learning in Python · Project 8 · LMU Munich**
Florian Wienand & Ming Zhou

▶ **6-minute explainer video:** [youtu.be/BigcTEIFCqI](https://youtu.be/BigcTEIFCqI)

String-kernel SVMs for SMS spam detection. We compare two string kernels, a
**spectrum (character k-mer) kernel** and a **Levenshtein (edit-distance)
kernel**, against a word **TF-IDF + linear-SVM** baseline. The string-kernel
models all run through the same `courselib` `BinaryKernelSVM`, so only the
representation changes, not the solver; a week-10 **MLP** on the same TF-IDF
features is the second course model. Beyond the comparison we study (i) the
non-PSD edit-distance kernel and how to use it correctly, (ii) each
representation's robustness to adversarial **obfuscation**, and (iii) a **convex
mixture** of the two kernels. The report has the full story; this file is how to
run things.

## Key findings

- On clean text, word TF-IDF and the spectrum kernel are statistically tied
  (~97% accuracy, ROC-AUC ≈ 0.99).
- The Levenshtein RBF kernel is **not positive semi-definite**. A diagonal PSD
  shift over-regularizes into a majority-class classifier; a **kernel-PCA**
  projection recovers a stable, high-precision one.
- Under **obfuscation** (leetspeak + spacing, applied to spam only), the clean
  ranking **inverts**: character n-grams collapse, word features degrade, and the
  edit-distance kernel becomes the most robust. The best kernel depends on the
  threat model.
- A mixture `αK_spectrum + (1−α)K_edit` at α = 0.25 keeps the edit kernel's
  robustness (recall 0.81 at max attack) at near-spectrum clean accuracy (96.2%).

## Repository structure

```
project8_spam/
├── run_all.py            # master script: every experiment in order
├── run_experiments.py    # main comparison: metrics, ROC/PR, heatmaps
├── run_obfuscation.py    # obfuscation robustness
├── run_sweep.py          # sigma / k sensitivity
├── run_interpolation.py  # kernel mixture
├── data/SMSSpamCollection    # dataset (ships with the repo)
├── results/              # generated JSON (metrics, sweeps)
├── figures/              # generated figures
├── src/                  # preprocessing, kernels, evaluation
├── report/               # PDF report from Overleaf
└── courselib/            # vendored course library (BinaryKernelSVM, ...)
```

## Setup and run

Python 3.10, from the project root so `courselib` is importable:

```bash
conda activate applied_ml
pip install -r requirements.txt
python run_all.py        # rebuilds every figure and JSON, ~20 min total
```

`run_all.py` chains the four experiment scripts; each also runs on its own.
The dataset ships with the repo (SMS Spam Collection, 5,574 messages, ~13%
spam), so nothing is downloaded. Note: the
file is latin-1 encoded, so a UTF-8 mirror turns `£100` into `Â£100` and shifts
the character kernels.

## Results (mean ± std over 5 seeds)

| Model | Accuracy | Precision | Recall | ROC-AUC | #SV |
|---|---|---|---|---|---|
| TF-IDF + linear (baseline) | 97.5 ± 1.0 | 0.98 | 0.84 | 0.986 | 343 |
| MLP (TF-IDF) | 91.8 ± 0.6 | 1.00 | 0.40 | 0.985 | n/a |
| Spectrum k=3 (normalized) | 97.0 ± 0.9 | 0.99 | 0.79 | **0.992** | 291 |
| Spectrum k=2 (normalized) | 96.8 ± 1.2 | 1.00 | 0.76 | 0.991 | 189 |
| Levenshtein direct (PSD shift) | 71.8 ± 28.5 | 0.33 | 0.25 | 0.703 | 419 |
| Levenshtein kPCA | 93.7 ± 0.9 | 0.99 | 0.54 | 0.980 | 311 |

Majority-class (all-ham) accuracy is 86.4%. The direct-shift row is kept only to
show the failed PSD repair: its ROC-AUC of 0.70 means the kernel still ranks spam
above ham, but its threshold is broken (kPCA fixes it). The report covers the
obfuscation and mixture results in full.

## Reproducibility

Stratified splits, seeds `[0, 1, 2, 3, 4]`, all numbers reported as mean ± std;
figures and the PSD diagnostic use seed 0. Every number in the report and this
file is reproduced by `run_all.py` and saved to `results/*.json`.

## References

- Lodhi et al. (2002), *Text Classification Using String Kernels*, JMLR.
- Ristad & Yianilos (1998), *Learning String Edit Distance*.