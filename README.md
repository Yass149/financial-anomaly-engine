# Most of the files are not present here as the project is still under review and grading by the panel 
# Privacy-Preserving Machine Learning for Financial Crime Detection

MSc Data Science and Advanced Computing dissertation project — University of Reading, Department of Computer Science.
**Candidate:** Yassine Malal · **Supervisor:** Dr. Xiaomin Chen

The dissertation itself is submitted separately and is not held in this repository, which contains the code, the experiment artefacts and the results console.


## What this is

A GRU-based fraud classifier trained on the Elliptic Bitcoin transaction graph (Weber et al., 2019) and, for generalisation testing, PaySim's synthetic mobile-money simulator (Lopez-Rojas et al., 2016), once normally and once under Differentially Private SGD (Abadi et al., 2016, via [Opacus](https://opacus.ai/)) — with the goal of *measuring*, rather than assuming, what Differential Privacy actually costs a fraud-detection model on realistically imbalanced data.

The central finding: naive DP-SGD combined with standard loss-based class weighting does not moderately degrade minority-class (fraud) recall — it eliminates it (0.727 → 0.000, reproduced on three independent seeds), because DP-SGD's per-sample gradient clipping mechanically cancels the effect of class weighting before it reaches the optimiser. Three controls test that mechanism directly rather than inferring it from the outcome: removing the class weight entirely changes the trained model by 0.0003 average precision, loosening the clipping norm restores recall from 0.000 to 0.982, and the same loosening applied to the oversampling fix moves recall by 0.005. A further refinement the controls forced: the collapsed model ranks fraud about as well as the non-private baseline (average precision 0.43 vs 0.42, chance 0.065) — what DP-SGD destroys is not the learned signal but the ability to place the decision boundary. Replacing loss reweighting with data-level minority oversampling restores recall to parity with the non-private baseline (0.743). Its reported privacy budget also falls (ε = 1.828 vs 2.274), but that is an artefact of the larger dataset rather than a privacy gain: Opacus charges per record, and oversampling puts five copies of each fraud window in the training set, so removing one fraud transaction removes five records. Corrected for group privacy the budget for a fraud transaction is ≈ 5 × 1.828 = 9.1 — roughly four times *weaker* than 2.274. The fix trades privacy for detection, and the headline ε hides the trade. A class-stratified membership inference audit then shows this fix doesn't leave privacy risk evenly distributed: DP-SGD substantially reduces the severe fraud-class leakage present in the non-private baseline (AUC 0.860 → 0.630) while increasing safe-class leakage (0.608 → 0.686). The same collapse-and-recovery pattern reproduces on PaySim, a structurally unrelated dataset ~140x larger with fraud ~76x rarer.

## Repository structure

```
.
├── dashboard/index.html       # THE viva demo — standalone console, opens offline in a browser
├── app.py                     # the older Streamlit dashboard (CPU-only, no torch, no GPU)
├── paths.py                   # single source of truth for where every artefact lives
│
├── results/                   # everything produced by a run — never hand-edited
│   ├── metrics/               # training_metrics_*.json, scalability_metrics.json
│   ├── attacks/               # mia_results*.json
│   ├── checkpoints/           # fraud_detector_*.pth
│   └── demo_cases.json        # precomputed predictions the dashboard reads
│
├── src/                       # importable project code
│   ├── model_arch.py          # the model definition (kept in sync with hpc/)
│   ├── load_model.py          # minimal hand-to-someone-else entry point
│   └── predict.py
│
├── hpc/                       # the RACC2 cluster bundle — self-contained
│   ├── data_loader.py, paysim_loader.py   # cuDF zero-copy data pipelines
│   ├── train.py, train_baseline.py        # Elliptic: DP-SGD and baseline
│   ├── train_paysim.py                    # PaySim training
│   ├── membership_inference_attack.py     # class-stratified MIA
│   └── benchmark_scalability.py           # cuDF vs pandas timing
│
└── tests/                     # 53 tests tying the write-up to the artefacts
```

**Two rules worth knowing before you move anything.**

`paths.py` owns every location. Code refers to results by bare name
(`training_metrics_oversampled.json`) and `paths.artefact()` resolves it, so
relocating a directory is a one-line change there rather than a hunt through the
dashboard, the tests and four figure generators.

`hpc/` is deliberately outside that scheme. Those files are copied to the
cluster and run from their own working directory, reading and writing bare
filenames with no knowledge of this layout. That is the contract with SLURM —
do not make them import `paths.py`.

Trained checkpoints (`results/checkpoints/*.pth`) **are** committed
deliberately: some runs predate random seeding and cannot be regenerated
exactly, which makes them the only provenance for the reported numbers, and at
~4MB the cost is negligible. The dissertation and everything built from it
(figures, LaTeX, slides, poster) are kept locally but **not** tracked — they
are submitted separately from this repository. `.gitignore` documents the
reasoning.

## Live demo console

`dashboard/index.html` is a standalone, dependency-free results console for
demoing the project live (viva Q&A) -- open it directly in any browser, no
server, no Python, no Streamlit. Six findings behind a persistent sidebar
nav so any question can be answered in one click regardless of order asked.
Every number traces to `results/metrics/*.json`, `results/attacks/*.json`,
or `results/demo_cases.json` -- cross-checked against
the metrics and attack JSON directly, nothing invented. Supersedes `app.py` (kept for reference, not the
recommended demo path -- Streamlit's async load/render made it unreliable
to verify and, at narrow widths, visibly broken).

## Running the dashboard locally

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

The dashboard deliberately does **not** load the model or run inference. Predictions and feature attributions for the real held-out cases are computed once on the cluster by `hpc/export_demo_cases.py` and read from `results/demo_cases.json`, so the dashboard needs only `streamlit`, `numpy` and `matplotlib` — no torch, no GPU, no checkpoint. Every figure it shows is read from the committed metrics JSON, never hardcoded.

## Running training on HPC (RACC2 / SLURM)

Requires a GPU node with PyTorch, [cuDF (RAPIDS)](https://rapids.ai/), and Opacus — see `requirements-hpc.txt` for the RAPIDS conda install command. The environment used was a conda env built on scratch with the `gpuscavenger` SLURM partition. Data paths in `hpc/data_loader.py`, `hpc/train.py`, `hpc/train_baseline.py`, and `hpc/membership_inference_attack.py` are hardcoded to this project's own scratch layout and will need source edits to point elsewhere; `hpc/train_paysim.py` and `hpc/benchmark_scalability.py` expose the equivalent path as `--csv_path` instead. Wherever you place `elliptic_txs_features.csv` / `elliptic_txs_classes.csv` / `paysim.csv` (not included in this repo), update accordingly.

```bash
# Elliptic, non-private baseline
python hpc/train_baseline.py --epochs 5 --minority_oversample_factor 1.0

# Elliptic, DP-SGD (the flags used for the headline ε=1.83 result --
# the --tag is what makes this produce fraud_detector_oversampled.pth, the
# checkpoint the dashboard actually loads; omit it and you get
# fraud_detector_private.pth instead, a different file)
python hpc/train.py --noise_multiplier 1.1 --max_grad_norm 1.0 --minority_oversample_factor 5.0 --tag oversampled

# Elliptic, max_grad_norm sweep (the dissertation's §4.4 clipping-norm sweep, noise_multiplier
# held at its default 1.1)
python hpc/train.py --max_grad_norm 0.5 --minority_oversample_factor 5.0 --tag mgn0.5
python hpc/train.py --max_grad_norm 2.0 --minority_oversample_factor 5.0 --tag mgn2.0
python hpc/train.py --max_grad_norm 4.0 --minority_oversample_factor 5.0 --tag mgn4.0

# PaySim, DP-SGD with 200x oversampling (the dissertation's §4.5)
python hpc/train_paysim.py --dp --noise_multiplier 1.1 --minority_oversample_factor 200.0

# Class-stratified membership inference audit against a trained checkpoint
python hpc/membership_inference_attack.py

# cuDF vs pandas scalability benchmark
python hpc/benchmark_scalability.py
```

All three training scripts accept `--tag` to avoid overwriting existing checkpoints, and refuse to silently overwrite an existing output file of the same name. Run `--help` on any script for the full flag list.

## Datasets

- **Elliptic** — Weber, M. et al. (2019). Not redistributed here; available via [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) under its own licence.
- **PaySim** — Lopez-Rojas, E.A., Elmir, A. and Axelson, S. (2016). Synthetic, not redistributed here; available via [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1).
