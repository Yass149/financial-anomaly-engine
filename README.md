# Privacy-preserving financial anomaly detection

An experimental fraud-detection study measuring the cost of Differentially Private SGD under severe class imbalance, using GRU-based transaction models on the Elliptic Bitcoin transaction graph and PaySim synthetic mobile-money data.

> **Project status:** research prototype and dissertation evidence. The dashboards are local artefacts; no financial decisioning service is deployed from this repository.

## Research question

What does Differential Privacy actually cost a fraud detector when the minority class is rare? The experiments compare non-private training with DP-SGD, class weighting and minority oversampling, then audit both detection quality and class-stratified membership-inference risk.

The main result is a mechanism, not a marketing claim: standard loss weighting can collapse minority recall under per-sample gradient clipping. Data-level oversampling restores recall, but its reported privacy budget must be corrected for duplicated records. The experiments also show that privacy risk is not evenly distributed across fraud and safe classes.

## Evidence

- Elliptic and PaySim provide structurally different fraud settings.
- DP-SGD is implemented with Opacus controls and explicit clipping/noise parameters.
- Membership-inference analysis is stratified by class.
- Results and dashboard cases trace back to committed JSON artefacts.
- The project includes tests tying the write-up to the reported outputs.

## Repository map

```text
dashboard/       dependency-free viva/results console
src/             importable model and prediction code
hpc/             RACC2/SLURM training and audit bundle
results/         metrics, attacks, checkpoints and demo cases
tests/           regression and consistency checks
paths.py         artefact-location contract
app.py           legacy Streamlit dashboard
```

The dashboard reads precomputed outputs; it does not train or perform live inference. This keeps a demo reproducible and CPU-accessible while making the distinction between experiment evidence and deployment explicit.

## Run the results console

```bash
git clone https://github.com/Yass149/financial-anomaly-engine.git
cd financial-anomaly-engine
pip install -r requirements-dashboard.txt
# open dashboard/index.html directly, or:
streamlit run app.py
```

For the headline training and audit commands, see the scripts under `hpc/`. They require the datasets, a GPU environment and (for the cluster workflow) RAPIDS/cuDF, PyTorch and Opacus. The datasets and dissertation are not redistributed here.

## Production gap

A deployable financial system would additionally need a feature-store contract, online/offline parity tests, model and threshold governance, explainability review, secure data handling, access control, drift and fairness monitoring, human review workflows and a rollback path. This repository is evidence for the research question, not authorisation to automate financial decisions.

## Reproducibility

The dashboard and experiments are designed around explicit artefact paths, committed metrics and checkpoints, and consistency checks. Re-run the tests and `scripts/check_consistency.py` before updating headline values.

## Context

MSc Data Science and Advanced Computing dissertation project, University of Reading. Candidate: Yassine Malal.
