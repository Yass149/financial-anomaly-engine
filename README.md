# Privacy-preserving financial anomaly detection

A research project studying how Differentially Private SGD changes fraud-detection performance under severe class imbalance.

> **Repository status:** the public repository currently contains this project overview only. The dissertation code, datasets, checkpoints and experiment artefacts are kept separately and are not exposed here.

## Research question

What does privacy cost when a fraud detector must learn from a rare minority class?

The underlying study compares non-private training with DP-SGD, class weighting and minority oversampling on the Elliptic Bitcoin transaction graph and PaySim synthetic mobile-money data. It measures detection quality alongside class-stratified membership-inference risk.

The central finding is a mechanism to investigate rather than a production claim: standard loss weighting can collapse minority recall under per-sample gradient clipping. Data-level oversampling can restore recall, but duplicated records change how a reported privacy budget should be interpreted.

## What the completed study covers

- Differentially Private SGD with explicit noise and clipping controls.
- Severe class imbalance and minority-class recall.
- Elliptic and PaySim generalisation experiments.
- Class-stratified membership-inference auditing.
- Reproducible metrics and dissertation evidence in the private research workspace.

## Why the code is not included here

The public repository deliberately does not redistribute the dissertation, source datasets or private experiment bundle. The datasets have their own access and licensing terms, and financial modelling artefacts need careful review before publication. This README describes the work without implying that a visitor can run the private experiments from this repository.

## Production gap

This is not a deployed financial decisioning system. A real service would additionally require versioned data contracts, online/offline feature parity, model and threshold governance, privacy review, access control, drift and fairness monitoring, human review, audit trails and rollback procedures.

## Project context

MSc Data Science and Advanced Computing dissertation project, University of Reading.

Candidate: Yassine Malal

The public repository is intentionally documentation-only at this stage. The next release should add a reviewed, runnable minimal example or a clearly linked companion repository if the underlying artefacts can be shared safely.
