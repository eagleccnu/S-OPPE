S-OPPE: Self-training Improves Prediction of NMR Order Parameters
====================================================

Overview
--------
S-OPPE is a machine-learning framework for predicting residue-level NMR order parameters (S²) in proteins.
It begins with 26 proteins containing experimental S² measurements and expands the training set using a
self-training strategy with 2755 unlabeled proteins. The framework iteratively refines teacher–student
models and ultimately produces a robust S² prediction.

Features
--------
- Self-training framework with iterative pseudo-label generation
- Multi-modal residue-level feature integration
- Teacher–student evolution: SVR → GBRT → XGBoost → Random Forest
- High-quality residue-level prediction suitable for structure/dynamics analysis
- Reproducible training and inference scripts

Repository Structure
--------------------
data_files/           - training datasets
model_files/          - Intermediate and final trained models
pdb_test/             - Test proteins used for evaluation
nmrstar/              - NMR-STAR files for experimental S²
train.py              - Multi-round self-training pipeline
predict.py            - S² prediction script for new proteins
utils.py              - Feature extraction and data parsing utilities

Required dependencies:
- numpy
- scipy
- scikit-learn
- xgboost
- dill
- pandas

Training
--------
To reproduce the full self-training pipeline:

python train.py

This executes five rounds of teacher–student learning and produces:

model_files/model_2781_rf.pkl  (final trained model)

Prediction
----------
To predict S² values for a protein:

1. Place the target PDB file in pdb_test/
2. Ensure the matching BMRB entry exists in nmrstar/
3. Run:

python predict.py

Per-protein PCC, MAE, RMSE metrics will be saved to results.csv.

