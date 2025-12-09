"""
Self-training pipeline for S-OPPE order parameter prediction.

Round 1:  Teacher = OPPE (model_26.pkl),   Student = SVR          -> model_126_svr.pkl
Round 2:  Teacher = model_126_svr.pkl,     Student = SVR          -> model_326_svr.pkl
Round 3:  Teacher = model_326_svr.pkl,     Student = GBR          -> model_726_gbr.pkl
Round 4:  Teacher = model_726_gbr.pkl,     Student = XGBR         -> model_1526_xgbr.pkl
Round 5:  Teacher = model_1526_xgbr.pkl,   Student = RF           -> model_2781_rf.pkl
"""

from __future__ import annotations

import json
from typing import List, Sequence, Tuple

import dill
import joblib
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import xgboost as xgb

np.random.seed(100)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ml_predict_s2(
    model_path: str,
    features: Sequence[Sequence[float]],
    scaler: RobustScaler,
) -> List[float]:

    model = joblib.load(model_path)

    s2_pred: List[float] = []

    with torch.no_grad():
        for feat in features:
            feat_arr = np.asarray(feat, dtype=np.float32)[np.newaxis, :]
            feat_scaled = scaler.transform(feat_arr)

            batch_feature = torch.tensor(feat_scaled, dtype=torch.float32, device=device)

            pred = model.predict(batch_feature.cpu().numpy())
            s2_pred.append(float(pred[0]))

    if not s2_pred:
        return []

    s2_smooth = [0.0] * len(s2_pred)
    for i in range(1, len(s2_pred) - 1):
        s2_smooth[i] = (s2_pred[i - 1] + s2_pred[i] + s2_pred[i + 1]) / 3.0

    s2_smooth[0] = s2_pred[0]
    s2_smooth[-1] = s2_pred[-1]

    return s2_smooth


def generate_pseudo_residue_data(
    teacher_model_path: str,
    scaler: RobustScaler,
    all_features: Sequence[np.ndarray],
    index_range: Tuple[int, int],
    pdb_ids: Sequence[str] | None = None,
) -> List[List[float]]:

    start_idx, end_idx = index_range
    pseudo_samples: List[List[float]] = []

    for i in range(start_idx, end_idx):
        if pdb_ids is not None:
            print(f"[Pseudo-labeling] protein index {i}, pdb_id = {pdb_ids[i]}")
        else:
            print(f"[Pseudo-labeling] protein index {i}")

        protein_features = all_features[i].tolist()
        protein_s2 = ml_predict_s2(teacher_model_path, protein_features, scaler)

        # Concatenate features and pseudo labels per residue
        pseudo_samples.extend(
            feat + [s2] for feat, s2 in zip(protein_features, protein_s2)
        )

    return pseudo_samples


def train_student_model(
    all_array_data: np.ndarray,
    student_model,
    round_name: str,
    model_path: str,
    cv_folds: int = 5,
) -> None:

    # Shuffle samples
    np.random.shuffle(all_array_data)

    # Fit scaler on all features and transform
    scaler = RobustScaler()
    all_array_data[:, :-1] = scaler.fit_transform(all_array_data[:, :-1])

    X = all_array_data[:, :-1]
    y = all_array_data[:, -1]

    print(f"[{round_name}] Feature shape: {X.shape}, Label shape: {y.shape}")

    scores = cross_val_score(
        student_model,
        X,
        y,
        cv=cv_folds,
        scoring="neg_mean_absolute_error",
    )
    mean_mae = -scores.mean()
    std_mae = scores.std()
    print(f"[{round_name}] CV MAE ({cv_folds}-fold): {mean_mae:.4f} ± {std_mae:.4f}")

    # Train on full data
    student_model.fit(X, y)

    # Save model
    joblib.dump(student_model, model_path)
    print(f"[{round_name}] Saved student model to: {model_path}")


def run_round(
    round_name: str,
    input_dataset_path: str,
    output_dataset_path: str,
    teacher_model_path: str,
    student_model,
    student_model_path: str,
    pseudo_index_range: Tuple[int, int],
    all_features: Sequence[np.ndarray],
    pdb_ids: Sequence[str],
) -> None:

    print("=" * 60)
    print(f"[{round_name}] Start")

    # 1. Load current labeled dataset
    with open(input_dataset_path, "rb") as f:
        array_data_labeled = dill.load(f)
    array_data_labeled = np.asarray(array_data_labeled)
    print(f"[{round_name}] Loaded labeled data from {input_dataset_path}, "
          f"shape = {array_data_labeled.shape}")

    # 2. Fit scaler on labeled features for pseudo-labeling
    scaler_for_pseudo = RobustScaler()
    scaler_for_pseudo.fit(array_data_labeled[:, :-1])

    # 3. Generate pseudo-labeled residue samples
    pseudo_samples = generate_pseudo_residue_data(
        teacher_model_path=teacher_model_path,
        scaler=scaler_for_pseudo,
        all_features=all_features,
        index_range=pseudo_index_range,
        pdb_ids=pdb_ids,
    )

    # Merge labeled + pseudo
    all_array_data = np.array(
        array_data_labeled.tolist() + pseudo_samples,
        dtype=np.float32,
    )

    # Save extended dataset (optional, for later analysis)
    with open(output_dataset_path, "wb") as f:
        dill.dump(all_array_data, f)
    print(f"[{round_name}] Saved extended dataset to {output_dataset_path}, "
          f"shape = {all_array_data.shape}")

    # 4. Train student model on extended dataset
    train_student_model(
        all_array_data=all_array_data,
        student_model=student_model,
        round_name=round_name,
        model_path=student_model_path,
    )

    print(f"[{round_name}] Done")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # Load global feature list and pdb IDs for 2755 pseudo-labeled proteins
    with open("feature_training_2755.dat", "rb") as f:
        all_array_feature = dill.load(f)

    with open("2755.json") as obj:
        list_pdb_file_pseudo = json.load(obj)

    # Sanity check
    if len(all_array_feature) != len(list_pdb_file_pseudo):
        raise ValueError(
            f"Length mismatch: features={len(all_array_feature)}, "
            f"ids={len(list_pdb_file_pseudo)}"
        )

    # ----------------------- Round 1 -------------------------------------
    run_round(
        round_name="Round 1: Teacher=OPPE, Student=SVR",
        input_dataset_path="./data_files/dataset_training_26.dat",
        output_dataset_path="./data_files/dataset_training_126.dat",
        teacher_model_path="./model_files/model_26.pkl",
        student_model=SVR(C=1.0, epsilon=1e-5, gamma=1e-5),
        student_model_path="./model_files/model_126_svr.pkl",
        pseudo_index_range=(0, 100),      # proteins [0, 100)
        all_features=all_array_feature,
        pdb_ids=list_pdb_file_pseudo,
    )

    # ----------------------- Round 2 -------------------------------------
    run_round(
        round_name="Round 2: Teacher=SVR, Student=SVR",
        input_dataset_path="./data_files/dataset_training_126.dat",
        output_dataset_path="./data_files/dataset_training_326.dat",
        teacher_model_path="./model_files/model_126_svr.pkl",
        student_model=SVR(C=1.0, epsilon=1e-6, gamma=1e-6),
        student_model_path="./model_files/model_326_svr.pkl",
        pseudo_index_range=(100, 300),    # proteins [100, 300)
        all_features=all_array_feature,
        pdb_ids=list_pdb_file_pseudo,
    )

    # ----------------------- Round 3 -------------------------------------
    run_round(
        round_name="Round 3: Teacher=SVR, Student=GBR",
        input_dataset_path="./data_files/dataset_training_326.dat",
        output_dataset_path="./data_files/dataset_training_726.dat",
        teacher_model_path="./model_files/model_326_svr.pkl",
        student_model=GradientBoostingRegressor(
            n_estimators=400,
            random_state=0,
        ),
        student_model_path="./model_files/model_726_gbr.pkl",
        pseudo_index_range=(300, 700),    # proteins [300, 700)
        all_features=all_array_feature,
        pdb_ids=list_pdb_file_pseudo,
    )

    # ----------------------- Round 4 -------------------------------------
    run_round(
        round_name="Round 4: Teacher=GBR, Student=XGBR",
        input_dataset_path="./data_files/dataset_training_726.dat",
        output_dataset_path="./data_files/dataset_training_1526.dat",
        teacher_model_path="./model_files/model_726_gbr.pkl",
        student_model=xgb.XGBRegressor(
            colsample_bylevel=0.3,
            learning_rate=0.1,
            max_depth=5,
            reg_alpha=10,
            n_estimators=100,
            subsample=0.8,
            n_jobs=-1,
        ),
        student_model_path="./model_files/model_1526_xgbr.pkl",
        pseudo_index_range=(700, 1500),   # proteins [700, 1500)
        all_features=all_array_feature,
        pdb_ids=list_pdb_file_pseudo,
    )

    # ----------------------- Round 5 -------------------------------------
    run_round(
        round_name="Round 5: Teacher=XGBR, Student=RF",
        input_dataset_path="./data_files/dataset_training_1526.dat",
        output_dataset_path="./data_files/dataset_training_2781.dat",
        teacher_model_path="./model_files/model_1526_xgbr.pkl",
        student_model=RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=0,
            n_jobs=-1,
        ),
        student_model_path="./model_files/model_2781_rf.pkl",
        pseudo_index_range=(1500, 2755),  # proteins [1500, 2755)
        all_features=all_array_feature,
        pdb_ids=list_pdb_file_pseudo,
    )


if __name__ == "__main__":
    main()



