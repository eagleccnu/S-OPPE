"""
Evaluate the trained S-OPPE model on a set of test proteins.

"""

# from __future__ import annotations

import dill
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler

import utils


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cpu"

DATASET_PATH = "./data_files/dataset_training_2781.dat"
MODEL_PATH = "./model_files/model_2781_rf.pkl"

CSV_PDB_BMRB = "pdb_bmrb_pair.csv"
DIR_NMRSTAR = "./nmrstar/"
PDB_TEST_DIR = "./pdb_test/"
RESULT_SAVE_PATH = "results.csv"

# Test protein list
TEST_PDB_IDS = [
    "1pd7",
    "1wrs",
    "1wrt",
    "1z9b",
    "2jwt",
    "2l6b",
    "2luo",
    "2m3o",
    "2xdi",
    "4aai",
]

# Secondary-structure one-hot encoding
SECONDARY_STRUCTURE_CODE = {
    "H": np.array([1, 0, 0]).tolist(),
    "E": np.array([0, 1, 0]).tolist(),
    "C": np.array([0, 0, 1]).tolist(),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_scaler(dataset_path: str) -> RobustScaler:

    with open(dataset_path, "rb") as f:
        numpy_dataset = dill.load(f)

    numpy_dataset = np.asarray(numpy_dataset)
    scaler = RobustScaler()
    scaler.fit(numpy_dataset[:, :-1])  # exclude the last column (S2 label)
    return scaler


def build_residue_feature(res, scaler: RobustScaler, protein_length_eff: int) -> np.ndarray:

    # Flag for terminal residues
    if res.index < 3 or res.index > protein_length_eff - 4:
        flag = 1
    else:
        flag = 0

    ss_vec = SECONDARY_STRUCTURE_CODE[res.state_3]

    feature = [
        flag,
        *ss_vec,
        res.phi_var,
        res.psi_var,
        res.dist_var[0],
        res.dist_var[1],
        res.dist_var[2],
        res.dist_var[3],
        res.dist_var[4],
        res.dist_var[5],
        res.concat_num,
        res.acc,
    ]

    feature = np.asarray(feature, dtype=np.float32)[np.newaxis, :]
    feature_scaled = scaler.transform(feature)
    return feature_scaled


def smooth_sequence(values):

    if not values:
        return []

    if len(values) < 3:
        return values[:]  # nothing to smooth

    smoothed = [0.0] * len(values)
    for i in range(1, len(values) - 1):
        smoothed[i] = (values[i - 1] + values[i] + values[i + 1]) / 3.0

    smoothed[0] = values[0]
    smoothed[-1] = values[-1]
    return smoothed


def predict_protein_s2(
    pdb_id: str,
    model,
    scaler: RobustScaler,
    dict_pdb_bmrb
):

    pdb_id_upper = pdb_id.upper()
    pdb_file = f"{PDB_TEST_DIR}{pdb_id}.pdb"

    bmrb_id = dict_pdb_bmrb[pdb_id_upper]
    bmrb_file = utils.search_file_with_bmrb(bmrb_id, DIR_NMRSTAR)
    bmrb_file_full_path = f"{DIR_NMRSTAR}{bmrb_file}"

    protein = utils.protein_s2(pdb_id_upper, bmrb_id)

    protein.read_seq(pdb_file, bmrb_file_full_path)
    protein.read_dist_var_from_pdb(pdb_id_upper, pdb_file)
    protein.read_torsion_var_from_pdb(pdb_id_upper, pdb_file)
    protein.read_s2_from_star(bmrb_file_full_path)
    protein.read_concat_num_from_pdb(pdb_id_upper, pdb_file)
    protein.cal_ss_from_pdb(pdb_id_upper, pdb_file)
    protein.get_rsa(pdb_id_upper, pdb_file)

    protein.merge(pdb_file, bmrb_file_full_path)

    s2_labels: list[float] = []
    s2_preds: list[float] = []

    for idx, res in enumerate(protein.pdb_seq):
        res.index = idx  # add index for terminal flag logic

        if res.s2 < -0.5:
            # No S2 label, skip this residue
            continue

        # Experimental label, transformed as (1 - S2) to match training
        s2_labels.append(1.0 - res.s2)

        feature_scaled = build_residue_feature(
            res=res,
            scaler=scaler,
            protein_length_eff=protein.length_eff,
        )

        pred = model.predict(feature_scaled)[0]
        s2_preds.append(1.0 - float(pred))

    # Smooth predicted sequence
    s2_preds_smoothed = smooth_sequence(s2_preds)

    return s2_labels, s2_preds_smoothed


def evaluate_protein(
    pdb_id,
    labels,
    preds
):

    labels_np = np.asarray(labels, dtype=np.float32)
    preds_np = np.asarray(preds, dtype=np.float32)

    rmse = np.sqrt(np.mean((labels_np - preds_np) ** 2))
    mae = mean_absolute_error(labels_np, preds_np)

    corr = pearsonr(labels_np, preds_np)
    pcc = float(corr[0])

    print(f"PDB_ID={pdb_id.upper()}, PCC={pcc:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    return pcc, mae, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load scaler from the final training dataset
    scaler = load_scaler(DATASET_PATH)

    # Load final trained model
    model = joblib.load(MODEL_PATH)

    # Load PDB-BMRB mapping
    dict_pdb_bmrb = utils.read_pdb_bmrb_dict_from_csv(CSV_PDB_BMRB)

    pdb_ids: list[str] = []
    all_pcc: list[float] = []
    all_mae: list[float] = []
    all_rmse: list[float] = []
    all_preds: list[list[float]] = []
    all_labels: list[list[float]] = []

    for pdb_id in TEST_PDB_IDS:
        pdb_ids.append(pdb_id.upper())

        labels, preds = predict_protein_s2(
            pdb_id=pdb_id,
            model=model,
            scaler=scaler,
            dict_pdb_bmrb=dict_pdb_bmrb,
        )

        pcc, mae, rmse = evaluate_protein(
            pdb_id=pdb_id,
            labels=labels,
            preds=preds,
        )

        all_pcc.append(pcc)
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_preds.append(preds)
        all_labels.append(labels)

    # Summary over all proteins
    mean_pcc = sum(all_pcc) / len(all_pcc)
    mean_mae = sum(all_mae) / len(all_mae)
    mean_rmse = sum(all_rmse) / len(all_rmse)

    print(
        "meanPCC={:.3f}, meanMAE={:.3f}, meanRMSE={:.3f}".format(
            mean_pcc,
            mean_mae,
            mean_rmse,
        )
    )

    # Save per-protein results as DataFrame
    df = pd.DataFrame(
        {
            "PDB": pdb_ids,
            "pre": all_preds,
            "real": all_labels,
            "PCC": all_pcc,
            "MAE": all_mae,
            "RMSE": all_rmse,
        }
    )
    df.to_csv(RESULT_SAVE_PATH, index=False)
    print(f"Saved detailed results to {RESULT_SAVE_PATH}")


if __name__ == "__main__":
    main()


