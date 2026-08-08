from itertools import product
from pathlib import Path

import numpy as np
import scipy.io
from scipy.stats import ttest_1samp
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import baseline_funcs
import centering_funcs
import scaling_funcs
import scatter_funcs
import smoothing_funcs


#SOURCE_DATASET = "wheatA1_20048"
#TARGET_DATASET = "wheatA3_48200"
SOURCE_DATASET = "nirshootout1"
TARGET_DATASET = "nirshootout2"
DATA_DIR = Path(__file__).resolve().parent.parent / "code_matlab"

#task = "fixed_lv_range"
task = "umprecht_lv_test"

CV_FOLDS = 4
LV_RANGE = range(6, 11)
UMPRECHT_LV_RANGE = range(1, 11)
RPD_THRESHOLD = [8.1, 6.5, 5.0, 3.1]
RANDOM_STATE = 42

PREPROCESSING_MODULES = [
    smoothing_funcs,
    baseline_funcs,
    scatter_funcs,
    centering_funcs,
    scaling_funcs,
]


def load_domain_data(data_dir, source_name, target_name):
    source = scipy.io.loadmat(data_dir / f"{source_name}.mat")
    target = scipy.io.loadmat(data_dir / f"{target_name}.mat")
    return {
        "X_source": source["X"],
        "y_source": source["y"].ravel(),
        "X_source_test": source.get("Xtest"),
        "y_source_test": source.get("ytest").ravel() if "ytest" in source else None,
        "X_target": target["X"],
        "X_target_test": target["Xtest"],
        "y_target_test": target["ytest"].ravel(),
    }


def apply_preprocessing(X_train, X_other, indices):
    X_train_proc = X_train.copy()
    X_other_proc = X_other.copy() if X_other is not None else None
    for module, method_idx in zip(PREPROCESSING_MODULES, indices):
        method = module.methods[method_idx]
        X_train_proc, X_other_proc = method(X_train_proc, X_other_proc)
    return X_train_proc, X_other_proc


def combination_name(indices):
    names = []
    for module, method_idx in zip(PREPROCESSING_MODULES, indices):
        module_name = module.__name__.replace("_funcs", "")
        method_name = module.methods[method_idx].__name__
        names.append(f"{module_name}:{method_name}")
    return " + ".join(names)


def valid_lv(lv, X_train):
    return max(1, min(lv, X_train.shape[0] - 1, X_train.shape[1]))


def cross_validated_rmse_by_lv(X, y, indices, lv_range):
    residuals_by_lv = cross_validated_residuals_by_lv(X, y, indices, lv_range)
    metrics_by_lv = {}
    for lv, residuals in residuals_by_lv.items():
        metrics_by_lv[lv] = (
            float(np.sqrt(np.mean(residuals ** 2))),
            float(np.std(residuals, ddof=1)),
        )
    return metrics_by_lv


def cross_validated_residuals_by_lv(X, y, indices, lv_range):
    residuals_by_lv = {lv: np.zeros_like(y, dtype=float) for lv in lv_range}
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    max_requested_lv = max(lv_range)

    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]

        X_train_proc, X_val_proc = apply_preprocessing(X_train_fold, X_val_fold, indices)
        max_components = valid_lv(max_requested_lv, X_train_proc)
        model = PLSRegression(n_components=max_components, scale=False)
        model.fit(X_train_proc, y_train_fold)

        for lv in lv_range:
            n_components = min(lv, max_components)
            coef = model.x_rotations_[:, :n_components] @ model.y_loadings_[:, :n_components].T
            y_pred = ((X_val_proc - model._x_mean) @ coef + model._y_mean).ravel()
            residuals_by_lv[lv][val_idx] = y[val_idx] - y_pred

    return residuals_by_lv


def select_lv_by_umprecht_test(residuals_by_lv):
    lv_values = sorted(residuals_by_lv)
    for current_lv, next_lv in zip(lv_values[:-1], lv_values[1:]):
        current_sq = residuals_by_lv[current_lv] ** 2
        next_sq = residuals_by_lv[next_lv] ** 2
        improvement = current_sq - next_sq
        _, p_value = ttest_1samp(improvement, 0.0, alternative="greater")
        if not np.isfinite(p_value) or p_value >= 0.05:
            return current_lv
    return lv_values[-1]


def source_cv_candidates(X, y, indices):
    if task == "fixed_lv_range":
        metrics_by_lv = cross_validated_rmse_by_lv(X, y, indices, LV_RANGE)
        return [
            {"lv": lv, "rmsecv": rmsecv, "rmsecv_std": rmsecv_std}
            for lv, (rmsecv, rmsecv_std) in metrics_by_lv.items()
        ]
    if task == "umprecht_lv_test":
        residuals_by_lv = cross_validated_residuals_by_lv(X, y, indices, UMPRECHT_LV_RANGE)
        selected_lv = select_lv_by_umprecht_test(residuals_by_lv)
        residuals = residuals_by_lv[selected_lv]
        return [{
            "lv": selected_lv,
            "rmsecv": float(np.sqrt(np.mean(residuals ** 2))),
            "rmsecv_std": float(np.std(residuals, ddof=1)),
        }]
    raise ValueError(f"Unknown task: {task}")


def rbf_mmd_squared(y_source_pred, y_target_pred):
    ys = np.asarray(y_source_pred, dtype=float).reshape(-1, 1)
    yt = np.asarray(y_target_pred, dtype=float).reshape(-1, 1)
    ns = ys.shape[0]
    nt = yt.shape[0]

    source_dist = np.abs(ys - ys.T)
    gamma = np.median(source_dist[np.triu_indices(ns)])
    if not np.isfinite(gamma) or gamma <= 0:
        gamma = np.finfo(float).eps

    kss = np.exp(-((ys - ys.T) ** 2) / gamma)
    ktt = np.exp(-((yt - yt.T) ** 2) / gamma)
    kst = np.exp(-((ys - yt.T) ** 2) / gamma)

    source_term = (np.sum(kss) - np.trace(kss)) / (ns * (ns - 1))
    target_term = (np.sum(ktt) - np.trace(ktt)) / (nt * (nt - 1))
    cross_term = 2.0 * np.mean(kst)
    return float(max(source_term + target_term - cross_term, 0.0))


def fit_pls_after_preprocessing(X_source, y_source, X_other, indices, lv):
    X_source_proc, X_other_proc = apply_preprocessing(X_source, X_other, indices)
    model = PLSRegression(n_components=valid_lv(lv, X_source_proc), scale=False)
    model.fit(X_source_proc, y_source)
    y_source_pred = model.predict(X_source_proc).ravel()
    y_other_pred = model.predict(X_other_proc).ravel() if X_other_proc is not None else None
    return model, X_source_proc, X_other_proc, y_source_pred, y_other_pred


def regression_metrics(y_true, y_pred):
    rmsep = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmsep": rmsep,
        "r2": float(r2_score(y_true, y_pred)),
        "rpd": float(np.std(y_true, ddof=1) / rmsep) if rmsep > 0 else float("inf"),
    }


def evaluate_candidates(X_source, y_source, X_target, rpd_threshold):
    threshold = float(np.std(y_source, ddof=1) / rpd_threshold)
    all_indices = product(*[range(len(module.methods)) for module in PREPROCESSING_MODULES])
    candidates = []

    for combo_idx, indices in enumerate(all_indices, start=1):
        indices = tuple(indices)
        if combo_idx % 50 == 0:
            print(f"Evaluated {combo_idx} preprocessing combinations...")
        for cv_result in source_cv_candidates(X_source, y_source, indices):
            lv = cv_result["lv"]
            rmsecv = cv_result["rmsecv"]
            rmsecv_std = cv_result["rmsecv_std"]
            if rmsecv >= threshold:
                continue

            _, _, _, y_source_pred, y_target_pred = fit_pls_after_preprocessing(
                X_source, y_source, X_target, indices, lv
            )
            mmd = rbf_mmd_squared(y_source_pred, y_target_pred)
            candidates.append({
                "indices": indices,
                "methods": combination_name(indices),
                "lv": lv,
                "rmsecv": rmsecv,
                "rmsecv_std": rmsecv_std,
                "mmd": mmd,
            })

    if not candidates:
        raise RuntimeError(
            f"No candidate passed the RMSECV threshold "
            f"tau={threshold:.6f} (std(y_source) / {rpd_threshold})."
        )

    return candidates, threshold


def evaluate_selected_candidate(data, candidate):
    _, _, X_target_test_proc, _, y_target_pred = fit_pls_after_preprocessing(
        data["X_source"],
        data["y_source"],
        data["X_target_test"],
        candidate["indices"],
        candidate["lv"],
    )
    target_metrics = regression_metrics(data["y_target_test"], y_target_pred)

    source_metrics = None
    if data["X_source_test"] is not None and data["y_source_test"] is not None:
        _, _, _, _, y_source_test_pred = fit_pls_after_preprocessing(
            data["X_source"],
            data["y_source"],
            data["X_source_test"],
            candidate["indices"],
            candidate["lv"],
        )
        source_metrics = regression_metrics(data["y_source_test"], y_source_test_pred)

    return target_metrics, source_metrics


def main():
    data = load_domain_data(DATA_DIR, SOURCE_DATASET, TARGET_DATASET)
    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Target dataset: {TARGET_DATASET}")
    print(f"LV selection task: {task}")
    if task == "fixed_lv_range":
        print(f"LV range: {min(LV_RANGE)}-{max(LV_RANGE)}")
    elif task == "umprecht_lv_test":
        print(f"Umprecht-style LV test range: {min(UMPRECHT_LV_RANGE)}-{max(UMPRECHT_LV_RANGE)}")

    rpd_thresholds = RPD_THRESHOLD
    if isinstance(rpd_thresholds, (int, float)):
        rpd_thresholds = [float(rpd_thresholds)]

    for rpd_threshold in rpd_thresholds:
        print("\n--- Umprecht-style MMD Baseline ---")
        print(f"RMSECV threshold rule: tau = std(y_source) / {rpd_threshold}")

        try:
            candidates, threshold = evaluate_candidates(
                data["X_source"], data["y_source"], data["X_target"], rpd_threshold
            )
        except RuntimeError as error:
            print(str(error))
            continue

        candidates = sorted(candidates, key=lambda item: item["mmd"])
        best = candidates[0]
        target_metrics, source_metrics = evaluate_selected_candidate(data, best)

        print(f"Qualified candidates: {len(candidates)}")
        print(f"RMSECV threshold tau: {threshold:.6f}")
        print(f"Selected LV: {best['lv']}")
        print(f"Selected preprocessing: {best['methods']}")
        print(f"RMSECV: {best['rmsecv']:.6f}")
        print(f"MMD: {best['mmd']:.6f}")
        print("\nTarget-domain test performance:")
        print(f"RMSEP: {target_metrics['rmsep']:.6f}")
        print(f"R2:    {target_metrics['r2']:.6f}")
        print(f"RPD:   {target_metrics['rpd']:.6f}")
        if source_metrics is not None:
            print("\nSource-domain test performance:")
            print(f"RMSEP: {source_metrics['rmsep']:.6f}")
            print(f"R2:    {source_metrics['r2']:.6f}")
            print(f"RPD:   {source_metrics['rpd']:.6f}")


if __name__ == "__main__":
    main()
