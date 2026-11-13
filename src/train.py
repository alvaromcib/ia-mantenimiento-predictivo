"""Predictive maintenance on NASA C-MAPSS FD001.

Predicts Remaining Useful Life (RUL) of turbofan engines from sensor data,
then turns the prediction into maintenance alerts and a simple cost simulation.

Usage: python src/train.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
# Sensors with (near) zero variance in FD001 carry no information.
DROP = ["op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]
RUL_CAP = 125          # piece-wise linear target: engines are "healthy" early on
WINDOWS = (5, 15, 30)  # rolling windows (cycles)
ALERT_RUL = 30         # alert when predicted RUL drops below this many cycles


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.txt", sep=r"\s+", header=None, names=COLS)
    return df.drop(columns=DROP)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    sensors = [c for c in df.columns if c.startswith("s")]
    g = df.groupby("unit")
    feats = [df]
    for w in WINDOWS:
        roll = g[sensors].rolling(w, min_periods=1)
        m = roll.mean().reset_index(level=0, drop=True).add_suffix(f"_m{w}")
        s = roll.std().reset_index(level=0, drop=True).fillna(0).add_suffix(f"_sd{w}")
        feats += [m, s]
    # Trend: difference between short and long window means
    out = pd.concat(feats, axis=1)
    for c in sensors:
        out[f"{c}_trend"] = out[f"{c}_m5"] - out[f"{c}_m30"]
    return out


def nasa_score(y_true, y_pred):
    """Asymmetric score from the PHM08 challenge: late predictions cost more."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def main():
    train, test = load("train_FD001"), load("test_FD001")
    rul_true = pd.read_csv(DATA / "RUL_FD001.txt", header=None, names=["rul"])["rul"].values

    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = (max_cycle - train["cycle"]).clip(upper=RUL_CAP)

    trf, tef = add_features(train), add_features(test)
    features = [c for c in trf.columns if c not in ("unit", "rul")]

    # Hold out 20 engines for validation (split by engine, never by row)
    rng = np.random.default_rng(42)
    val_units = rng.choice(trf["unit"].unique(), 20, replace=False)
    is_val = trf["unit"].isin(val_units)

    params = dict(objective="regression", learning_rate=0.03, num_leaves=31,
                  min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
    dtr = lgb.Dataset(trf.loc[~is_val, features], trf.loc[~is_val, "rul"])
    dva = lgb.Dataset(trf.loc[is_val, features], trf.loc[is_val, "rul"])
    model = lgb.train(params, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = model.best_iteration
    # Refit on all engines with the chosen number of trees
    model = lgb.train(params, lgb.Dataset(trf[features], trf["rul"]), best_iter)

    # --- Test set: prediction at each engine's last observed cycle ---
    last = tef.groupby("unit").tail(1)
    pred = np.clip(model.predict(last[features]), 0, None)
    y = np.minimum(rul_true, RUL_CAP)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    score = nasa_score(y, pred)

    # Baseline: always predict the training mean RUL
    baseline = np.full_like(pred, trf["rul"].mean())
    rmse_base = float(np.sqrt(mean_squared_error(y, baseline)))

    # --- Alerts: does the model flag engines that fail within ALERT_RUL cycles? ---
    actual_alert = rul_true <= ALERT_RUL
    pred_alert = pred <= ALERT_RUL
    tp = int(np.sum(actual_alert & pred_alert))
    fp = int(np.sum(~actual_alert & pred_alert))
    fn = int(np.sum(actual_alert & ~pred_alert))
    tn = int(np.sum(~actual_alert & ~pred_alert))

    # --- Illustrative cost simulation (assumptions, not client data) ---
    COST_FAILURE = 50_000   # unplanned failure: downtime + emergency repair
    COST_PLANNED = 10_000   # planned intervention
    reactive = int(actual_alert.sum()) * COST_FAILURE
    predictive = tp * COST_PLANNED + fn * COST_FAILURE + fp * COST_PLANNED
    saving = reactive - predictive

    # --- Per-engine trajectories on validation engines for the dashboard ---
    val_pred = model.predict(trf.loc[is_val, features])
    traj = trf.loc[is_val, ["unit", "cycle", "rul"]].assign(pred=np.clip(val_pred, 0, None))
    true_rul_uncapped = (trf.loc[is_val].groupby("unit")["cycle"].transform("max")
                         - trf.loc[is_val, "cycle"])
    traj["rul_true"] = true_rul_uncapped.values
    trajectories = {
        int(u): {"cycle": d["cycle"].tolist(),
                 "rul_true": d["rul_true"].tolist(),
                 "rul_capped": d["rul"].tolist(),
                 "pred": [round(v, 1) for v in d["pred"]]}
        for u, d in traj.groupby("unit")
    }

    # Sensor degradation example (one engine, raw sensors)
    eng = train[train["unit"] == int(val_units[0])]
    sensor_example = {"unit": int(val_units[0]), "cycle": eng["cycle"].tolist(),
                      **{s: eng[s].round(3).tolist() for s in ["s2", "s4", "s11", "s15"]}}

    imp = pd.Series(model.feature_importance("gain"), index=features)
    by_sensor = imp.groupby(lambda c: c.split("_")[0]).sum().sort_values(ascending=False)
    importance = (by_sensor / by_sensor.sum()).round(4).head(10).to_dict()

    results = {
        "dataset": "NASA C-MAPSS FD001 (100 train / 100 test engines)",
        "metrics": {"rmse": round(rmse, 2), "mae": round(mae, 2),
                    "nasa_score": round(score, 1), "rmse_baseline": round(rmse_base, 2),
                    "improvement_vs_baseline_pct": round(100 * (1 - rmse / rmse_base), 1)},
        "alerts": {"threshold_cycles": ALERT_RUL, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   "recall": round(tp / max(tp + fn, 1), 3),
                   "precision": round(tp / max(tp + fp, 1), 3)},
        "cost_simulation": {"assumptions": {"cost_unplanned_failure_eur": COST_FAILURE,
                                            "cost_planned_intervention_eur": COST_PLANNED},
                            "reactive_eur": reactive, "predictive_eur": predictive,
                            "saving_eur": saving,
                            "saving_pct": round(100 * saving / max(reactive, 1), 1)},
        "test_predictions": {"true": [int(v) for v in rul_true],
                             "pred": [round(float(v), 1) for v in pred]},
        "trajectories": trajectories,
        "sensor_example": sensor_example,
        "importance": importance,
        "best_iteration": int(best_iter),
    }
    (OUT / "results.json").write_text(json.dumps(results))
    print(json.dumps({k: results[k] for k in ("metrics", "alerts", "cost_simulation")}, indent=2))


if __name__ == "__main__":
    main()
