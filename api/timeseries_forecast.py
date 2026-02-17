
# ------------------------------------------------------------
# this file contains helper functions for time-series forecasting
# using Random Forest and LSTM.
# it also has small utility functions for cleaning data,
# splitting datasets, and drawing result plots.
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
from io import BytesIO

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# pytorch for lstm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# defaults
DEFAULT_FORECAST_STEPS = 6
DEFAULT_RF_LAGS = 4
DEFAULT_RF_TREES = 200
SPLIT_RATIO = 0.80
MIN_TEST_POINTS = 2

LEARNING_RATE = 0.001
DEFAULT_LSTM_UNITS = 32
DEFAULT_LSTM_ACT = "tanh"  

np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# optional labels 
LABELS = {
    "R&D_MUSD":        "R&D Expenditure",
    "Enrollment":      "Enrollment",
    "Employees":       "Employees",
    "Total_Graduates": "Total Graduates",
}


# tiny metrics/cleaning helpers
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = y_true != 0
    if m.sum() == 0: return np.nan
    return np.mean(np.abs((y_true[m]-y_pred[m]) / y_true[m])) * 100

def metric_row(name, y_true, y_pred):
    return {
        "Model": name,
        "R²": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred)**0.5,
        "MAPE %": mape(y_true, y_pred),
    }

def split_index(n, split=SPLIT_RATIO, min_test=MIN_TEST_POINTS, min_train=None, rf_lags=DEFAULT_RF_LAGS):
    if min_train is None: min_train = rf_lags + 3
    si = max(int(n * split), min_train)
    if n - si < min_test: si = n - min_test
    return max(1, min(si, n - 1))

def clean_numeric(df: pd.DataFrame):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all").sort_index().interpolate().ffill().bfill()
    return out

def coerce_dt_index(df: pd.DataFrame, date_col: Optional[str]=None):
    cand = date_col
    if cand is None:
        for c in ["Date","date","Year","year","Time","time","Timestamp","timestamp","Date/Time"]:
            if c in df.columns:
                cand = c; break
    if cand is None: raise KeyError("No date/time column found.")
    s = pd.to_datetime(df[cand], errors="coerce")
    out = df.loc[~s.isna()].copy()
    out.index = s[~s.isna()]
    out = out.sort_index()
    if cand in out.columns: out = out.drop(columns=[cand])
    return out, cand

def infer_freq(idx: pd.DatetimeIndex):
    if len(idx) < 3: return "YS"
    d = pd.Series(idx[1:] - idx[:-1]).dt.days
    m = int(d.median())
    if m >= 300: return "YS"
    if 25 <= m <= 35: return "MS"
    return "D"

def future_index(hist: pd.DatetimeIndex, steps: int):
    f = infer_freq(hist); last = hist.max()
    if f == "YS":
        start = pd.Timestamp(year=last.year + 1, month=1, day=1)
        return pd.date_range(start=start, periods=steps, freq="YS")
    if f == "MS":
        start = (last + pd.offsets.MonthBegin()).normalize()
        return pd.date_range(start=start, periods=steps, freq="MS")
    return pd.date_range(start=last + pd.Timedelta(days=1), periods=steps, freq="D")

def fmt_display_idx(idx: pd.DatetimeIndex):
    f = infer_freq(idx)
    if f == "YS":  return pd.Index(idx.strftime("%Y"), name="Date")
    if f == "MS":  return pd.Index(idx.strftime("%Y-%m"), name="Date")
    return pd.Index(idx.strftime("%Y-%m-%d"), name="Date")

# generic ML helpers you reuse in UI
def enc_scale_fit(X: pd.DataFrame):
    enc = {}
    Xp = X.copy()
    for c in Xp.columns:
        if (Xp[c].dtype == "O") or (not pd.api.types.is_numeric_dtype(Xp[c])):
            s = Xp[c].astype(str).fillna("")
            cls = pd.Index(sorted(s.unique()))
            mp = {v:i for i,v in enumerate(cls)}
            Xp[c] = s.map(mp).fillna(-1).astype(int)
            enc[c] = {"classes": cls, "map": mp}
    sc = StandardScaler()
    Xp[Xp.columns] = sc.fit_transform(Xp[Xp.columns])
    return Xp, enc, sc

def enc_scale_apply(X: pd.DataFrame, enc, sc):
    Xp = X.copy()
    for c,info in enc.items():
        mp = info["map"]
        if c in Xp.columns: s = Xp[c].astype(str).fillna("")
        else: s = pd.Series([""]*len(Xp), index=Xp.index)
        Xp[c] = s.map(lambda v: mp.get(v, -1)).astype(int)
    for c in sc.feature_names_in_:
        if c not in Xp.columns: Xp[c] = 0
    Xp = Xp[sc.feature_names_in_]
    Xp[Xp.columns] = sc.transform(Xp[Xp.columns])
    return Xp

def _safe_split(X, y, is_clf, test_size=0.2, seed=42):
    strat = None
    if is_clf and y.nunique() >= 2:
        cnt = y.value_counts(dropna=False)
        if cnt.min() >= 2: strat = y
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)


# PyTorch LSTM utilities

class SeqModel(nn.Module):
    # simple seq2one LSTM -> Dense(1)
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last step
        out = self.drop(out)
        return self.fc(out)          # (B, 1)

def _make_sequences(y_scaled: np.ndarray, X_scaled: np.ndarray, lags: int):
    # build sliding windows of length lags
    Xseq, yseq = [], []
    for i in range(len(y_scaled) - lags):
        Xseq.append(np.concatenate([y_scaled[i:i+lags], X_scaled[i:i+lags]], axis=1))
        yseq.append(y_scaled[i+lags, 0])
    if len(Xseq) == 0:
        return None, None
    Xseq = np.asarray(Xseq, dtype=np.float32)  # (N, T, F)
    yseq = np.asarray(yseq, dtype=np.float32).reshape(-1, 1)
    return Xseq, yseq

def _train_lstm(Xtr_s, ytr_s, lags: int, hidden: int, max_epochs: int = 200, patience: int = 12):
    # Xtr_s: (N,T,F), ytr_s: (N,1)
    ds = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr_s))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = SeqModel(in_dim=Xtr_s.shape[2], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    bad = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds)

        if ep_loss < best_loss - 1e-6:
            best_loss = ep_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


# validation: rf +  lstm

def validate_series(series: pd.Series, label_for_rounding: str, df_all: pd.DataFrame,
                    rf_lags: int, rf_trees: int, lstm_units: int, lstm_act_unused: str):
    s = series.sort_index().astype(float).interpolate().ffill().bfill()
    if len(s) < max(rf_lags + 3, 8): return None

    cov = df_all.drop(columns=[series.name], errors="ignore")
    cov = clean_numeric(cov).reindex(s.index).ffill().bfill()

    si = split_index(len(s), rf_lags=rf_lags)
    train, test = s.iloc[:si], s.iloc[si:]
    y_true = test.values
    rows = []

    # naive baseline
    rows.append(metric_row("Naive (last)", y_true, np.repeat(train.iloc[-1], len(test))))

    # random forest rolling validation
    lag = pd.DataFrame({"y": s})
    for i in range(1, rf_lags + 1): lag[f"lag_{i}"] = lag["y"].shift(i)
    X_full = pd.concat([lag, cov], axis=1).dropna()
    if not X_full.empty:
        X_tr = X_full.iloc[:si].drop(columns=["y"]).values
        y_tr = X_full.iloc[:si]["y"].values
        rf = RandomForestRegressor(n_estimators=rf_trees, random_state=42).fit(X_tr, y_tr)

        preds, hist = [], s.iloc[:si].copy()
        idx_te = s.index[si:]
        for t in range(len(idx_te)):
            lag_vals = hist.iloc[-rf_lags:].values
            cov_vals = cov.loc[idx_te[t]].values if not cov.empty else np.zeros((0,))
            x = np.concatenate([lag_vals, cov_vals]).reshape(1, -1)
            y_hat = rf.predict(x)[0]
            preds.append(y_hat)
            hist = pd.concat([hist, pd.Series([y_hat], index=[idx_te[t]])])
        rows.append(metric_row("Random Forest", y_true, np.array(preds)))

    # lstm validation (seq2one on covariates + target)
    if not cov.empty:
        cov_cols = list(cov.columns)
        sy = MinMaxScaler().fit(train.values.reshape(-1,1))
        sx = MinMaxScaler().fit(cov.loc[train.index, cov_cols].values)
        y_tr_s = sy.transform(train.values.reshape(-1,1))
        X_tr_s = sx.transform(cov.loc[train.index, cov_cols].values)

        Xseq, yseq = _make_sequences(y_tr_s, X_tr_s, rf_lags)
        if Xseq is not None:
            model = _train_lstm(Xseq, yseq, lags=rf_lags, hidden=lstm_units)

            # roll over test period using known covariates at each step
            X_te_s = sx.transform(cov.loc[test.index, cov_cols].values)
            win = np.concatenate([y_tr_s, X_tr_s], axis=1)[-rf_lags:, :].copy()
            lstm_preds = []
            for k in range(len(test)):
                xb = torch.from_numpy(win.reshape(1, rf_lags, -1)).float().to(DEVICE)
                with torch.no_grad():
                    nxt_s = float(model(xb).cpu().numpy().squeeze())
                nxt_row = np.concatenate([[nxt_s], X_te_s[k]]).astype(float)
                win = np.vstack([win[1:], nxt_row])
                y_inv = sy.inverse_transform([[nxt_s]]).ravel()[0]
                lstm_preds.append(y_inv)

            rows.append(metric_row("LSTM", y_true, np.array(lstm_preds)))

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# forecasting: rf + lstm

def predict_series(series: pd.Series, label_for_rounding: str, df_all: pd.DataFrame,
                   steps: int, rf_lags: int, rf_trees: int, lstm_units: int, lstm_act_unused: str):
    s = series.sort_index().astype(float).interpolate().ffill().bfill()
    cov = df_all.drop(columns=[series.name], errors="ignore")
    cov = clean_numeric(cov).reindex(s.index).ffill().bfill()
    fut_idx = future_index(s.index, steps)

    # rf one-step recursive
    lag = pd.DataFrame({"y": s})
    for i in range(1, rf_lags + 1): lag[f"lag_{i}"] = lag["y"].shift(i)
    X_full = pd.concat([lag, cov], axis=1).dropna()
    X = X_full.drop(columns=["y"]).values
    y = X_full["y"].values
    rf = RandomForestRegressor(n_estimators=rf_trees, random_state=42).fit(X, y)

    hist = s.copy()
    last_cov = cov.iloc[-1].values if not cov.empty else np.zeros((0,))
    rf_fore = []
    for _ in range(steps):
        last_lags = hist.iloc[-rf_lags:].values
        x = np.concatenate([last_lags, last_cov]).reshape(1, -1)
        y_hat = rf.predict(x)[0]
        rf_fore.append(y_hat)
        hist.loc[fut_idx[len(rf_fore)-1]] = y_hat

    # lstm multi-step with fixed covariates (last known)
    lstm_fore = None
    if not cov.empty:
        cov_cols = list(cov.columns)
        sy = MinMaxScaler().fit(s.values.reshape(-1,1))
        sx = MinMaxScaler().fit(cov.loc[s.index, cov_cols].values)
        y_all_s = sy.transform(s.values.reshape(-1,1))
        X_all_s = sx.transform(cov.loc[s.index, cov_cols].values)

        Xseq, yseq = _make_sequences(y_all_s, X_all_s, rf_lags)
        if Xseq is not None:
            model = _train_lstm(Xseq, yseq, lags=rf_lags, hidden=lstm_units)

            last_cov_s = sx.transform(cov.loc[[s.index[-1]], cov_cols].values)[0]
            win = np.concatenate([y_all_s, X_all_s], axis=1)[-rf_lags:, :].copy()
            preds_s = []
            for _ in range(steps):
                xb = torch.from_numpy(win.reshape(1, rf_lags, -1)).float().to(DEVICE)
                with torch.no_grad():
                    nxt_s = float(model(xb).cpu().numpy().squeeze())
                nxt_row = np.concatenate([[nxt_s], last_cov_s]).astype(float)
                win = np.vstack([win[1:], nxt_row])
                preds_s.append(nxt_s)
            lstm_raw = sy.inverse_transform(np.array(preds_s).reshape(-1,1)).ravel()
            lstm_fore = np.array(lstm_raw)

    return np.array(rf_fore), (None if lstm_fore is None else np.array(lstm_fore)), fut_idx


# plotting

def plot_single(series, label, rf_fore, lstm_fore, fut_idx, title=""):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, "o-", label="Actual")
    ax.plot([series.index[-1]] + list(fut_idx), [series.values[-1]] + list(rf_fore), "--x", label="RF")
    if lstm_fore is not None:
        ax.plot([series.index[-1]] + list(fut_idx), [series.values[-1]] + list(lstm_fore), ":o", label="LSTM")
    ax.set_title(title or label)
    ax.set_xlabel("Date")
    ax.set_ylabel(label)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True); ax.legend(ncol=3); fig.tight_layout()
    return fig

