#!/usr/bin/env python3
"""
Final hierarchical execution strategy.

This script extracts the notebook's final strategy into a standalone pipeline:

1. Build order-book alpha features from train/test CSV files.
2. Select alpha features from training data only.
3. Fit separate BUY and SELL Ridge models for each stock.
4. Tune execution thresholds on an internal validation split.
5. Execute the final Hierarchical Ridge + VWOF filter strategy.
6. Report the required improvement metric versus TWAP on train and test data.

Default expected file names:
    Project_Train_Datasets/AMZN_5levels_train.csv
    Project_Test_Datasets/AMZN_5levels_test.csv

The AAPL strategy is fully autonomous: if an AAPL/APPL train file is present,
the same training, feature selection, and threshold tuning process is applied.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_STOCKS = ("AMZN", "GOOG", "INTC", "MSFT", "AAPL")

CANDIDATE_ALPHA_COLS = [
    "bid1",
    "spread",
    "bsz1",
    "asz1",
    "depth_value_1",
    "imbalance_1",
    "micro_minus_mid",
    "total_bid_size_5",
    "imbalance_5",
    "mid_chg1",
    "spread_chg1",
    "imbalance_1_chg1",
    "imbalance_5_chg1",
    "micro_minus_mid_chg1",
    "bid_liq_1_chg1",
    "ask_liq_1_chg1",
    "total_bid_size_5_chg1",
    "total_ask_size_5_chg1",
    "sec_in_min",
    "event_idx_in_min",
    "n_events_in_min",
    "event_frac_in_min",
]

T_STAT_CUTOFF = 2.0
P_VALUE_CUTOFF = 0.05
SPEARMAN_CUTOFF = 0.01
MIN_FEATURES_PER_MODEL = 5
TOP_K_FALLBACK = 8
THRESHOLD_QUANTILE_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


@dataclass
class StockStrategy:
    stock: str
    buy_model: Pipeline
    sell_model: Pipeline
    buy_features: List[str]
    sell_features: List[str]
    buy_threshold_quantile: float
    sell_threshold_quantile: float
    buy_threshold: float
    sell_threshold: float
    validation_improvement: float


def canonical_stock(stock: str) -> str:
    stock = stock.upper().strip()
    return "AAPL" if stock == "APPL" else stock


def aliases_for_stock(stock: str) -> List[str]:
    stock = canonical_stock(stock)
    aliases = [stock]
    if stock == "AAPL":
        aliases.append("APPL")
    return aliases


def parse_stock_list(raw: Sequence[str]) -> List[str]:
    stocks: List[str] = []
    for item in raw:
        for token in item.replace(",", " ").split():
            stock = canonical_stock(token)
            if stock and stock not in stocks:
                stocks.append(stock)
    return stocks


def find_stock_file(data_dir: Path, stock: str, split: str) -> Optional[Path]:
    if data_dir is None or not data_dir.exists():
        return None

    split = split.lower()
    exact_names: List[str] = []
    for alias in aliases_for_stock(stock):
        exact_names.extend(
            [
                f"{alias}_5levels_{split}.csv",
                f"{alias}_{split}.csv",
                f"{alias.upper()}_5levels_{split}.csv",
                f"{alias.upper()}_{split}.csv",
                f"{alias.lower()}_5levels_{split}.csv",
                f"{alias.lower()}_{split}.csv",
            ]
        )

    for name in exact_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate

    csv_files = sorted(data_dir.glob("*.csv"))
    for alias in aliases_for_stock(stock):
        alias_lower = alias.lower()
        matches = [
            path
            for path in csv_files
            if alias_lower in path.name.lower() and split in path.name.lower()
        ]
        if matches:
            return matches[0]

    return None


def require_columns(df: pd.DataFrame, required: Iterable[str], path_hint: str = "") -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        suffix = f" in {path_hint}" if path_hint else ""
        raise ValueError(f"Missing required columns{suffix}: {missing}")


def parse_time_column(time_series: pd.Series) -> pd.Series:
    text = time_series.astype(str)
    ts = pd.to_datetime(text, format="%H:%M:%S.%f", errors="coerce")
    missing = ts.isna()
    if missing.any():
        ts.loc[missing] = pd.to_datetime(text.loc[missing], errors="coerce")
    return ts


def build_orderbook_features(raw_df: pd.DataFrame, stock: str, path_hint: str = "") -> pd.DataFrame:
    required = [
        "Time",
        "BidPrice_1",
        "AskPrice_1",
        "BidSize_1",
        "AskSize_1",
        "Direction_1=Buy_-1=Sell",
        "Size",
        "NewLimitOrder_1=Yes_0=No",
        "PartialCancel_1=Yes_0=No",
        "FullDelete_1=Yes_0=No",
        "VisibleExecution_1=Yes_0=No",
        "HiddenExecution_1=Yes_0=No",
    ]
    required.extend([f"BidSize_{i}" for i in range(1, 6)])
    required.extend([f"AskSize_{i}" for i in range(1, 6)])
    require_columns(raw_df, required, path_hint)

    df = raw_df.copy()
    df["ts"] = parse_time_column(df["Time"])
    df = df.dropna(subset=["ts"]).copy()
    df = df.sort_values("ts").reset_index(drop=True)

    df["stock"] = canonical_stock(stock)
    df["minute"] = df["ts"].dt.floor("min")
    df["sec_in_min"] = df["ts"].dt.second + df["ts"].dt.microsecond / 1_000_000

    df["bid1"] = df["BidPrice_1"]
    df["ask1"] = df["AskPrice_1"]
    df["bsz1"] = df["BidSize_1"]
    df["asz1"] = df["AskSize_1"]
    df["mid"] = (df["bid1"] + df["ask1"]) / 2
    df["spread"] = df["ask1"] - df["bid1"]

    df["bid_liq_1"] = df["bid1"] * df["bsz1"]
    df["ask_liq_1"] = df["ask1"] * df["asz1"]
    df["depth_value_1"] = df["bid_liq_1"] + df["ask_liq_1"]

    denom1 = df["bsz1"] + df["asz1"]
    df["imbalance_1"] = np.where(denom1 > 0, (df["bsz1"] - df["asz1"]) / denom1, 0.0)
    df["microprice"] = np.where(
        denom1 > 0,
        (df["ask1"] * df["bsz1"] + df["bid1"] * df["asz1"]) / denom1,
        df["mid"],
    )
    df["micro_minus_mid"] = df["microprice"] - df["mid"]

    bid_size_cols = [f"BidSize_{i}" for i in range(1, 6)]
    ask_size_cols = [f"AskSize_{i}" for i in range(1, 6)]
    df["total_bid_size_5"] = df[bid_size_cols].sum(axis=1)
    df["total_ask_size_5"] = df[ask_size_cols].sum(axis=1)

    total_depth_5 = df["total_bid_size_5"] + df["total_ask_size_5"]
    df["imbalance_5"] = np.where(
        total_depth_5 > 0,
        (df["total_bid_size_5"] - df["total_ask_size_5"]) / total_depth_5,
        0.0,
    )

    change_cols = [
        "mid",
        "spread",
        "imbalance_1",
        "imbalance_5",
        "micro_minus_mid",
        "bid_liq_1",
        "ask_liq_1",
        "total_bid_size_5",
        "total_ask_size_5",
    ]
    for col in change_cols:
        df[f"{col}_chg1"] = df[col].diff().fillna(0)

    df["event_idx_in_min"] = df.groupby("minute").cumcount()
    df["n_events_in_min"] = df.groupby("minute")["minute"].transform("size")
    df["event_frac_in_min"] = np.where(
        df["n_events_in_min"] > 1,
        df["event_idx_in_min"] / (df["n_events_in_min"] - 1),
        0.0,
    )

    df["future_min_ask_in_min"] = df.groupby("minute")["ask1"].transform(
        lambda s: s.iloc[::-1].cummin().iloc[::-1]
    )
    df["buy_regret"] = df["ask1"] - df["future_min_ask_in_min"]

    df["future_max_bid_in_min"] = df.groupby("minute")["bid1"].transform(
        lambda s: s.iloc[::-1].cummax().iloc[::-1]
    )
    df["sell_regret"] = df["future_max_bid_in_min"] - df["bid1"]

    return df.replace([np.inf, -np.inf], np.nan)


def load_feature_frame(path: Path, stock: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return build_orderbook_features(raw, stock, str(path))


def add_fit_tune_split(df: pd.DataFrame, train_frac: float = 0.70) -> pd.DataFrame:
    df = df.copy()
    df["internal_sample"] = "fit"

    for stock in sorted(df["stock"].unique()):
        stock_minutes = np.array(sorted(df.loc[df["stock"] == stock, "minute"].unique()))
        split_idx = max(1, int(len(stock_minutes) * train_frac))
        if split_idx >= len(stock_minutes):
            split_idx = max(1, len(stock_minutes) - 1)

        fit_minutes = set(stock_minutes[:split_idx])
        tune_minutes = set(stock_minutes[split_idx:])

        df.loc[(df["stock"] == stock) & (df["minute"].isin(fit_minutes)), "internal_sample"] = "fit"
        df.loc[(df["stock"] == stock) & (df["minute"].isin(tune_minutes)), "internal_sample"] = "tune"

    return df


def split_train_test_by_minute(
    df: pd.DataFrame,
    train_frac: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    train_parts = []
    test_parts = []

    for stock in sorted(df["stock"].unique()):
        stock_df = df[df["stock"] == stock].copy()
        stock_minutes = np.array(sorted(stock_df["minute"].unique()))

        if len(stock_minutes) < 2:
            raise ValueError(f"{stock}: need at least two minutes to split train/test.")

        split_idx = max(1, int(len(stock_minutes) * train_frac))
        if split_idx >= len(stock_minutes):
            split_idx = len(stock_minutes) - 1

        train_minutes = set(stock_minutes[:split_idx])
        test_minutes = set(stock_minutes[split_idx:])

        train_parts.append(stock_df[stock_df["minute"].isin(train_minutes)].copy())
        test_parts.append(stock_df[stock_df["minute"].isin(test_minutes)].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sort_values(["stock", "minute", "ts"])
    test_df = pd.concat(test_parts, ignore_index=True).sort_values(["stock", "minute", "ts"])

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def validate_alpha_against_regret(
    data: pd.DataFrame,
    alpha_col: str,
    target_col: str,
    stock: str,
) -> Dict[str, object]:
    tmp = (
        data[[alpha_col, target_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .rename(columns={alpha_col: "alpha", target_col: "target"})
    )

    side = "buy" if target_col == "buy_regret" else "sell"
    base = {
        "stock": stock,
        "side": side,
        "alpha": alpha_col,
        "target": target_col,
        "n": len(tmp),
        "pearson": np.nan,
        "spearman": np.nan,
        "slope": np.nan,
        "t_slope": np.nan,
        "p_slope": np.nan,
        "bucket_spread_top_minus_bottom": np.nan,
        "abs_t_slope": np.nan,
        "abs_spearman": np.nan,
        "abs_bucket_spread": np.nan,
        "note": "",
    }

    if len(tmp) < 100 or tmp["alpha"].nunique() < 3:
        base["note"] = "too few observations or too few unique values"
        return base

    try:
        base["pearson"] = tmp["alpha"].corr(tmp["target"])
        base["spearman"] = tmp["alpha"].corr(tmp["target"], method="spearman")
    except Exception as exc:  # pragma: no cover - defensive for unusual data
        base["note"] = f"correlation failed: {exc}"

    try:
        slope, _, _, p_value, std_err = stats.linregress(tmp["alpha"], tmp["target"])
        t_slope = slope / std_err if std_err is not None and std_err > 0 else np.nan
        base["slope"] = slope
        base["t_slope"] = t_slope
        base["p_slope"] = p_value
    except Exception as exc:
        base["note"] = f"regression failed: {exc}"

    try:
        buckets = pd.qcut(tmp["alpha"], q=10, labels=False, duplicates="drop")
        bucket_means = tmp.assign(bucket=buckets).groupby("bucket")["target"].mean()
        if len(bucket_means) >= 2:
            base["bucket_spread_top_minus_bottom"] = bucket_means.iloc[-1] - bucket_means.iloc[0]
    except Exception:
        pass

    base["abs_t_slope"] = abs(base["t_slope"]) if pd.notna(base["t_slope"]) else np.nan
    base["abs_spearman"] = abs(base["spearman"]) if pd.notna(base["spearman"]) else np.nan
    base["abs_bucket_spread"] = (
        abs(base["bucket_spread_top_minus_bottom"])
        if pd.notna(base["bucket_spread_top_minus_bottom"])
        else np.nan
    )

    return base


def select_features(fit_df: pd.DataFrame, stock: str) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    candidate_cols = [col for col in CANDIDATE_ALPHA_COLS if col in fit_df.columns]
    summaries: List[Dict[str, object]] = []

    for target_col in ("buy_regret", "sell_regret"):
        for alpha_col in candidate_cols:
            summaries.append(validate_alpha_against_regret(fit_df, alpha_col, target_col, stock))

    alpha_summary = pd.DataFrame(summaries)
    strong_alpha_df = alpha_summary[
        (alpha_summary["p_slope"] <= P_VALUE_CUTOFF)
        & (alpha_summary["abs_t_slope"] >= T_STAT_CUTOFF)
        & (alpha_summary["abs_spearman"] >= SPEARMAN_CUTOFF)
    ].copy()

    selected: Dict[str, List[str]] = {}
    for side in ("buy", "sell"):
        strong = strong_alpha_df[(strong_alpha_df["stock"] == stock) & (strong_alpha_df["side"] == side)]
        features = strong["alpha"].drop_duplicates().tolist()

        if len(features) < MIN_FEATURES_PER_MODEL:
            fallback = (
                alpha_summary[
                    (alpha_summary["stock"] == stock)
                    & (alpha_summary["side"] == side)
                    & (alpha_summary["note"] == "")
                ]
                .sort_values("abs_t_slope", ascending=False)
                .head(TOP_K_FALLBACK)
            )
            features = fallback["alpha"].drop_duplicates().tolist()

        if not features:
            features = candidate_cols[: min(MIN_FEATURES_PER_MODEL, len(candidate_cols))]

        selected[side] = features

    return selected, alpha_summary


def fit_side_model(df: pd.DataFrame, features: List[str], target_col: str) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].replace([np.inf, -np.inf], np.nan).fillna(0)
    model.fit(X, y)
    return model


def fit_models(df: pd.DataFrame, selected_features: Dict[str, List[str]]) -> Tuple[Pipeline, Pipeline]:
    buy_model = fit_side_model(df, selected_features["buy"], "buy_regret")
    sell_model = fit_side_model(df, selected_features["sell"], "sell_regret")
    return buy_model, sell_model


def add_model_predictions(
    df: pd.DataFrame,
    buy_model: Pipeline,
    sell_model: Pipeline,
    buy_features: List[str],
    sell_features: List[str],
) -> pd.DataFrame:
    df = df.copy()
    X_buy = df[buy_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_sell = df[sell_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["pred_buy_regret"] = buy_model.predict(X_buy)
    df["pred_sell_regret"] = sell_model.predict(X_sell)
    return df


def add_hierarchical_filter_signals(df: pd.DataFrame) -> pd.DataFrame:
    signal_frames = []

    for stock, stock_df in df.groupby("stock", sort=False):
        s = stock_df.sort_values("ts").reset_index(drop=True).copy()

        direction = s["Direction_1=Buy_-1=Sell"].values
        size = s["Size"].values
        is_new = s["NewLimitOrder_1=Yes_0=No"].values
        is_cancel = (
            s["PartialCancel_1=Yes_0=No"].astype(bool)
            | s["FullDelete_1=Yes_0=No"].astype(bool)
        ).values
        is_exec = (
            s["VisibleExecution_1=Yes_0=No"].astype(bool)
            | s["HiddenExecution_1=Yes_0=No"].astype(bool)
        ).values

        flow = np.zeros(len(s))
        flow = np.where(is_new == 1, direction * size, flow)
        flow = np.where(is_cancel == 1, -direction * size, flow)
        flow = np.where(is_exec == 1, -direction * size, flow)

        net_flow_50 = pd.Series(flow, index=s.index).rolling(50, min_periods=10).sum()
        total_vol_50 = pd.Series(np.abs(flow), index=s.index).rolling(50, min_periods=10).sum()

        s["H_VWOF_Raw"] = (net_flow_50 / (total_vol_50 + 1e-9)).fillna(0.0)
        s["H_VWOF"] = s["H_VWOF_Raw"].shift(1).fillna(0.0)

        s["H_MidPrice"] = (s["BidPrice_1"] + s["AskPrice_1"]) / 2
        s["H_MicroPrice"] = (
            s["BidPrice_1"] * s["AskSize_1"] + s["AskPrice_1"] * s["BidSize_1"]
        ) / (s["BidSize_1"] + s["AskSize_1"] + 1e-9)
        s["H_MicroMomentum"] = (s["H_MicroPrice"] - s["H_MidPrice"]).shift(1).fillna(0.0)

        s["H_Spread"] = s["AskPrice_1"] - s["BidPrice_1"]
        s["H_Depth_1"] = s["BidSize_1"] + s["AskSize_1"]

        lookback = 600
        min_obs = 50
        spread_hist = s["H_Spread"].rolling(lookback, min_periods=min_obs)
        vwof_abs_hist = s["H_VWOF"].abs().rolling(lookback, min_periods=min_obs)
        mom_abs_hist = s["H_MicroMomentum"].abs().rolling(lookback, min_periods=min_obs)
        depth_hist = s["H_Depth_1"].rolling(lookback, min_periods=min_obs)

        s["H_Spread_Limit"] = spread_hist.quantile(0.70).shift(1)
        s["H_Spread_Median"] = spread_hist.median().shift(1)
        s["H_VWOF_Guard"] = vwof_abs_hist.quantile(0.65).shift(1).clip(0.08, 0.35)
        s["H_Momentum_Guard"] = mom_abs_hist.quantile(0.60).shift(1)
        s["H_Depth_Median"] = depth_hist.median().shift(1)

        s["H_Spread_Limit"] = s["H_Spread_Limit"].ffill().fillna(s["H_Spread"])
        s["H_Spread_Median"] = s["H_Spread_Median"].ffill().fillna(s["H_Spread"])
        s["H_VWOF_Guard"] = s["H_VWOF_Guard"].ffill().fillna(0.20)
        s["H_Momentum_Guard"] = s["H_Momentum_Guard"].ffill().fillna(0.0)
        s["H_Depth_Median"] = s["H_Depth_Median"].ffill().fillna(s["H_Depth_1"])

        signal_frames.append(s)

    return pd.concat(signal_frames, ignore_index=True).sort_values(["stock", "minute", "ts"]).reset_index(drop=True)


def execute_twap(df: pd.DataFrame) -> pd.DataFrame:
    twap = df.sort_values(["stock", "minute", "ts"]).groupby(["stock", "minute"], as_index=False).first()
    cols = ["stock", "minute", "ts", "ask1", "bid1"]
    if "sample" in twap.columns:
        cols.append("sample")
    twap = twap[cols].rename(
        columns={
            "ts": "twap_time",
            "ask1": "twap_buy_price",
            "bid1": "twap_sell_price",
        }
    )
    return twap.sort_values(["stock", "minute"]).reset_index(drop=True)


def threshold_dict(strategy: StockStrategy) -> Dict[str, Dict[str, float]]:
    return {
        strategy.stock: {
            "buy_threshold": strategy.buy_threshold,
            "sell_threshold": strategy.sell_threshold,
        }
    }


def execute_strategy4_threshold(
    df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    fallback: str = "last",
) -> pd.DataFrame:
    trades = []
    df = df.sort_values(["stock", "minute", "ts"]).copy()

    for (stock, minute), g in df.groupby(["stock", "minute"], sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        buy_th = thresholds[stock]["buy_threshold"]
        sell_th = thresholds[stock]["sell_threshold"]

        buy_hits = g[g["pred_buy_regret"] <= buy_th]
        sell_hits = g[g["pred_sell_regret"] <= sell_th]

        if len(buy_hits) > 0:
            buy_row = buy_hits.iloc[0]
            buy_trigger = "threshold_hit"
        else:
            buy_row = g.iloc[-1] if fallback == "last" else g.iloc[0]
            buy_trigger = f"fallback_{fallback}"

        if len(sell_hits) > 0:
            sell_row = sell_hits.iloc[0]
            sell_trigger = "threshold_hit"
        else:
            sell_row = g.iloc[-1] if fallback == "last" else g.iloc[0]
            sell_trigger = f"fallback_{fallback}"

        sample = g["sample"].iloc[0] if "sample" in g.columns else "unknown"
        trades.append(
            {
                "stock": stock,
                "minute": minute,
                "sample": sample,
                "algo_buy_time": buy_row["ts"],
                "algo_sell_time": sell_row["ts"],
                "algo_buy_price": buy_row["ask1"],
                "algo_sell_price": sell_row["bid1"],
                "pred_buy_regret": buy_row["pred_buy_regret"],
                "pred_sell_regret": sell_row["pred_sell_regret"],
                "buy_threshold": buy_th,
                "sell_threshold": sell_th,
                "buy_trigger": buy_trigger,
                "sell_trigger": sell_trigger,
            }
        )

    return pd.DataFrame(trades).sort_values(["stock", "minute"]).reset_index(drop=True)


def execute_hierarchical_strategy(
    df: pd.DataFrame,
    thresholds: Dict[str, Dict[str, float]],
    fallback: str = "last",
) -> pd.DataFrame:
    trades = []
    df = df.sort_values(["stock", "minute", "ts"]).copy()

    for (stock, minute), g in df.groupby(["stock", "minute"], sort=False):
        g = g.sort_values("ts").reset_index(drop=True)
        buy_th = thresholds[stock]["buy_threshold"]
        sell_th = thresholds[stock]["sell_threshold"]
        sample = g["sample"].iloc[0] if "sample" in g.columns else "unknown"

        time_progress = ((g["ts"] - g["minute"]).dt.total_seconds() / 60.0).clip(0.0, 1.0)
        late = time_progress > 0.85
        spread_safe = g["H_Spread"] <= g["H_Spread_Limit"]

        buy_vwof_guard = np.where(late, 1.5 * g["H_VWOF_Guard"], g["H_VWOF_Guard"])
        buy_momentum_guard = np.where(late, 1.5 * g["H_Momentum_Guard"], g["H_Momentum_Guard"])
        buy_condition = (
            (g["pred_buy_regret"] <= buy_th)
            & spread_safe
            & (g["H_VWOF"] >= -buy_vwof_guard)
            & (g["H_MicroMomentum"] >= -buy_momentum_guard)
        )

        sell_vwof_guard = np.where(late, 1.5 * g["H_VWOF_Guard"], g["H_VWOF_Guard"])
        sell_momentum_guard = np.where(late, 1.5 * g["H_Momentum_Guard"], g["H_Momentum_Guard"])
        sell_condition = (
            (g["pred_sell_regret"] <= sell_th)
            & spread_safe
            & (g["H_VWOF"] <= sell_vwof_guard)
            & (g["H_MicroMomentum"] <= sell_momentum_guard)
        )

        if buy_condition.any():
            buy_row = g.loc[buy_condition[buy_condition].index[0]]
            buy_trigger = "ridge_signal_passed_vwof_filter"
        else:
            buy_row = g.iloc[-1] if fallback == "last" else g.iloc[0]
            buy_trigger = f"fallback_{fallback}"

        if sell_condition.any():
            sell_row = g.loc[sell_condition[sell_condition].index[0]]
            sell_trigger = "ridge_signal_passed_vwof_filter"
        else:
            sell_row = g.iloc[-1] if fallback == "last" else g.iloc[0]
            sell_trigger = f"fallback_{fallback}"

        trades.append(
            {
                "stock": stock,
                "minute": minute,
                "sample": sample,
                "algo_buy_time": buy_row["ts"],
                "algo_sell_time": sell_row["ts"],
                "algo_buy_price": buy_row["ask1"],
                "algo_sell_price": sell_row["bid1"],
                "pred_buy_regret": buy_row["pred_buy_regret"],
                "pred_sell_regret": sell_row["pred_sell_regret"],
                "buy_vwof": buy_row["H_VWOF"],
                "sell_vwof": sell_row["H_VWOF"],
                "buy_micro_momentum": buy_row["H_MicroMomentum"],
                "sell_micro_momentum": sell_row["H_MicroMomentum"],
                "buy_spread": buy_row["H_Spread"],
                "sell_spread": sell_row["H_Spread"],
                "buy_threshold": buy_th,
                "sell_threshold": sell_th,
                "buy_trigger": buy_trigger,
                "sell_trigger": sell_trigger,
            }
        )

    return pd.DataFrame(trades).sort_values(["stock", "minute"]).reset_index(drop=True)


def tune_strategy4_threshold_values(
    val_pred_df: pd.DataFrame,
    stock: str,
    quantile_grid: Sequence[float] = THRESHOLD_QUANTILE_GRID,
) -> Tuple[float, float, float, pd.DataFrame]:
    stock = canonical_stock(stock)
    twap_val = execute_twap(val_pred_df)

    buy_candidates = val_pred_df["pred_buy_regret"].quantile(quantile_grid).values
    sell_candidates = val_pred_df["pred_sell_regret"].quantile(quantile_grid).values

    best_score = -np.inf
    best_buy_th = float(buy_candidates[0])
    best_sell_th = float(sell_candidates[0])
    rows = []

    for buy_th in buy_candidates:
        for sell_th in sell_candidates:
            thresholds = {stock: {"buy_threshold": float(buy_th), "sell_threshold": float(sell_th)}}
            candidate_trades = execute_strategy4_threshold(val_pred_df, thresholds, fallback="last")
            metric = evaluate_required_metric(
                candidate_trades,
                twap_val,
                "Strategy 4 Alpha-Filtered Ridge",
                "val",
            )
            score = metric["PCT_IMPROVEMENT"]
            rows.append(
                {
                    "stock": stock,
                    "buy_threshold": float(buy_th),
                    "sell_threshold": float(sell_th),
                    "validation_improvement": score,
                    "buy_trigger_rate": (candidate_trades["buy_trigger"] == "threshold_hit").mean(),
                    "sell_trigger_rate": (candidate_trades["sell_trigger"] == "threshold_hit").mean(),
                }
            )

            if pd.notna(score) and score > best_score:
                best_score = float(score)
                best_buy_th = float(buy_th)
                best_sell_th = float(sell_th)

    return best_buy_th, best_sell_th, best_score, pd.DataFrame(rows)


def evaluate_required_metric(
    algo_trades: pd.DataFrame,
    twap_trades: pd.DataFrame,
    strategy_name: str,
    sample: str,
) -> Dict[str, object]:
    merge_cols = ["stock", "minute"]
    if "sample" in algo_trades.columns and "sample" in twap_trades.columns:
        merge_cols.append("sample")

    merged = algo_trades.merge(twap_trades, on=merge_cols, how="inner")
    total_algo_buy = merged["algo_buy_price"].sum()
    total_algo_sell = merged["algo_sell_price"].sum()
    total_twap_buy = merged["twap_buy_price"].sum()
    total_twap_sell = merged["twap_sell_price"].sum()

    algo_cost = total_algo_buy - total_algo_sell
    twap_cost = total_twap_buy - total_twap_sell
    pct_improvement = np.nan if twap_cost == 0 else 100 - 100 * algo_cost / twap_cost

    stock = merged["stock"].iloc[0] if len(merged) else "unknown"
    return {
        "strategy": strategy_name,
        "sample": sample,
        "stock": stock,
        "n_minutes": len(merged),
        "TOTAL_ALGO_BUY": total_algo_buy,
        "TOTAL_ALGO_SELL": total_algo_sell,
        "TOTAL_TWAP_BUY": total_twap_buy,
        "TOTAL_TWAP_SELL": total_twap_sell,
        "ALGO_COST": algo_cost,
        "TWAP_COST": twap_cost,
        "PCT_IMPROVEMENT": pct_improvement,
    }


def tune_threshold_quantiles(
    tune_signal_df: pd.DataFrame,
    stock: str,
    quantile_grid: Sequence[float] = THRESHOLD_QUANTILE_GRID,
    twap_tune: Optional[pd.DataFrame] = None,
) -> Tuple[float, float, float, pd.DataFrame]:
    stock = canonical_stock(stock)
    if twap_tune is None:
        twap_tune = execute_twap(tune_signal_df)

    buy_quantiles = tune_signal_df["pred_buy_regret"].quantile(quantile_grid)
    sell_quantiles = tune_signal_df["pred_sell_regret"].quantile(quantile_grid)

    best_score = -np.inf
    best_buy_q = float(quantile_grid[0])
    best_sell_q = float(quantile_grid[0])
    rows = []

    for buy_q, buy_th in buy_quantiles.items():
        for sell_q, sell_th in sell_quantiles.items():
            thresholds = {stock: {"buy_threshold": float(buy_th), "sell_threshold": float(sell_th)}}
            candidate_trades = execute_hierarchical_strategy(tune_signal_df, thresholds, fallback="last")
            metric = evaluate_required_metric(
                candidate_trades,
                twap_tune,
                "Hierarchical Ridge + VWOF Filter",
                "internal_tune",
            )
            score = metric["PCT_IMPROVEMENT"]
            rows.append(
                {
                    "stock": stock,
                    "buy_threshold_quantile": float(buy_q),
                    "sell_threshold_quantile": float(sell_q),
                    "buy_threshold": float(buy_th),
                    "sell_threshold": float(sell_th),
                    "validation_improvement": score,
                    "buy_hit_rate": (
                        candidate_trades["buy_trigger"] == "ridge_signal_passed_vwof_filter"
                    ).mean(),
                    "sell_hit_rate": (
                        candidate_trades["sell_trigger"] == "ridge_signal_passed_vwof_filter"
                    ).mean(),
                }
            )
            if pd.notna(score) and score > best_score:
                best_score = float(score)
                best_buy_q = float(buy_q)
                best_sell_q = float(sell_q)

    return best_buy_q, best_sell_q, best_score, pd.DataFrame(rows)


def quantile_threshold(series: pd.Series, q: float) -> float:
    value = series.replace([np.inf, -np.inf], np.nan).dropna().quantile(q)
    if pd.isna(value):
        value = series.replace([np.inf, -np.inf], np.nan).dropna().median()
    if pd.isna(value):
        value = 0.0
    return float(value)


def train_stock_strategy(
    train_features: pd.DataFrame,
    stock: str,
    train_frac: float,
) -> Tuple[StockStrategy, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock = canonical_stock(stock)
    split_df = add_fit_tune_split(train_features, train_frac=train_frac)
    fit_df = split_df[split_df["internal_sample"] == "fit"].copy()
    tune_df = split_df[split_df["internal_sample"] == "tune"].copy()

    selected_features, alpha_summary = select_features(fit_df, stock)

    calibration_buy_model, calibration_sell_model = fit_models(fit_df, selected_features)
    calibration_pred = add_model_predictions(
        split_df,
        calibration_buy_model,
        calibration_sell_model,
        selected_features["buy"],
        selected_features["sell"],
    )
    calibration_signal = add_hierarchical_filter_signals(calibration_pred)
    tune_signal = calibration_signal[calibration_signal["internal_sample"] == "tune"].copy()
    tune_pred = calibration_pred[calibration_pred["internal_sample"] == "tune"].copy()
    twap_tune = execute_twap(tune_pred)

    if len(tune_df) == 0 or tune_signal["minute"].nunique() == 0:
        warnings.warn(f"{stock}: no internal tuning rows; using default threshold quantiles.")
        best_buy_q, best_sell_q, best_score = 0.20, 0.20, np.nan
        threshold_search = pd.DataFrame()
    else:
        best_buy_q, best_sell_q, best_score, threshold_search = tune_threshold_quantiles(
            tune_signal,
            stock,
            twap_tune=twap_tune,
        )

    final_buy_model, final_sell_model = fit_models(train_features, selected_features)
    final_pred = add_model_predictions(
        train_features,
        final_buy_model,
        final_sell_model,
        selected_features["buy"],
        selected_features["sell"],
    )

    buy_threshold = quantile_threshold(final_pred["pred_buy_regret"], best_buy_q)
    sell_threshold = quantile_threshold(final_pred["pred_sell_regret"], best_sell_q)

    strategy = StockStrategy(
        stock=stock,
        buy_model=final_buy_model,
        sell_model=final_sell_model,
        buy_features=selected_features["buy"],
        sell_features=selected_features["sell"],
        buy_threshold_quantile=best_buy_q,
        sell_threshold_quantile=best_sell_q,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        validation_improvement=best_score,
    )

    selected_feature_rows = pd.DataFrame(
        [
            {
                "stock": stock,
                "side": side,
                "feature": feature,
                "rank": rank,
            }
            for side in ("buy", "sell")
            for rank, feature in enumerate(selected_features[side], start=1)
        ]
    )

    return strategy, alpha_summary, threshold_search, selected_feature_rows


def run_strategy_on_features(
    features: pd.DataFrame,
    strategy: StockStrategy,
    sample: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = add_model_predictions(
        features,
        strategy.buy_model,
        strategy.sell_model,
        strategy.buy_features,
        strategy.sell_features,
    )
    pred_df["sample"] = sample
    twap = execute_twap(pred_df)
    signal_df = add_hierarchical_filter_signals(pred_df)
    trades = execute_hierarchical_strategy(signal_df, threshold_dict(strategy), fallback="last")
    eval_row = evaluate_required_metric(
        trades,
        twap,
        "Hierarchical Ridge + VWOF Filter",
        sample,
    )
    return trades, twap, pd.DataFrame([eval_row])


def run_m2_style_validation(
    full_features: pd.DataFrame,
    stock: str,
    train_frac: float,
) -> Tuple[StockStrategy, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features, val_features = split_train_test_by_minute(full_features, train_frac=train_frac)
    train_features = train_features.copy()
    val_features = val_features.copy()
    train_features["sample"] = "train"
    val_features["sample"] = "val"

    selected_features, alpha_summary = select_features(train_features, stock)
    buy_model, sell_model = fit_models(train_features, selected_features)

    strategy_df = pd.concat([train_features, val_features], ignore_index=True)
    pred_df = add_model_predictions(
        strategy_df,
        buy_model,
        sell_model,
        selected_features["buy"],
        selected_features["sell"],
    )

    val_pred_df = pred_df[pred_df["sample"] == "val"].copy()
    best_buy_th, best_sell_th, best_score, threshold_search = tune_strategy4_threshold_values(
        val_pred_df,
        stock,
    )

    strategy = StockStrategy(
        stock=canonical_stock(stock),
        buy_model=buy_model,
        sell_model=sell_model,
        buy_features=selected_features["buy"],
        sell_features=selected_features["sell"],
        buy_threshold_quantile=np.nan,
        sell_threshold_quantile=np.nan,
        buy_threshold=best_buy_th,
        sell_threshold=best_sell_th,
        validation_improvement=best_score,
    )

    signal_df = add_hierarchical_filter_signals(pred_df)
    trades = execute_hierarchical_strategy(signal_df, threshold_dict(strategy), fallback="last")
    twap = execute_twap(pred_df)

    eval_rows = []
    for sample in ("train", "val"):
        eval_rows.append(
            evaluate_required_metric(
                trades[trades["sample"] == sample].copy(),
                twap[twap["sample"] == sample].copy(),
                "Hierarchical Ridge + VWOF Filter",
                sample,
            )
        )

    selected_feature_rows = pd.DataFrame(
        [
            {
                "stock": canonical_stock(stock),
                "side": side,
                "feature": feature,
                "rank": rank,
            }
            for side in ("buy", "sell")
            for rank, feature in enumerate(selected_features[side], start=1)
        ]
    )

    return (
        strategy,
        trades,
        pd.DataFrame(eval_rows),
        alpha_summary,
        threshold_search,
        selected_feature_rows,
    )


def trigger_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby(["sample", "stock"])
        .agg(
            n_minutes=("minute", "count"),
            buy_hit_rate=("buy_trigger", lambda x: (x == "ridge_signal_passed_vwof_filter").mean()),
            sell_hit_rate=("sell_trigger", lambda x: (x == "ridge_signal_passed_vwof_filter").mean()),
            avg_buy_vwof=("buy_vwof", "mean"),
            avg_sell_vwof=("sell_vwof", "mean"),
            avg_buy_spread=("buy_spread", "mean"),
            avg_sell_spread=("sell_spread", "mean"),
        )
        .reset_index()
    )


def strategy_summary_row(strategy: StockStrategy) -> Dict[str, object]:
    return {
        "stock": strategy.stock,
        "buy_threshold_quantile": strategy.buy_threshold_quantile,
        "sell_threshold_quantile": strategy.sell_threshold_quantile,
        "buy_threshold": strategy.buy_threshold,
        "sell_threshold": strategy.sell_threshold,
        "validation_improvement": strategy.validation_improvement,
        "buy_features": ", ".join(strategy.buy_features),
        "sell_features": ", ".join(strategy.sell_features),
    }


def save_if_not_empty(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def plot_improvement_summary(eval_df: pd.DataFrame, output_dir: Path) -> None:
    if eval_df.empty or "PCT_IMPROVEMENT" not in eval_df.columns:
        return

    try:
        cache_dir = output_dir / "_matplotlib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional reporting output
        warnings.warn(f"Could not create plots because matplotlib is unavailable: {exc}")
        return

    plot_df = eval_df.copy()
    plot_df["sample"] = pd.Categorical(
        plot_df["sample"],
        categories=["train", "val", "test"],
        ordered=True,
    )
    plot_df = plot_df.sort_values(["sample", "stock"])

    pivot_sample = plot_df.pivot_table(
        index="stock",
        columns="sample",
        values="PCT_IMPROVEMENT",
        aggfunc="first",
        observed=False,
    )
    pivot_sample = pivot_sample.dropna(axis=1, how="all")

    ax = pivot_sample.plot(kind="bar", figsize=(10, 5), width=0.72)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Hierarchical Strategy Improvement vs TWAP")
    ax.set_ylabel("Percentage Improvement")
    ax.set_xlabel("Stock")
    ax.legend(title="Sample")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "hierarchical_improvement_by_sample.png", dpi=180)
    plt.close()

    holdout_label = "test" if (plot_df["sample"] == "test").any() else "val"
    holdout_df = plot_df[plot_df["sample"] == holdout_label].copy()
    if not holdout_df.empty:
        ax = holdout_df.plot(
            x="stock",
            y="PCT_IMPROVEMENT",
            kind="bar",
            legend=False,
            figsize=(8, 4.5),
            width=0.68,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{holdout_label.title()} Improvement vs TWAP")
        ax.set_ylabel("Percentage Improvement")
        ax.set_xlabel("Stock")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / f"hierarchical_{holdout_label}_improvement.png", dpi=180)
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final hierarchical execution strategy.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=REPO_ROOT / "Project_Train_Datasets",
        help="Directory containing *_train.csv files.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=REPO_ROOT / "Project_Test_Datasets",
        help="Directory containing *_test.csv files. Missing test files are skipped.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "final_strategy_outputs",
        help="Directory where result CSV files will be written.",
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=list(DEFAULT_STOCKS),
        help="Stocks to run. Accepts AAPL or APPL for Apple.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of each stock's training minutes used for model fitting before internal tuning.",
    )
    parser.add_argument(
        "--split-train-test",
        action="store_true",
        help=(
            "Use each *_train.csv as a full historical sample, then split it by minute into "
            "a model-training part and a holdout test part. This is useful before real test "
            "datasets are released."
        ),
    )
    parser.add_argument(
        "--m2-style-validation",
        action="store_true",
        help=(
            "Reproduce the M2 notebook validation setup: split each train CSV into train/val, "
            "fit Ridge on train, tune Strategy 4 thresholds on val, then report Hierarchical "
            "performance on train and val. This mode is for notebook comparison, not final "
            "out-of-sample testing."
        ),
    )
    parser.add_argument(
        "--holdout-train-frac",
        type=float,
        default=0.70,
        help="When --split-train-test is used, fraction of minutes used for strategy training.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stocks = parse_stock_list(args.stocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_train_trades = []
    all_test_trades = []
    all_evals = []
    all_threshold_search = []
    all_alpha_summary = []
    all_selected_features = []
    strategy_rows = []

    for stock in stocks:
        train_path = find_stock_file(args.train_dir, stock, "train")
        if train_path is None:
            warnings.warn(f"{stock}: train file not found in {args.train_dir}; skipping.")
            continue

        print(f"\n=== {stock}: training from {train_path} ===")
        full_train_features = load_feature_frame(train_path, stock)

        if args.m2_style_validation:
            strategy, trades, eval_df_stock, alpha_summary, threshold_search, selected_features = (
                run_m2_style_validation(full_train_features, stock, args.holdout_train_frac)
            )
            strategy_rows.append(strategy_summary_row(strategy))
            all_alpha_summary.append(alpha_summary)
            all_threshold_search.append(threshold_search)
            all_selected_features.append(selected_features)
            all_train_trades.append(trades[trades["sample"] == "train"].copy())
            all_test_trades.append(trades[trades["sample"] == "val"].copy())
            all_evals.append(eval_df_stock)

            train_row = eval_df_stock[eval_df_stock["sample"] == "train"].iloc[0]
            val_row = eval_df_stock[eval_df_stock["sample"] == "val"].iloc[0]
            print(
                f"{stock}: M2-style train improvement = "
                f"{train_row['PCT_IMPROVEMENT']:.6f}% "
                f"({int(train_row['n_minutes'])} stock-minute trades)"
            )
            print(
                f"{stock}: M2-style val improvement = "
                f"{val_row['PCT_IMPROVEMENT']:.6f}% "
                f"({int(val_row['n_minutes'])} stock-minute trades)"
            )
            continue

        if args.split_train_test:
            train_features, holdout_test_features = split_train_test_by_minute(
                full_train_features,
                train_frac=args.holdout_train_frac,
            )
            print(
                f"{stock}: split existing train file into "
                f"{train_features['minute'].nunique()} train minutes and "
                f"{holdout_test_features['minute'].nunique()} test minutes."
            )
        else:
            train_features = full_train_features
            holdout_test_features = None

        strategy, alpha_summary, threshold_search, selected_features = train_stock_strategy(
            train_features,
            stock,
            args.train_frac,
        )
        strategy_rows.append(strategy_summary_row(strategy))
        all_alpha_summary.append(alpha_summary)
        all_threshold_search.append(threshold_search)
        all_selected_features.append(selected_features)

        train_trades, _, train_eval = run_strategy_on_features(train_features, strategy, "train")
        all_train_trades.append(train_trades)
        all_evals.append(train_eval)

        print(
            f"{stock}: train improvement = "
            f"{train_eval['PCT_IMPROVEMENT'].iloc[0]:.6f}% "
            f"({len(train_trades)} stock-minute trades)"
        )

        if args.split_train_test:
            test_features = holdout_test_features
            print(f"{stock}: testing on holdout split from the existing train file.")
        else:
            test_path = find_stock_file(args.test_dir, stock, "test")
            if test_path is None:
                print(f"{stock}: test file not found in {args.test_dir}; skipping test evaluation.")
                continue
            print(f"{stock}: testing from {test_path}")
            test_features = load_feature_frame(test_path, stock)

        if test_features is None or test_features.empty:
            print(f"{stock}: test file not found in {args.test_dir}; skipping test evaluation.")
            continue

        test_trades, _, test_eval = run_strategy_on_features(test_features, strategy, "test")
        all_test_trades.append(test_trades)
        all_evals.append(test_eval)

        print(
            f"{stock}: test improvement = "
            f"{test_eval['PCT_IMPROVEMENT'].iloc[0]:.6f}% "
            f"({len(test_trades)} stock-minute trades)"
        )

    eval_df = pd.concat(all_evals, ignore_index=True) if all_evals else pd.DataFrame()
    train_trades_df = pd.concat(all_train_trades, ignore_index=True) if all_train_trades else pd.DataFrame()
    test_trades_df = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    threshold_search_df = (
        pd.concat([df for df in all_threshold_search if not df.empty], ignore_index=True)
        if all_threshold_search
        else pd.DataFrame()
    )
    alpha_summary_df = (
        pd.concat([df for df in all_alpha_summary if not df.empty], ignore_index=True)
        if all_alpha_summary
        else pd.DataFrame()
    )
    selected_features_df = (
        pd.concat([df for df in all_selected_features if not df.empty], ignore_index=True)
        if all_selected_features
        else pd.DataFrame()
    )
    strategy_df = pd.DataFrame(strategy_rows)
    diagnostics_df = trigger_diagnostics(pd.concat([train_trades_df, test_trades_df], ignore_index=True))

    save_if_not_empty(eval_df, args.output_dir / "hierarchical_improvement_summary.csv")
    save_if_not_empty(train_trades_df, args.output_dir / "hierarchical_train_trades.csv")
    save_if_not_empty(test_trades_df, args.output_dir / "hierarchical_test_trades.csv")
    save_if_not_empty(strategy_df, args.output_dir / "hierarchical_strategy_parameters.csv")
    save_if_not_empty(threshold_search_df, args.output_dir / "hierarchical_threshold_search.csv")
    save_if_not_empty(alpha_summary_df, args.output_dir / "hierarchical_alpha_validation.csv")
    save_if_not_empty(selected_features_df, args.output_dir / "hierarchical_selected_features.csv")
    save_if_not_empty(diagnostics_df, args.output_dir / "hierarchical_trigger_diagnostics.csv")
    plot_improvement_summary(eval_df, args.output_dir)

    if not eval_df.empty:
        print("\n=== Required improvement summary ===")
        print(
            eval_df[
                [
                    "stock",
                    "sample",
                    "n_minutes",
                    "ALGO_COST",
                    "TWAP_COST",
                    "PCT_IMPROVEMENT",
                ]
            ].to_string(index=False)
        )

    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
