"""
data_preprocessing.py
---------------------
Load and engineer features.

Column names are preserved exactly as they appear in the raw CSV:
    Time, BidPrice_5..1, BidSize_5..1, AskPrice_1..5, AskSize_1..5,
    OrderID, Size, Price, Direction_1=Buy_-1=Sell,
    NewLimitOrder_1=Yes_0=No, PartialCancel_1=Yes_0=No,
    FullDelete_1=Yes_0=No, VisibleExecution_1=Yes_0=No,
    HiddenExecution_1=Yes_0=No, TradingHalt_1=Yes_0=No,
    Spread, MidPrice

Time format: "HH:MM:SS.mmm"  (parsed with format="%H:%M:%S.%f")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

TICKERS  = ["AMZN", "GOOG", "INTC", "MSFT"]
N_LEVELS = 5

# Column groups (original names)
BID_PRICE_COLS = [f"BidPrice_{i}" for i in range(1, N_LEVELS + 1)]   # 1=best
ASK_PRICE_COLS = [f"AskPrice_{i}" for i in range(1, N_LEVELS + 1)]
BID_SIZE_COLS  = [f"BidSize_{i}"  for i in range(1, N_LEVELS + 1)]
ASK_SIZE_COLS  = [f"AskSize_{i}"  for i in range(1, N_LEVELS + 1)]

# Note: in the raw CSV BidPrice_5 is the *worst* bid, BidPrice_1 is the *best*.
# AskPrice_1 is the *best* ask, AskPrice_5 is the *worst*.
# All downstream code uses _1 for best level, consistent with this.

EVENT_COLS = [
    "NewLimitOrder_1=Yes_0=No",
    "PartialCancel_1=Yes_0=No",
    "FullDelete_1=Yes_0=No",
    "VisibleExecution_1=Yes_0=No",
    "HiddenExecution_1=Yes_0=No",
    "TradingHalt_1=Yes_0=No",
]


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_lob_data(
    filepath: str | Path,
    ticker: str,
    drop_halts: bool = True,
) -> pd.DataFrame:
    """
    Load one ticker's training CSV.

    * Timestamps are parsed directly from the raw "HH:MM:SS.%f" strings.
    * All original column names are kept.
    * A 'ticker' column is appended.
    * Trading-halt rows are optionally removed.

    Parameters
    ----------
    filepath   : path to "{ticker}_5levels_train.csv"
    ticker     : e.g. "AMZN"
    drop_halts : remove rows where TradingHalt_1=Yes_0=No == 1
    """
    df = pd.read_csv(Path(filepath))

    # ── Timestamp → DatetimeIndex ──────────────────────────────────────────────
    # Time column is "HH:MM:SS.mmm"
    df["Time_dt"] = pd.to_datetime(
        df["Time"],
        format="%H:%M:%S.%f",
        errors="coerce",
    )
    n_failed = df["Time_dt"].isna().sum()
    if n_failed:
        print(f"  [{ticker}] WARNING: {n_failed:,} rows failed timestamp parse — dropped")
        df = df[df["Time_dt"].notna()]

    df = df.set_index("Time_dt")
    df.index.name = "Time_dt"
    df = df.sort_index()

    # ── Minute bucket + row position within minute ─────────────────────────────
    # (Kept as columns so strategy can use them directly)
    df["minute"] = df.index.floor("min")
    df["row_in_minute"] = df.groupby("minute").cumcount()

    # ── Log return on MidPrice ─────────────────────────────────────────────────
    df["log_mid"] = np.log(df["MidPrice"])
    df["log_ret"] = df["log_mid"].diff()

    # ── Drop trading halts ─────────────────────────────────────────────────────
    halt_col = "TradingHalt_1=Yes_0=No"
    if drop_halts and halt_col in df.columns:
        n_before = len(df)
        df = df[df[halt_col] != 1]
        removed = n_before - len(df)
        if removed:
            print(f"  [{ticker}] Removed {removed:,} trading-halt rows")

    # ── Basic sanity ───────────────────────────────────────────────────────────
    df = df[(df["AskPrice_1"] > 0) & (df["BidPrice_1"] > 0)]
    df = df[df["AskPrice_1"] > df["BidPrice_1"]]

    # ── Tag ticker ─────────────────────────────────────────────────────────────
    df["ticker"] = ticker

    print(f"  [{ticker}] Loaded {len(df):,} rows  "
          f"[{df.index[0].strftime('%H:%M:%S.%f')[:-3]}  →  "
          f"{df.index[-1].strftime('%H:%M:%S.%f')[:-3]}]")
    return df


def load_all_tickers(
    data_dir: str | Path = ".",
    drop_halts: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load all four tickers from "{ticker}_5levels_train.csv" in data_dir.
    """
    data_dir = Path(data_dir)
    frames   = {}
    for ticker in TICKERS:
        fp = data_dir / f"{ticker}_5levels_train.csv"
        if not fp.exists():
            # Try case variants
            matches = list(data_dir.glob(f"*{ticker}*5level*train*")) \
                    + list(data_dir.glob(f"*{ticker}*train*"))
            fp = matches[0] if matches else fp
        if fp.exists():
            frames[ticker] = load_lob_data(fp, ticker, drop_halts=drop_halts)
        else:
            print(f"  [WARN] {fp} not found — skipping {ticker}")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add microstructure signals to a loaded LOB DataFrame.
    All new column names are lowercase to distinguish them from raw data.
    Original columns are never renamed or overwritten.

    New columns
    -----------
    spread_bps              Spread / MidPrice × 10,000
    microprice              Size-weighted mid at best bid/ask
    micro_minus_mid         microprice − MidPrice
    micro_norm              micro_minus_mid / Spread   ∈ [−0.5, 0.5]

    bid_depth               Total bid quantity (all 5 levels)
    ask_depth               Total ask quantity (all 5 levels)
    depth_ratio             bid_depth / ask_depth

    obi                     Order-book imbalance (5 levels)  ∈ [−1, 1]
    obi_l1                  OBI level-1 only
    obi_l2                  OBI levels 1–2

    bid_slope / ask_slope   Linear price slope across levels

    momentum_5s             Δ MidPrice over previous ~5 s
    momentum_15s            Δ MidPrice over previous ~15 s
    momentum_30s            Δ MidPrice over previous ~30 s
    bid_momentum_norm_5s    Δ BidPrice_1 / Spread over previous ~5 s
    micro_trend_5s          Δ micro_norm over previous ~5 s
    obi_change_5s           Δ obi over previous ~5 s
    realised_vol_30         Rolling 30-row std of log_ret

    exec_flag               1 if visible or hidden execution on this row
    cancel_flag             1 if partial or full cancel on this row
    buy_pressure            Rolling-10 count of buy executions
    sell_pressure           Rolling-10 count of sell executions
    order_flow_imbalance    (buy_pressure − sell_pressure) / total  ∈ [−1, 1]

    score                   Composite signal (positive = upward pressure)
    score_buy               −score  (high value = good time to BUY)
    score_sell              sell-specific score  (high value = good time to SELL)
    """
    df = df.copy()
    df["_row_id"] = np.arange(len(df), dtype=int)

    # ── Spread / basis ─────────────────────────────────────────────────────────
    df["spread_bps"] = df["Spread"] / df["MidPrice"] * 1e4

    # ── Microprice ─────────────────────────────────────────────────────────────
    qb1, qa1 = df["BidSize_1"], df["AskSize_1"]
    df["microprice"]      = (df["AskPrice_1"] * qb1 + df["BidPrice_1"] * qa1) / (qb1 + qa1)
    df["micro_minus_mid"] = df["microprice"] - df["MidPrice"]
    df["micro_norm"]      = df["micro_minus_mid"] / df["Spread"].replace(0, np.nan)

    # ── Depth ──────────────────────────────────────────────────────────────────
    bid_sz = [df[f"BidSize_{i}"] for i in range(1, N_LEVELS + 1)]
    ask_sz = [df[f"AskSize_{i}"] for i in range(1, N_LEVELS + 1)]
    df["bid_depth"] = sum(bid_sz)
    df["ask_depth"] = sum(ask_sz)
    df["depth_ratio"] = df["bid_depth"] / df["ask_depth"].replace(0, np.nan)

    # ── OBI variants ──────────────────────────────────────────────────────────
    denom = (df["bid_depth"] + df["ask_depth"]).replace(0, np.nan)
    df["obi"] = (df["bid_depth"] - df["ask_depth"]) / denom

    d1 = (qb1 + qa1).replace(0, np.nan)
    df["obi_l1"] = (qb1 - qa1) / d1

    qb2, qa2 = df["BidSize_2"], df["AskSize_2"]
    d2 = (qb1 + qb2 + qa1 + qa2).replace(0, np.nan)
    df["obi_l2"] = (qb1 + qb2 - qa1 - qa2) / d2

    # ── Book slope ─────────────────────────────────────────────────────────────
    lvl = np.arange(1, N_LEVELS + 1, dtype=float)
    mu_lvl  = lvl.mean()
    var_lvl = ((lvl - mu_lvl) ** 2).sum()

    def slope_vec(price_cols):
        mat  = np.column_stack([df[c].values for c in price_cols])
        mu_y = mat.mean(axis=1, keepdims=True)
        cov  = ((mat - mu_y) * (lvl - mu_lvl)).sum(axis=1)
        return cov / (var_lvl + 1e-12)

    # Note: BidPrice_1 is best, BidPrice_5 is worst → slope should be negative
    df["bid_slope"] = slope_vec([f"BidPrice_{i}" for i in range(1, N_LEVELS + 1)])
    df["ask_slope"] = slope_vec([f"AskPrice_{i}" for i in range(1, N_LEVELS + 1)])

    # ── Momentum (row-based shift; avoids duplicate-index issues) ─────────────
    # Approximate: assumes ~1 row per second on average.
    # For precise time-based momentum, resample first or deduplicate index.
    for sec in [5, 15, 30]:
        df[f"momentum_{sec}s"] = df["MidPrice"].diff(sec)
    df["bid_momentum_5s"] = df["BidPrice_1"].diff(5)
    df["bid_momentum_norm_5s"] = df["bid_momentum_5s"] / df["Spread"].replace(0, np.nan)
    df["micro_trend_5s"] = df["micro_norm"].diff(5)
    df["obi_change_5s"] = df["obi"].diff(5)

    # ── Realised volatility ────────────────────────────────────────────────────
    if "log_ret" not in df.columns:
        df["log_ret"] = np.log(df["MidPrice"]).diff()
    df["realised_vol_30"] = df["log_ret"].rolling(30, min_periods=5).std()

    # ── Order-flow imbalance (uses Direction + Execution columns) ─────────────
    vis_col = "VisibleExecution_1=Yes_0=No"
    hid_col = "HiddenExecution_1=Yes_0=No"
    pcl_col = "PartialCancel_1=Yes_0=No"
    fdl_col = "FullDelete_1=Yes_0=No"
    dir_col = "Direction_1=Buy_-1=Sell"

    if vis_col in df.columns:
        df["exec_flag"]   = ((df[vis_col] == 1) | (df[hid_col] == 1)).astype(float)
        df["cancel_flag"] = ((df[pcl_col] == 1) | (df[fdl_col] == 1)).astype(float)

        direction = df[dir_col] if dir_col in df.columns else pd.Series(0, index=df.index)
        df["buy_exec"]    = ((df["exec_flag"] == 1) & (direction ==  1)).astype(float)
        df["sell_exec"]   = ((df["exec_flag"] == 1) & (direction == -1)).astype(float)
        df["buy_pressure"]  = df["buy_exec" ].rolling(10, min_periods=1).sum()
        df["sell_pressure"] = df["sell_exec"].rolling(10, min_periods=1).sum()
        ofi_denom = (df["buy_pressure"] + df["sell_pressure"]).replace(0, np.nan)
        df["order_flow_imbalance"] = (
            (df["buy_pressure"] - df["sell_pressure"]) / ofi_denom
        )
    else:
        for c in ["exec_flag", "cancel_flag", "buy_pressure",
                  "sell_pressure", "order_flow_imbalance"]:
            df[c] = 0.0

    # ── Composite score ────────────────────────────────────────────────────────
    # Default weights — overridden at runtime by StrategyConfig
    W1, W2, W3, W4 = 0.40, 0.25, 0.20, 0.15
    mom   = df["momentum_5s"].fillna(0)
    micro = df["micro_norm"].fillna(0)
    ofi   = df["order_flow_imbalance"].fillna(0)

    df["score"] = (
        W1 * df["obi"].fillna(0)
        + W2 * micro
        + W3 * np.sign(mom) * np.log1p(mom.abs())
        + W4 * ofi
    )
    df["score_buy"]  = -df["score"]
    sell_bid_mom = df["bid_momentum_norm_5s"].fillna(0)
    sell_fade = (-df["micro_trend_5s"].fillna(0)).clip(lower=0) \
              + (-df["obi_change_5s"].fillna(0)).clip(lower=0)
    sell_chase = sell_bid_mom.clip(lower=0)
    df["score_sell"] = (
        0.35 * df["obi"].fillna(0).clip(lower=0)
        + 0.25 * micro.clip(lower=0)
        + 0.15 * ofi.clip(lower=0)
        + 0.15 * sell_fade
        + 0.10 * sell_bid_mom.clip(lower=0)
        - 0.10 * sell_chase.pow(2)
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_lob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper matching the notebook pattern shown in the spec.
    Returns df with ts, minute, row_in_minute, log_mid, log_ret columns
    (already added by load_lob_data; this is a no-op if called again).
    """
    df = df.copy()
    if "minute" not in df.columns:
        df["minute"]         = df.index.floor("min")
        df["row_in_minute"]  = df.groupby("minute").cumcount()
    if "log_mid" not in df.columns:
        df["log_mid"] = np.log(df["MidPrice"])
        df["log_ret"] = df["log_mid"].diff()
    return df


def split_train_test(
    df: pd.DataFrame,
    test_fraction: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological split — no shuffling."""
    n = int(len(df) * (1 - test_fraction))
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def data_summary(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Print and return a compact summary table."""
    rows = []
    for ticker, df in frames.items():
        n_exec = int(df.get("VisibleExecution_1=Yes_0=No",
                            pd.Series(0)).sum()
                   + df.get("HiddenExecution_1=Yes_0=No",
                            pd.Series(0)).sum())
        n_min  = df["minute"].nunique() if "minute" in df.columns else \
                 df.index.floor("min").nunique()
        rows.append({
            "Ticker":       ticker,
            "Rows":         f"{len(df):,}",
            "Minutes":      n_min,
            "Start":        df.index[0].strftime("%H:%M:%S"),
            "End":          df.index[-1].strftime("%H:%M:%S"),
            "AvgBid ($)":   round(df["BidPrice_1"].mean(), 4),
            "AvgAsk ($)":   round(df["AskPrice_1"].mean(), 4),
            "AvgSpread($)": round(df["Spread"].mean(),     5),
            "AvgSprd(bps)": round((df["Spread"] / df["MidPrice"] * 1e4).mean(), 3),
            "Executions":   f"{n_exec:,}",
        })
    tbl = pd.DataFrame(rows)
    print("\n" + "═" * 100)
    print("  DATA SUMMARY")
    print("═" * 100)
    print(tbl.to_string(index=False))
    print("═" * 100 + "\n")
    return tbl

