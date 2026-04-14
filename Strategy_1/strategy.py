"""
strategy.py
-----------
OBI + Microprice + OFI Timing Strategy.

Execution guarantee
-------------------
Exactly ONE trade is placed per complete minute per ticker.
The strategy selects the best tick within each minute based on
the composite score. A hard deadline rule (last `deadline_seconds`
of the minute) ensures a trade always executes even when no
favourable signal fires.

TWAP benchmark
--------------
TWAP executes at the very first tick of each minute (no signal).
Both strategies fill at the best available quote:
    BUY  → AskPrice_1  (market order lifts the ask)
    SELL → BidPrice_1  (market order hits the bid)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional
from data_preprocessing import compute_features, N_LEVELS


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # Composite score weights (must be positive; normalised internally)
    w_obi:       float = 0.40   # order-book imbalance
    w_micro:     float = 0.25   # microprice deviation (normalised)
    w_momentum:  float = 0.20   # short-term mid-price momentum
    w_ofi:       float = 0.15   # order-flow imbalance (exec direction)

    # Within-minute execution parameters
    score_percentile: float = 80.0   # trigger when score in top/bottom X%
    deadline_seconds: int   = 5      # force trade in last N seconds of minute
    momentum_window:  int   = 5      # seconds for momentum signal

    # Volatility filter: blank score when momentum is extreme
    vol_filter:     bool  = True
    vol_multiplier: float = 2.5      # threshold = multiplier × median |momentum|

    # Sell-side tightening
    sell_min_micro:      float = 0.0
    sell_min_obi:        float = 0.0
    sell_reversion_bonus: float = 0.20
    sell_chase_penalty:   float = 0.12
    sell_future_horizon:  int   = 30


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────

def composite_score(
    row: pd.Series,
    cfg: StrategyConfig,
    side: Literal["buy", "sell"] = "buy",
) -> float:
    """
    score = w1·OBI + w2·micro_norm + w3·sign(mom)·log(1+|mom|) + w4·OFI

    Positive → upward pressure → good for SELL
    Negative → downward pressure → good for BUY
    """
    def _get(col, default=0.0):
        v = row.get(col, default)
        return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    obi   = _get("obi")
    micro = _get("micro_norm")
    mom   = _get(f"momentum_{cfg.momentum_window}s")
    ofi   = _get("order_flow_imbalance")

    if side == "buy":
        return (
            cfg.w_obi      * obi
            + cfg.w_micro  * micro
            + cfg.w_momentum * np.sign(mom) * np.log1p(abs(mom))
            + cfg.w_ofi    * ofi
        )

    bid_mom = _get("bid_momentum_norm_5s")
    micro_trend = _get("micro_trend_5s")
    obi_change = _get("obi_change_5s")
    fade_bonus = max(-micro_trend, 0.0) + max(-obi_change, 0.0)
    chase_penalty = max(bid_mom, 0.0) ** 2

    return (
        cfg.w_obi       * max(obi, 0.0)
        + cfg.w_micro   * max(micro, 0.0)
        + cfg.w_momentum * max(bid_mom, 0.0)
        + cfg.w_ofi     * max(ofi, 0.0)
        + cfg.sell_reversion_bonus * fade_bonus
        - cfg.sell_chase_penalty * chase_penalty
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker:        str
    minute:        pd.Timestamp    # minute bucket (floor of exec_time)
    exec_time:     pd.Timestamp    # exact timestamp of execution
    price:         float           # fill price
    side:          str             # "buy" | "sell"
    score_at_exec: float           # composite score at the chosen tick
    forced:        bool = False    # True = deadline-triggered
    twap_price:    float = 0.0     # first-tick price (TWAP benchmark)
    n_ticks:       int  = 0        # number of ticks in this minute
    row_id:        int  = -1       # row number in the source frame


# ─────────────────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────────────────

class OBIMicropriceStrategy:
    """
    Runs over each complete 1-minute window in the data and places
    exactly one trade per minute.

    Within each minute:
    1. Compute composite score at every tick.
    2. Apply volatility filter (optional).
    3. BUY:  trigger on most-negative score (sell-side pressure).
       SELL: trigger on most-positive score (buy-side pressure).
    4. Prefer ticks without a concurrent execution (adverse-selection filter).
    5. Deadline fallback: if no signal fires before the last
       `deadline_seconds`, pick the best-scoring tick in that window.
    """

    def __init__(self, cfg: Optional[StrategyConfig] = None):
        self.cfg = cfg or StrategyConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        side: Literal["buy", "sell"],
        ticker: str,
    ) -> list[Trade]:
        """
        Parameters
        ----------
        df     : DataFrame loaded by load_lob_data + compute_features
        side   : "buy" or "sell"
        ticker : e.g. "AMZN"

        Returns
        -------
        list[Trade] — exactly one Trade per complete minute
        """
        # Ensure features exist
        if "obi" not in df.columns:
            df = compute_features(df)

        df = df.copy()

        # Recompute side-specific score with current cfg weights
        df["_score"] = df.apply(lambda r: composite_score(r, self.cfg, side), axis=1)

        # Volatility filter: neutralise score on runaway ticks
        if self.cfg.vol_filter:
            mom_col = f"momentum_{self.cfg.momentum_window}s"
            if mom_col in df.columns:
                mom_abs = df[mom_col].abs().fillna(0)
                med_mom = mom_abs.median()
                if med_mom > 0:
                    df.loc[mom_abs > self.cfg.vol_multiplier * med_mom, "_score"] = 0.0

        # Minute column (use pre-computed if available)
        if "minute" not in df.columns:
            df["minute"] = df.index.floor("min")

        trades: list[Trade] = []
        for minute, mdf in df.groupby("minute"):
            if len(mdf) < 2:
                continue   # skip incomplete / single-tick minutes
            t = self._execute_minute(mdf, side, ticker, pd.Timestamp(minute))
            if t is not None:
                trades.append(t)

        return trades

    # ── Within-minute logic ────────────────────────────────────────────────────

    def _execute_minute(
        self,
        mdf: pd.DataFrame,
        side: str,
        ticker: str,
        minute: pd.Timestamp,
    ) -> Optional[Trade]:
        cfg    = self.cfg
        scores = mdf["_score"]

        # TWAP benchmark: first tick of the minute
        twap_price = self._fill_price(mdf.iloc[0], side)

        # Percentile-based trigger (adapts to intra-minute score distribution)
        if side == "buy":
            cutoff      = np.percentile(scores, 100 - cfg.score_percentile)
            signal_mask = scores <= cutoff           # most negative = buy signal
            pick_best   = lambda s: s.idxmin()
        else:
            cutoff      = np.percentile(scores, cfg.score_percentile)
            signal_mask = scores >= cutoff           # most positive = sell signal
            pick_best   = lambda s: s.idxmax()

        # Deadline window
        deadline_ts = mdf.index[-1] - pd.Timedelta(seconds=cfg.deadline_seconds)
        pre         = mdf[mdf.index <= deadline_ts]
        deadline    = mdf[mdf.index >  deadline_ts]

        # Adverse-selection filter: prefer ticks with no concurrent execution
        exec_flag = mdf.get("exec_flag", pd.Series(0, index=mdf.index))
        no_exec   = exec_flag == 0
        structural_ok = self._structural_filter(mdf, side)

        forced    = False
        exec_idx  = None

        # Priority 1: signal + no concurrent execution
        pre_len = len(pre)
        pre_signal = signal_mask.to_numpy(dtype=bool)[:pre_len]
        pre_no_exec = no_exec.to_numpy(dtype=bool)[:pre_len]
        pre_structural = structural_ok.to_numpy(dtype=bool)[:pre_len]
        cand1 = pre.loc[pre_signal & pre_no_exec & pre_structural]
        if not cand1.empty:
            exec_idx = pick_best(scores.loc[cand1.index])

        # Priority 2: signal tick that passes structural filter
        if exec_idx is None:
            cand2 = pre.loc[pre_signal & pre_structural]
            if not cand2.empty:
                exec_idx = pick_best(scores.loc[cand2.index])

        # Priority 3: any signal tick (execution may be concurrent)
        if exec_idx is None:
            cand3 = pre.loc[pre_signal]
            if not cand3.empty:
                exec_idx = pick_best(scores.loc[cand3.index])

        # Priority 4: deadline window — best available
        if exec_idx is None:
            window = deadline if not deadline.empty else mdf.iloc[[-1]]
            if side == "sell":
                window_structural = self._structural_filter(window, side)
                filtered_window = window.loc[window_structural.to_numpy(dtype=bool)]
                if not filtered_window.empty:
                    window = filtered_window
            exec_idx = pick_best(scores.loc[window.index])
            forced   = True

        row   = mdf.loc[[exec_idx]].iloc[0]
        price = self._fill_price(row, side)
        score_at_exec = float(scores.loc[[exec_idx]].iloc[0])

        return Trade(
            ticker        = ticker,
            minute        = minute,
            exec_time     = exec_idx,
            price         = price,
            side          = side,
            score_at_exec = score_at_exec,
            forced        = forced,
            twap_price    = twap_price,
            n_ticks       = len(mdf),
            row_id        = int(row.get("_row_id", -1)),
        )

    # ── Fill price ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fill_price(row: pd.Series, side: str) -> float:
        """
        Market-order fill assumption:
            BUY  → AskPrice_1  (lifts the best ask)
            SELL → BidPrice_1  (hits the best bid)
        """
        return float(row["AskPrice_1"]) if side == "buy" else float(row["BidPrice_1"])

    def _structural_filter(self, mdf: pd.DataFrame, side: str) -> pd.Series:
        if side != "sell":
            return pd.Series(True, index=mdf.index)

        micro = mdf.get("micro_norm", pd.Series(0.0, index=mdf.index)).fillna(0.0)
        obi = mdf.get("obi", pd.Series(0.0, index=mdf.index)).fillna(0.0)
        micro_trend = mdf.get("micro_trend_5s", pd.Series(0.0, index=mdf.index)).fillna(0.0)
        bid_mom = mdf.get("bid_momentum_norm_5s", pd.Series(0.0, index=mdf.index)).fillna(0.0)

        momentum_cap = bid_mom.quantile(0.85) if len(bid_mom) > 1 else bid_mom.iloc[0]
        return (
            (micro >= self.cfg.sell_min_micro)
            & (obi >= self.cfg.sell_min_obi)
            & (micro_trend <= 0)
            & (bid_mom <= max(momentum_cap, 0.0))
        )


# ─────────────────────────────────────────────────────────────────────────────
# Weight optimiser
# ─────────────────────────────────────────────────────────────────────────────

def optimise_weights(
    df_train: pd.DataFrame,
    side: Literal["buy", "sell"],
    ticker: str,
    grid_steps: int = 4,
) -> StrategyConfig:
    """
    Grid-search (w_obi, w_micro, w_momentum, w_ofi) on training data.
    Objective: maximise mean per-trade improvement over TWAP.
    """
    grid   = np.linspace(0.05, 0.65, grid_steps)
    best   = -np.inf
    best_cfg = StrategyConfig()

    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                w4 = max(0.05, 1.0 - w1 - w2 - w3)
                if w4 > 0.70:
                    continue
                cfg    = StrategyConfig(w_obi=w1, w_micro=w2,
                                        w_momentum=w3, w_ofi=w4)
                if side == "sell":
                    imp = _sell_future_bid_objective(df_train, cfg, ticker)
                else:
                    trades = OBIMicropriceStrategy(cfg).run(df_train, side, ticker)
                    imp = _mean_improvement_bps(trades, side)
                if imp > best:
                    best     = imp
                    best_cfg = cfg

    print(f"  [{ticker}/{side}]  OBI={best_cfg.w_obi:.2f} "
          f"micro={best_cfg.w_micro:.2f} mom={best_cfg.w_momentum:.2f} "
          f"ofi={best_cfg.w_ofi:.2f}  →  {best:+.3f} bps objective")
    return best_cfg


def _mean_improvement(trades: list[Trade], side: str) -> float:
    if not trades:
        return 0.0
    d = [(t.twap_price - t.price) if side == "buy"
         else (t.price - t.twap_price) for t in trades]
    return float(np.mean(d))


def _mean_improvement_bps(trades: list[Trade], side: str) -> float:
    if not trades:
        return 0.0
    mean_improvement = _mean_improvement(trades, side)
    mean_twap = np.mean([t.twap_price for t in trades])
    return float(mean_improvement / (mean_twap + 1e-12) * 1e4)


def _sell_future_bid_objective(
    df_train: pd.DataFrame,
    cfg: StrategyConfig,
    ticker: str,
) -> float:
    trades = OBIMicropriceStrategy(cfg).run(df_train, "sell", ticker)
    if not trades:
        return -np.inf

    horizon = cfg.sell_future_horizon
    bid = df_train["BidPrice_1"].to_numpy(dtype=float)
    edges = []
    for trade in trades:
        row_id = trade.row_id
        if row_id < 0 or row_id + horizon >= len(bid):
            continue
        current_bid = bid[row_id]
        future_bid = bid[row_id + horizon]
        edges.append(current_bid - future_bid)

    if not edges:
        return -np.inf

    avg_bid = float(np.mean(bid))
    edge_bps = float(np.mean(edges) / (avg_bid + 1e-12) * 1e4)
    realised_bps = float(_mean_improvement(trades, "sell") / (avg_bid + 1e-12) * 1e4)

    # Keep the optimizer anchored to both future bid edge and realised TWAP improvement.
    return edge_bps + 0.5 * realised_bps


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run all tickers
# ─────────────────────────────────────────────────────────────────────────────

def run_all(
    lob_data: dict[str, pd.DataFrame],
    cfg: Optional[StrategyConfig] = None,
    optimise: bool = False,
    train_fraction: float = 0.70,
) -> dict[str, dict[str, list[Trade]]]:
    """
    Run buy + sell strategy for every ticker.

    Returns
    -------
    results[ticker]["buy"]  → list[Trade]
    results[ticker]["sell"] → list[Trade]
    """
    results = {}
    for ticker, df in lob_data.items():
        print(f"\n  ── {ticker}  ({len(df):,} ticks) ──")
        if "obi" not in df.columns:
            df = compute_features(df)

        if optimise:
            split    = int(len(df) * train_fraction)
            df_tr    = df.iloc[:split]
            df_te    = df.iloc[split:]
            buy_cfg  = optimise_weights(df_tr, "buy",  ticker)
            sell_cfg = optimise_weights(df_tr, "sell", ticker)
        else:
            df_te    = df
            buy_cfg  = sell_cfg = cfg or StrategyConfig()

        buy_trades  = OBIMicropriceStrategy(buy_cfg ).run(df_te, "buy",  ticker)
        sell_trades = OBIMicropriceStrategy(sell_cfg).run(df_te, "sell", ticker)
        results[ticker] = {"buy": buy_trades, "sell": sell_trades}

        print(f"    buy:  {len(buy_trades):3d} trades  "
              f"| sell: {len(sell_trades):3d} trades")
    return results
