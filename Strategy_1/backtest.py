"""
backtest.py
-----------
Performance metrics and visualisation for the OBI+Microprice+OFI strategy.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Optional

from strategy import Trade


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[Trade], side: str) -> dict:
    if not trades:
        return {}

    prices  = np.array([t.price      for t in trades])
    twaprx  = np.array([t.twap_price for t in trades])
    forced  = np.array([t.forced     for t in trades])

    per_trade  = (twaprx - prices) if side == "buy" else (prices - twaprx)
    per_share  = per_trade.mean()
    avg_twap   = twaprx.mean()

    return {
        "n_trades":              len(trades),
        "avg_strategy_price":    prices.mean(),
        "avg_twap_price":        avg_twap,
        "improvement_per_share": per_share,
        "improvement_bps":       per_share / avg_twap * 1e4,
        "pct_better":            (per_trade > 0).mean() * 100,
        "forced_pct":            forced.mean() * 100,
        "total_improvement":     per_share * len(trades),
        "std_per_trade":         per_trade.std(),
        "sharpe_improvement":    per_share / (per_trade.std() + 1e-8),
    }


def summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for ticker, sides in results.items():
        for side, trades in sides.items():
            m = compute_metrics(trades, side)
            if not m:
                continue
            rows.append({
                "Ticker":            ticker,
                "Side":              side.upper(),
                "Trades":            m["n_trades"],
                "Avg Strategy ($)":  round(m["avg_strategy_price"],    4),
                "Avg TWAP ($)":      round(m["avg_twap_price"],         4),
                "Improvement ($)":   round(m["improvement_per_share"],  5),
                "Improvement (bps)": round(m["improvement_bps"],        4),
                "% Better":          round(m["pct_better"],             1),
                "% Forced":          round(m["forced_pct"],             1),
                "Total α ($)":       round(m["total_improvement"],      3),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = {
        "Ticker":            "ALL",
        "Side":              "—",
        "Trades":            df["Trades"].sum(),
        "Avg Strategy ($)":  "—",
        "Avg TWAP ($)":      "—",
        "Improvement ($)":   "—",
        "Improvement (bps)": round(df["Improvement (bps)"].mean(), 4),
        "% Better":          round(df["% Better"].mean(),           1),
        "% Forced":          round(df["% Forced"].mean(),           1),
        "Total α ($)":       round(df["Total α ($)"].sum(),         3),
    }
    return pd.concat([df, pd.DataFrame([agg])], ignore_index=True)


def print_summary(results: dict) -> None:
    df = summary_table(results)
    if df.empty:
        print("No results.")
        return
    sep = "═" * 108
    print(f"\n{sep}")
    print("  PERFORMANCE SUMMARY  –  OBI + Microprice + OFI  vs  TWAP")
    print(sep)
    print(df.to_string(index=False))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

C = dict(strategy="#2563EB", twap="#DC2626",
         pos="#16A34A",       neg="#DC2626", neutral="#6B7280")


def _time_fmt(ax):
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: pd.Timestamp(x).strftime("%H:%M")))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# ─────────────────────────────────────────────────────────────────────────────
# Main performance dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    results: dict,
    save_dir: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    save_dir = Path(save_dir or "."); save_dir.mkdir(parents=True, exist_ok=True)

    for side in ("buy", "sell"):
        tickers = [t for t in results if results[t].get(side)]
        if not tickers:
            continue

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"OBI + Microprice + OFI Strategy  –  {side.upper()}",
            fontsize=15, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

        # ── (0,0) bps per ticker ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        bps = [compute_metrics(results[tk][side], side).get("improvement_bps", 0)
               for tk in tickers]
        clr = [C["pos"] if v >= 0 else C["neg"] for v in bps]
        bars = ax.bar(tickers, bps, color=clr, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set(title="Improvement vs TWAP (bps)", ylabel="bps")

        # ── (0,1) % better ───────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        pct = [compute_metrics(results[tk][side], side).get("pct_better", 0)
               for tk in tickers]
        clr2 = [C["pos"] if v >= 50 else C["neg"] for v in pct]
        bars2 = ax.bar(tickers, pct, color=clr2, edgecolor="white")
        ax.axhline(50, color="black", lw=0.8, ls="--", label="50%")
        ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set(title="% Minutes Beating TWAP", ylabel="%", ylim=(0, 108))
        ax.legend(fontsize=8)

        # ── (0,2) cumulative α ───────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        tk0   = tickers[0]
        trd0  = results[tk0][side]
        times = [t.exec_time for t in trd0]
        dlts  = [(t.twap_price - t.price) if side == "buy"
                 else (t.price - t.twap_price) for t in trd0]
        cum   = np.cumsum(dlts)
        ax.plot(times, cum, color=C["strategy"], lw=1.5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.fill_between(times, 0, cum, where=cum >= 0, alpha=0.15, color=C["pos"])
        ax.fill_between(times, 0, cum, where=cum <  0, alpha=0.15, color=C["neg"])
        ax.set(title=f"Cumulative α ({tk0}, $)", ylabel="$")
        _time_fmt(ax)

        # ── (1,0) per-trade improvement distribution ──────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        all_d = []
        for tk in tickers:
            for t in results[tk][side]:
                all_d.append((t.twap_price - t.price) if side == "buy"
                             else (t.price - t.twap_price))
        if all_d:
            ax.hist(all_d, bins=40, color=C["strategy"], alpha=0.75, edgecolor="white")
            ax.axvline(0, color="black", lw=1, ls="--")
            ax.axvline(np.mean(all_d), color=C["pos"], lw=1.5,
                       label=f"mean={np.mean(all_d):.5f}")
            ax.set(title="Per-Trade Improvement ($)", xlabel="$", ylabel="count")
            ax.legend(fontsize=8)

        # ── (1,1) score vs improvement ────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        sc, di = [], []
        for tk in tickers:
            for t in results[tk][side]:
                sc.append(t.score_at_exec)
                di.append((t.twap_price - t.price) if side == "buy"
                          else (t.price - t.twap_price))
        if sc:
            clr3 = [C["pos"] if v >= 0 else C["neg"] for v in di]
            ax.scatter(sc, di, c=clr3, alpha=0.35, s=12)
            ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.axvline(0, color="black", lw=0.7, ls="--")
            ax.set(title="Score at Execution vs Improvement",
                   xlabel="composite score", ylabel="$ vs TWAP")

        # ── (1,2) strategy vs TWAP price scatter ─────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        for tk in tickers:
            sp = [t.price      for t in results[tk][side]]
            tp = [t.twap_price for t in results[tk][side]]
            ax.scatter(tp, sp, alpha=0.35, s=12, label=tk)
        all_p = [t.price for tk in tickers for t in results[tk][side]] \
              + [t.twap_price for tk in tickers for t in results[tk][side]]
        lo, hi = min(all_p), max(all_p)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="parity")
        ax.set(title="Strategy vs TWAP Price",
               xlabel="TWAP ($)", ylabel="Strategy ($)")
        ax.legend(fontsize=7, markerscale=2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = save_dir / f"performance_{side}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        if show:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Signal decomposition per ticker/side
# ─────────────────────────────────────────────────────────────────────────────

def plot_signals(
    df: pd.DataFrame,
    trades: list[Trade],
    side: str,
    ticker: str,
    n_minutes: int = 15,
    save_dir: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """5-panel signal plot for the first n_minutes of the test window."""
    save_dir = Path(save_dir or "."); save_dir.mkdir(parents=True, exist_ok=True)
    if "score" not in df.columns:
        from data_preprocessing import compute_features
        df = compute_features(df)

    end = df.index[0] + pd.Timedelta(minutes=n_minutes)
    sub = df[df.index < end]
    sub_trades = [t for t in trades if t.exec_time < end]

    fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=True)
    fig.suptitle(
        f"{ticker} – Signal Decomposition ({side.upper()}) – first {n_minutes} min",
        fontsize=13, fontweight="bold",
    )

    # 1. Mid + microprice + executions
    ax = axes[0]
    ax.plot(sub.index, sub["MidPrice"],   color=C["neutral"],  lw=0.9, label="MidPrice")
    ax.plot(sub.index, sub["microprice"], color="#7C3AED", lw=0.8, ls="--",
            alpha=0.8, label="microprice")
    for t in sub_trades:
        clr = C["strategy"] if not t.forced else "#F59E0B"
        ax.axvline(t.exec_time, color=clr, lw=0.9, alpha=0.55)
        ax.scatter([t.exec_time], [t.price],     color=clr,      s=40, zorder=5)
        ax.scatter([t.minute],   [t.twap_price], color=C["twap"], s=25,
                   marker="x", zorder=5)
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.set_title("MidPrice & Microprice  (● strategy, ✕ TWAP, orange = forced)")

    # 2. OBI
    ax = axes[1]
    obi = sub["obi"].fillna(0)
    ax.fill_between(sub.index, 0, obi, where=obi > 0, color=C["pos"], alpha=0.4,
                    label="bid dominant")
    ax.fill_between(sub.index, 0, obi, where=obi < 0, color=C["neg"], alpha=0.4,
                    label="ask dominant")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("OBI")
    ax.legend(fontsize=8)
    ax.set_title("Order Book Imbalance (5 levels)")

    # 3. Order-flow imbalance
    ax = axes[2]
    ofi = sub.get("order_flow_imbalance", pd.Series(0, index=sub.index)).fillna(0)
    ax.fill_between(sub.index, 0, ofi, where=ofi > 0, color=C["pos"], alpha=0.4,
                    label="buy OFI")
    ax.fill_between(sub.index, 0, ofi, where=ofi < 0, color=C["neg"], alpha=0.4,
                    label="sell OFI")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("OFI")
    ax.legend(fontsize=8)
    ax.set_title("Order-Flow Imbalance (Direction × Execution, rolling 10)")

    # 4. Momentum
    ax = axes[3]
    mom = sub["momentum_5s"].fillna(0)
    ax.bar(sub.index, mom,
           width=pd.Timedelta(milliseconds=500),
           color=[C["pos"] if v >= 0 else C["neg"] for v in mom],
           alpha=0.65)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Δ MidPrice (5s)")
    ax.set_title("Short-Term Momentum (5-second Δ MidPrice)")

    # 5. Composite score
    ax = axes[4]
    sc = sub["score"].fillna(0)
    ax.plot(sub.index, sc, color="#7C3AED", lw=0.9, label="composite score")
    ax.axhline(0, color="black", lw=0.5)
    fav = sc < 0 if side == "buy" else sc > 0
    ax.fill_between(sub.index, sc, 0, where=fav,
                    color=C["pos"], alpha=0.25, label="favourable")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.set_title("Composite Score  (w·OBI + w·micro_norm + w·momentum + w·OFI)")

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = save_dir / f"signals_{ticker}_{side}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Event composition
# ─────────────────────────────────────────────────────────────────────────────

def plot_event_composition(
    frames: dict,
    save_dir: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    save_dir = Path(save_dir or "."); save_dir.mkdir(parents=True, exist_ok=True)
    event_map = {
        "NewLimitOrder_1=Yes_0=No":     "New Limit Order",
        "PartialCancel_1=Yes_0=No":     "Partial Cancel",
        "FullDelete_1=Yes_0=No":        "Full Delete",
        "VisibleExecution_1=Yes_0=No":  "Visible Exec",
        "HiddenExecution_1=Yes_0=No":   "Hidden Exec",
    }
    tickers = list(frames.keys())
    data = {}
    for col, label in event_map.items():
        data[label] = [int(frames[tk][col].sum()) if col in frames[tk].columns else 0
                       for tk in tickers]

    x, w = np.arange(len(tickers)), 0.15
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, vals) in enumerate(data.items()):
        ax.bar(x + i * w, vals, w, label=label, alpha=0.85)
    ax.set_xticks(x + w * 2); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Event-Type Composition per Ticker", fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.tight_layout()
    out = save_dir / "event_composition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Signal diagnostics heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_signal_diagnostics(
    lob_test: dict,
    save_dir: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    save_dir = Path(save_dir or "."); save_dir.mkdir(parents=True, exist_ok=True)
    signal_cols = [
        "obi", "obi_l1", "obi_l2",
        "micro_norm",
        "momentum_5s", "momentum_15s", "momentum_30s",
        "order_flow_imbalance",
        "spread_bps", "depth_ratio",
        "realised_vol_30",
        "score",
    ]

    records = []
    for ticker, df in lob_test.items():
        df = df.copy()
        df["fwd_30"] = df["MidPrice"].shift(-30) - df["MidPrice"]
        df = df.dropna(subset=["fwd_30"])
        for col in signal_cols:
            if col in df.columns:
                mask = df[col].notna()
                corr = df.loc[mask, col].corr(df.loc[mask, "fwd_30"])
                records.append({"Ticker": ticker, "Signal": col,
                                "Corr": round(corr, 4)})

    if not records:
        return

    pivot = pd.DataFrame(records).pivot(
        index="Signal", columns="Ticker", values="Corr")
    print("\n  Signal → 30-row Forward MidPrice Return Correlations")
    print(pivot.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.15, vmax=0.15, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)));  ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)));    ax.set_yticklabels(pivot.index,   fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Signal Correlations with 30-row Forward MidPrice Return",
                 fontweight="bold")
    plt.tight_layout()
    out = save_dir / "signal_diagnostics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)
