"""
main.py
-------
Full pipeline for the OBI + Microprice + OFI Strategy.

With train data (place CSV files in --data_dir):
    python main.py --data_dir /path/to/data

With weight optimisation:
    python main.py --data_dir /path/to/data --optimise

With synthetic data (dev / CI):
    python main.py --synthetic
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data_preprocessing import (
    TICKERS,
    compute_features,
    data_summary,
    load_all_tickers,
    split_train_test,
)
from strategy import StrategyConfig, OBIMicropriceStrategy, optimise_weights
from backtest import (
    compute_metrics,
    print_summary,
    plot_results,
    plot_signals,
    plot_event_composition,
    plot_signal_diagnostics,
    summary_table,
)


BASE_PRICES = {"AMZN": 185.0, "GOOG": 175.0, "INTC": 22.0, "MSFT": 415.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=str, default=".",
                   help="Directory containing {ticker}_5levels_train.csv files")
    # p.add_argument("--synthetic", action="store_true",
    #                help="Use synthetic data instead of real files")
    p.add_argument("--n_rows",    type=int, default=5000,
                   help="Rows per ticker for synthetic data")
    p.add_argument("--optimise",  action="store_true",
                   help="Grid-search signal weights on training portion")
    p.add_argument("--test_frac", type=float, default=0.30)
    p.add_argument("--out_dir",   type=str, default="output")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("OBI + Microprice + OFI Timing Strategy")

    # ── 1. Load data ───────────────────────────────────────────────────────────
    section("1  Loading data")
    lob_raw = load_all_tickers(args.data_dir)
    if not lob_raw:
        raise RuntimeError(
            f"No CSV files found in '{args.data_dir}'.\n"
            "Expected: AMZN_5levels_train.csv, GOOG_5levels_train.csv, "
            "INTC_5levels_train.csv, MSFT_5levels_train.csv"
        )

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    section("2  Computing features")
    lob_feat = {}
    for ticker, df in lob_raw.items():
        df = compute_features(df)
        lob_feat[ticker] = df
        n_min = df["minute"].nunique() if "minute" in df.columns else \
                df.index.floor("min").nunique()
        print(f"  [{ticker}] {df.shape[1]} columns  |  {n_min} complete minutes")

    data_summary(lob_feat)

    # ── 3. Train / test split ──────────────────────────────────────────────────
    section(f"3  Train/test split  (test = {args.test_frac:.0%})")
    lob_train, lob_test = {}, {}
    for ticker, df in lob_feat.items():
        tr, te = split_train_test(df, test_fraction=args.test_frac)
        lob_train[ticker] = tr
        lob_test[ticker]  = te
        n_min_tr = tr["minute"].nunique() if "minute" in tr.columns else \
                   tr.index.floor("min").nunique()
        n_min_te = te["minute"].nunique() if "minute" in te.columns else \
                   te.index.floor("min").nunique()
        print(f"  [{ticker}] train: {len(tr):,} rows / {n_min_tr} min  "
              f"|  test: {len(te):,} rows / {n_min_te} min")

    # ── 4. Strategy ────────────────────────────────────────────────────────────
    section("4  Running strategy")
    results = {}
    for ticker in lob_test:
        print(f"\n  ── {ticker} ──")
        df_te = lob_test[ticker]

        if args.optimise:
            print("    Optimising weights on training data …")
            buy_cfg  = optimise_weights(lob_train[ticker], "buy",  ticker)
            sell_cfg = optimise_weights(lob_train[ticker], "sell", ticker)
        else:
            buy_cfg = sell_cfg = StrategyConfig()

        buy_trades  = OBIMicropriceStrategy(buy_cfg ).run(df_te, "buy",  ticker)
        sell_trades = OBIMicropriceStrategy(sell_cfg).run(df_te, "sell", ticker)
        results[ticker] = {"buy": buy_trades, "sell": sell_trades}

        # Per-ticker quick stats
        for side, trades in [("buy", buy_trades), ("sell", sell_trades)]:
            m = compute_metrics(trades, side)
            if m:
                print(f"    {side.upper():4s}: {m['n_trades']:3d} trades  "
                      f"imp = {m['improvement_bps']:+.4f} bps  "
                      f"pct_better = {m['pct_better']:.1f}%  "
                      f"forced = {m['forced_pct']:.1f}%")

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    section("5  Performance summary")
    print_summary(results)
    tbl = summary_table(results)
    csv_path = out_dir / "performance_summary.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    # ── 6. Plots ───────────────────────────────────────────────────────────────
    section("6  Generating plots")
    plot_results(results, save_dir=out_dir)
    plot_event_composition(lob_test, save_dir=out_dir)

    for ticker in lob_test:
        for side in ("buy", "sell"):
            trades = results[ticker].get(side, [])
            if trades:
                plot_signals(
                    df        = lob_test[ticker],
                    trades    = trades,
                    side      = side,
                    ticker    = ticker,
                    n_minutes = 15,
                    save_dir  = out_dir,
                )

    # ── 7. Signal diagnostics ──────────────────────────────────────────────────
    section("7  Signal diagnostics")
    plot_signal_diagnostics(lob_test, save_dir=out_dir)

    banner(f"Done.  All outputs saved to: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────

def banner(msg: str):
    print(f"\n{'═'*65}\n  {msg}\n{'═'*65}")

def section(msg: str):
    print(f"\n[{msg}]")


if __name__ == "__main__":
    main()
