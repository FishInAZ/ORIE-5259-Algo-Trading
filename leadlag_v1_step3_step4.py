from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from leadlag_v1_step1_preprocessing import Step1Config, prepare_step1_panel


LOOKBACK_SECONDS = 5
FUTURE_HORIZONS = [1, 3, 5]
N_QUANTILES = 5
OUTPUT_DIR = Path("leadlag_v1_outputs")
SIGNAL_OUTPUT_DIR = OUTPUT_DIR / "step3_step4_signal"
TARGET_STOCK = "AMZN"


def compute_equal_weight_signal(mid_matrix: pd.DataFrame, lookback_seconds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    past_returns = mid_matrix.pct_change(periods=lookback_seconds)
    signals = pd.DataFrame(index=mid_matrix.index, columns=mid_matrix.columns, dtype=float)

    for stock in mid_matrix.columns:
        peer_average = past_returns.drop(columns=stock).mean(axis=1)
        signals[stock] = peer_average - past_returns[stock]

    return past_returns, signals


def build_signal_panel(aligned_panel: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    signal_long = (
        signals.stack()
        .rename("signal")
        .rename_axis(index=["timestamp", "stock"])
        .reset_index()
    )
    panel = aligned_panel.merge(signal_long, on=["timestamp", "stock"], how="left")
    return panel.sort_values(["stock", "timestamp"]).reset_index(drop=True)


def add_future_returns(signal_panel: pd.DataFrame, mid_matrix: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    future_returns = {}
    for horizon in horizons:
        future_returns[horizon] = mid_matrix.shift(-horizon).sub(mid_matrix).div(mid_matrix)

    future_long_frames = []
    for horizon, df in future_returns.items():
        long_df = (
            df.stack()
            .rename(f"future_return_{horizon}s")
            .rename_axis(index=["timestamp", "stock"])
            .reset_index()
        )
        future_long_frames.append(long_df)

    merged = signal_panel.copy()
    for future_df in future_long_frames:
        merged = merged.merge(future_df, on=["timestamp", "stock"], how="left")

    return merged


def summarize_signal_quality(signal_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stock, group in signal_panel.groupby("stock"):
        signal = group["signal"]
        rows.append(
            {
                "stock": stock,
                "rows": len(group),
                "signal_nan_count": int(signal.isna().sum()),
                "signal_nan_pct": signal.isna().mean(),
                "first_valid_timestamp": signal.loc[signal.notna()].index.min(),
                "signal_mean": signal.mean(),
                "signal_std": signal.std(),
                "signal_min": signal.min(),
                "signal_max": signal.max(),
            }
        )
    return pd.DataFrame(rows)


def compute_quintile_table(signal_panel: pd.DataFrame, horizon: int, n_quantiles: int) -> pd.DataFrame:
    future_col = f"future_return_{horizon}s"
    tables = []

    for stock, group in signal_panel.groupby("stock"):
        valid = group[["timestamp", "stock", "signal", future_col]].dropna().copy()
        valid["quintile"] = pd.qcut(valid["signal"], q=n_quantiles, labels=False, duplicates="drop") + 1

        summary = (
            valid.groupby("quintile", observed=False)
            .agg(
                avg_signal=("signal", "mean"),
                avg_future_return=(future_col, "mean"),
                obs=(future_col, "size"),
            )
            .reset_index()
        )
        summary.insert(0, "stock", stock)
        summary.insert(1, "horizon", f"{horizon}s")
        tables.append(summary)

    return pd.concat(tables, ignore_index=True)


def make_sanity_plot(quintile_table: pd.DataFrame, horizon: int, output_dir: Path) -> Path:
    subset = quintile_table[quintile_table["horizon"] == f"{horizon}s"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for stock, group in subset.groupby("stock"):
        ax.plot(group["quintile"], group["avg_future_return"], marker="o", linewidth=2, label=stock)

    ax.set_title(f"Signal Quintile vs Average Future Return ({horizon}s)")
    ax.set_xlabel("Signal Quintile")
    ax.set_ylabel("Average Future Return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / f"signal_quintile_vs_future_return_{TARGET_STOCK}_{horizon}s.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    SIGNAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = Step1Config(data_dir=Path("."))
    aligned_panel, mid_matrix = prepare_step1_panel(config)

    if TARGET_STOCK not in mid_matrix.columns:
        raise ValueError(f"TARGET_STOCK={TARGET_STOCK} is not available in the data.")

    past_returns, signals = compute_equal_weight_signal(mid_matrix, LOOKBACK_SECONDS)
    signal_panel = build_signal_panel(aligned_panel, signals)
    signal_panel = add_future_returns(signal_panel, mid_matrix, FUTURE_HORIZONS)
    signal_panel = signal_panel[signal_panel["stock"] == TARGET_STOCK].copy()

    signal_quality = summarize_signal_quality(signal_panel)
    signal_quality["first_valid_timestamp"] = signal_quality["first_valid_timestamp"].map(
        lambda idx: None if pd.isna(idx) else signal_panel.loc[idx, "timestamp"]
    )

    print("STEP 3 SIGNAL CHECK")
    print("=" * 80)
    print(f"target_stock = {TARGET_STOCK}")
    print(f"lookback_seconds = {LOOKBACK_SECONDS}")
    print()
    print("Signal sample:")
    print(
        signal_panel.loc[:, ["timestamp", "stock", "mid_price", "signal"]]
        .head(12)
        .to_string(index=False)
    )
    print()
    print("Signal quality summary:")
    print(signal_quality.to_string(index=False))
    print()
    print("NaN check: signal NaN should only appear in the first few seconds because of the 5-second lookback.")
    print()

    all_quintile_tables = []
    for horizon in FUTURE_HORIZONS:
        quintile_table = compute_quintile_table(signal_panel, horizon, N_QUANTILES)
        all_quintile_tables.append(quintile_table)
        plot_path = make_sanity_plot(quintile_table, horizon, SIGNAL_OUTPUT_DIR)

        print(f"STEP 4 SANITY CHECK ({horizon}s future return)")
        print("=" * 80)
        print(quintile_table.to_string(index=False))
        print(f"Saved plot: {plot_path}")
        print()

    final_quintile_table = pd.concat(all_quintile_tables, ignore_index=True)
    signal_panel.to_csv(SIGNAL_OUTPUT_DIR / f"signal_panel_{TARGET_STOCK}.csv", index=False)
    final_quintile_table.to_csv(SIGNAL_OUTPUT_DIR / f"signal_quintile_tables_{TARGET_STOCK}.csv", index=False)
    signal_quality.to_csv(SIGNAL_OUTPUT_DIR / f"signal_quality_summary_{TARGET_STOCK}.csv", index=False)

    print("Saved files:")
    print(SIGNAL_OUTPUT_DIR / f"signal_panel_{TARGET_STOCK}.csv")
    print(SIGNAL_OUTPUT_DIR / f"signal_quintile_tables_{TARGET_STOCK}.csv")
    print(SIGNAL_OUTPUT_DIR / f"signal_quality_summary_{TARGET_STOCK}.csv")


if __name__ == "__main__":
    main()
