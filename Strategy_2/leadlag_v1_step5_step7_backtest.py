from __future__ import annotations

from pathlib import Path

import pandas as pd

from Strategy_2.leadlag_v1_step1_preprocessing import Step1Config, prepare_step1_panel
from Strategy_2.leadlag_v1_step3_step4 import LOOKBACK_SECONDS, add_future_returns, build_signal_panel, compute_equal_weight_signal


OUTPUT_DIR = Path("leadlag_v1_outputs")
BACKTEST_OUTPUT_DIR = OUTPUT_DIR / "step5_step7_backtest"
BUY_SIDE = "BUY"
SIGNAL_QUANTILE = 0.80


def prepare_signal_panel() -> pd.DataFrame:
    config = Step1Config()
    aligned_panel, mid_matrix = prepare_step1_panel(config)
    _, signals = compute_equal_weight_signal(mid_matrix, LOOKBACK_SECONDS)
    signal_panel = build_signal_panel(aligned_panel, signals)
    signal_panel = add_future_returns(signal_panel, mid_matrix, [])
    signal_panel["minute"] = signal_panel["timestamp"].dt.floor("min")
    signal_panel["second_in_minute"] = signal_panel["timestamp"].dt.second
    return signal_panel.sort_values(["stock", "timestamp"]).reset_index(drop=True)


def build_buy_benchmark(signal_panel: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        signal_panel[signal_panel["second_in_minute"] == 0]
        .copy()
        .loc[:, ["timestamp", "minute", "stock", "best_ask"]]
        .rename(columns={"best_ask": "benchmark_execution_price"})
    )
    benchmark["side"] = BUY_SIDE
    benchmark = benchmark[
        ["timestamp", "minute", "stock", "side", "benchmark_execution_price"]
    ].reset_index(drop=True)
    return benchmark


def build_buy_strategy(signal_panel: pd.DataFrame, theta: float) -> pd.DataFrame:
    trades = []

    for (stock, minute), group in signal_panel.groupby(["stock", "minute"], sort=True):
        group = group.reset_index(drop=True)
        triggered_trade = None

        for idx in range(len(group) - 1):
            signal_t = group.loc[idx, "signal"]
            if pd.notna(signal_t) and signal_t > theta:
                exec_row = group.loc[idx + 1]
                triggered_trade = {
                    "stock": stock,
                    "minute": minute,
                    "signal_trigger_time": group.loc[idx, "timestamp"],
                    "execution_time": exec_row["timestamp"],
                    "execution_price": exec_row["best_ask"],
                    "whether_forced_execution": False,
                }
                break

        if triggered_trade is None:
            exec_row = group.iloc[-1]
            triggered_trade = {
                "stock": stock,
                "minute": minute,
                "signal_trigger_time": pd.NaT,
                "execution_time": exec_row["timestamp"],
                "execution_price": exec_row["best_ask"],
                "whether_forced_execution": True,
            }

        trades.append(triggered_trade)

    return pd.DataFrame(trades)


def compare_strategy_vs_benchmark(strategy_log: pd.DataFrame, benchmark_log: pd.DataFrame) -> pd.DataFrame:
    merged = benchmark_log.merge(
        strategy_log,
        on=["stock", "minute"],
        how="inner",
        validate="one_to_one",
    )
    merged["improvement"] = merged["benchmark_execution_price"] - merged["execution_price"]
    merged["improvement_bps"] = merged["improvement"] / merged["benchmark_execution_price"] * 10000.0
    return merged


def summarize_results(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_summary = (
        comparison.groupby("stock")
        .agg(
            trades=("improvement", "size"),
            average_improvement=("improvement", "mean"),
            median_improvement=("improvement", "median"),
            hit_rate=("improvement", lambda s: (s > 0).mean()),
            average_improvement_bps=("improvement_bps", "mean"),
            forced_execution_rate=("whether_forced_execution", "mean"),
        )
        .reset_index()
    )

    overall_summary = pd.DataFrame(
        [
            {
                "scope": "overall",
                "trades": len(comparison),
                "average_improvement": comparison["improvement"].mean(),
                "median_improvement": comparison["improvement"].median(),
                "hit_rate": (comparison["improvement"] > 0).mean(),
                "average_improvement_bps": comparison["improvement_bps"].mean(),
                "forced_execution_rate": comparison["whether_forced_execution"].mean(),
            }
        ]
    )
    return stock_summary, overall_summary


def save_plots(stock_summary: pd.DataFrame, comparison: pd.DataFrame) -> tuple[Path, Path]:
    avg_bar_path = BACKTEST_OUTPUT_DIR / "buy_average_improvement_bar_all.png"
    hist_path = BACKTEST_OUTPUT_DIR / "buy_improvement_histogram_all.png"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return avg_bar_path, hist_path

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stock_summary["stock"], stock_summary["average_improvement"], color="#2E86AB")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Average Buy Improvement vs Benchmark (All Stocks)")
    ax.set_xlabel("Stock")
    ax.set_ylabel("Average Improvement")
    fig.tight_layout()
    fig.savefig(avg_bar_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(comparison["improvement"], bins=30, color="#F18F01", edgecolor="black", alpha=0.8)
    ax.axvline(comparison["improvement"].mean(), color="black", linestyle="--", linewidth=2)
    ax.set_title("Buy Improvement Distribution (All Stocks)")
    ax.set_xlabel("Benchmark Price - Strategy Price")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    return avg_bar_path, hist_path


def main() -> None:
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    signal_panel = prepare_signal_panel()
    theta_by_stock = signal_panel.groupby("stock")["signal"].quantile(SIGNAL_QUANTILE).to_dict()

    benchmark_log = build_buy_benchmark(signal_panel)
    strategy_frames = []
    for stock, stock_panel in signal_panel.groupby("stock", sort=True):
        strategy_frames.append(build_buy_strategy(stock_panel, theta_by_stock[stock]))
    strategy_log = pd.concat(strategy_frames, ignore_index=True)
    comparison = compare_strategy_vs_benchmark(strategy_log, benchmark_log)
    stock_summary, overall_summary = summarize_results(comparison)
    avg_bar_path, hist_path = save_plots(stock_summary, comparison)

    benchmark_log.to_csv(BACKTEST_OUTPUT_DIR / "buy_benchmark_log_all.csv", index=False)
    strategy_log.to_csv(BACKTEST_OUTPUT_DIR / "buy_strategy_log_all.csv", index=False)
    comparison.to_csv(BACKTEST_OUTPUT_DIR / "buy_strategy_vs_benchmark_all.csv", index=False)
    stock_summary.to_csv(BACKTEST_OUTPUT_DIR / "buy_summary_by_stock_all.csv", index=False)
    overall_summary.to_csv(BACKTEST_OUTPUT_DIR / "buy_summary_overall_all.csv", index=False)

    print("STEP 5 BENCHMARK CHECK")
    print("=" * 80)
    print(f"stocks = {', '.join(sorted(theta_by_stock))}")
    print(f"side = {BUY_SIDE}")
    print(f"benchmark trades = {len(benchmark_log)}")
    print()
    print("Benchmark sample:")
    print(benchmark_log.head(10).to_string(index=False))
    print()

    print("STEP 6 STRATEGY CHECK")
    print("=" * 80)
    print("signal_threshold_theta_by_stock:")
    for stock, theta in sorted(theta_by_stock.items()):
        print(f"  {stock}: {theta:.8f}")
    print(f"strategy trades = {len(strategy_log)}")
    print()
    print("Strategy sample:")
    print(strategy_log.head(10).to_string(index=False))
    print()

    print("STEP 7 COMPARISON")
    print("=" * 80)
    print("Summary by stock:")
    print(stock_summary.to_string(index=False))
    print()
    print("Overall summary:")
    print(overall_summary.to_string(index=False))
    print()
    print("Comparison sample:")
    print(
        comparison[
            [
                "minute",
                "timestamp",
                "benchmark_execution_price",
                "signal_trigger_time",
                "execution_time",
                "execution_price",
                "whether_forced_execution",
                "improvement",
                "improvement_bps",
            ]
        ].head(10).to_string(index=False)
    )
    print()
    print("Saved files:")
    print(BACKTEST_OUTPUT_DIR / "buy_benchmark_log_all.csv")
    print(BACKTEST_OUTPUT_DIR / "buy_strategy_log_all.csv")
    print(BACKTEST_OUTPUT_DIR / "buy_strategy_vs_benchmark_all.csv")
    print(BACKTEST_OUTPUT_DIR / "buy_summary_by_stock_all.csv")
    print(BACKTEST_OUTPUT_DIR / "buy_summary_overall_all.csv")
    print(avg_bar_path)
    print(hist_path)


if __name__ == "__main__":
    main()
