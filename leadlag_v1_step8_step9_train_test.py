from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from leadlag_v1_step1_preprocessing import Step1Config, prepare_step1_panel
from leadlag_v1_step3_step4 import build_signal_panel, compute_equal_weight_signal


OUTPUT_DIR = Path("leadlag_v1_outputs")
STEP89_OUTPUT_DIR = OUTPUT_DIR / "step8_step9_train_test"
LOOKBACK_GRID = [3, 5, 10]
THRESHOLD_QUANTILE_GRID = [0.7, 0.8, 0.9]
TRAIN_RATIO = 0.70


def prepare_target_panel(lookback_seconds: int, stock: str) -> pd.DataFrame:
    config = Step1Config()
    aligned_panel, mid_matrix = prepare_step1_panel(config)
    _, signals = compute_equal_weight_signal(mid_matrix, lookback_seconds)
    panel = build_signal_panel(aligned_panel, signals)
    panel = panel[panel["stock"] == stock].copy()
    panel["minute"] = panel["timestamp"].dt.floor("min")
    panel["second_in_minute"] = panel["timestamp"].dt.second
    return panel.sort_values("timestamp").reset_index(drop=True)


def split_train_test_minutes(panel: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    minutes = pd.Index(sorted(panel["minute"].unique()))
    split_idx = max(1, int(len(minutes) * TRAIN_RATIO))
    split_idx = min(split_idx, len(minutes) - 1)
    return minutes[:split_idx], minutes[split_idx:]


def subset_by_minutes(panel: pd.DataFrame, minutes: pd.Index) -> pd.DataFrame:
    return panel[panel["minute"].isin(minutes)].copy().reset_index(drop=True)


def build_benchmark(panel: pd.DataFrame, side: str) -> pd.DataFrame:
    price_col = "best_ask" if side == "BUY" else "best_bid"
    out_col = "benchmark_execution_price"
    benchmark = (
        panel[panel["second_in_minute"] == 0]
        .copy()
        .loc[:, ["timestamp", "minute", "stock", price_col]]
        .rename(columns={price_col: out_col})
    )
    benchmark["side"] = side
    return benchmark[["timestamp", "minute", "stock", "side", out_col]].reset_index(drop=True)


def build_strategy(panel: pd.DataFrame, side: str, theta: float) -> pd.DataFrame:
    price_col = "best_ask" if side == "BUY" else "best_bid"
    trigger_condition = (
        (lambda x: x > theta) if side == "BUY" else (lambda x: x < -theta)
    )
    stock = panel["stock"].iloc[0]

    trades = []
    for minute, group in panel.groupby("minute", sort=True):
        group = group.reset_index(drop=True)
        trade = None

        for idx in range(len(group) - 1):
            signal_t = group.loc[idx, "signal"]
            if pd.notna(signal_t) and trigger_condition(signal_t):
                exec_row = group.loc[idx + 1]
                trade = {
                    "stock": stock,
                    "side": side,
                    "minute": minute,
                    "signal_trigger_time": group.loc[idx, "timestamp"],
                    "execution_time": exec_row["timestamp"],
                    "execution_price": exec_row[price_col],
                    "whether_forced_execution": False,
                }
                break

        if trade is None:
            exec_row = group.iloc[-1]
            trade = {
                "stock": stock,
                "side": side,
                "minute": minute,
                "signal_trigger_time": pd.NaT,
                "execution_time": exec_row["timestamp"],
                "execution_price": exec_row[price_col],
                "whether_forced_execution": True,
            }

        trades.append(trade)

    return pd.DataFrame(trades)


def compare_strategy_vs_benchmark(
    strategy_log: pd.DataFrame,
    benchmark_log: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    merged = benchmark_log.merge(
        strategy_log,
        on=["stock", "side", "minute"],
        how="inner",
        validate="one_to_one",
    )
    if side == "BUY":
        merged["improvement"] = merged["benchmark_execution_price"] - merged["execution_price"]
    else:
        merged["improvement"] = merged["execution_price"] - merged["benchmark_execution_price"]

    merged["improvement_bps"] = merged["improvement"] / merged["benchmark_execution_price"] * 10000.0
    return merged


def summarize_side(comparison: pd.DataFrame, side: str, split_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": split_name,
                "stock": comparison["stock"].iloc[0],
                "side": side,
                "trades": len(comparison),
                "average_improvement": comparison["improvement"].mean(),
                "median_improvement": comparison["improvement"].median(),
                "hit_rate": (comparison["improvement"] > 0).mean(),
                "average_improvement_bps": comparison["improvement_bps"].mean(),
                "forced_execution_rate": comparison["whether_forced_execution"].mean(),
            }
        ]
    )


def evaluate_params(panel: pd.DataFrame, theta_quantile: float) -> dict:
    theta = panel["signal"].quantile(theta_quantile)

    buy_benchmark = build_benchmark(panel, "BUY")
    sell_benchmark = build_benchmark(panel, "SELL")
    buy_strategy = build_strategy(panel, "BUY", theta)
    sell_strategy = build_strategy(panel, "SELL", theta)

    buy_cmp = compare_strategy_vs_benchmark(buy_strategy, buy_benchmark, "BUY")
    sell_cmp = compare_strategy_vs_benchmark(sell_strategy, sell_benchmark, "SELL")

    pooled = pd.concat([buy_cmp.assign(side_eval="BUY"), sell_cmp.assign(side_eval="SELL")], ignore_index=True)

    return {
        "theta": theta,
        "buy_benchmark": buy_benchmark,
        "sell_benchmark": sell_benchmark,
        "buy_strategy": buy_strategy,
        "sell_strategy": sell_strategy,
        "buy_cmp": buy_cmp,
        "sell_cmp": sell_cmp,
        "pooled_cmp": pooled,
        "objective_bps": pooled["improvement_bps"].mean(),
    }


def make_test_plots(summary_df: pd.DataFrame, pooled_test: pd.DataFrame, stock: str) -> tuple[Path, Path]:
    avg_bar_path = STEP89_OUTPUT_DIR / f"test_average_improvement_by_side_{stock}.png"
    hist_path = STEP89_OUTPUT_DIR / f"test_pooled_improvement_histogram_{stock}.png"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return avg_bar_path, hist_path

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary_df["side"], summary_df["average_improvement"], color=["#2E86AB", "#C0392B"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Test Average Improvement by Side ({stock})")
    ax.set_xlabel("Side")
    ax.set_ylabel("Average Improvement")
    fig.tight_layout()
    fig.savefig(avg_bar_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pooled_test["improvement"], bins=30, color="#7DCEA0", edgecolor="black", alpha=0.85)
    ax.axvline(pooled_test["improvement"].mean(), color="black", linestyle="--", linewidth=2)
    ax.set_title(f"Test Pooled Improvement Distribution ({stock})")
    ax.set_xlabel("Improvement")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    return avg_bar_path, hist_path


def main() -> None:
    STEP89_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tuning = []
    all_side_summary = []
    all_overall_summary = []
    all_test_buy = []
    all_test_sell = []

    for stock in ["AMZN", "GOOG", "INTC", "MSFT"]:
        tuning_rows = []
        best_result = None
        best_panel = None
        train_minutes = None
        test_minutes = None

        for lookback_seconds, theta_quantile in product(LOOKBACK_GRID, THRESHOLD_QUANTILE_GRID):
            panel = prepare_target_panel(lookback_seconds, stock)
            if train_minutes is None or test_minutes is None:
                train_minutes, test_minutes = split_train_test_minutes(panel)

            train_panel = subset_by_minutes(panel, train_minutes)
            result = evaluate_params(train_panel, theta_quantile)

            tuning_rows.append(
                {
                    "stock": stock,
                    "lookback_seconds": lookback_seconds,
                    "threshold_quantile": theta_quantile,
                    "theta": result["theta"],
                    "train_pooled_average_improvement_bps": result["objective_bps"],
                    "train_buy_average_improvement_bps": result["buy_cmp"]["improvement_bps"].mean(),
                    "train_sell_average_improvement_bps": result["sell_cmp"]["improvement_bps"].mean(),
                }
            )

            if best_result is None or result["objective_bps"] > best_result["objective_bps"]:
                best_result = result | {
                    "lookback_seconds": lookback_seconds,
                    "threshold_quantile": theta_quantile,
                }
                best_panel = panel

        if best_result is None or best_panel is None or train_minutes is None or test_minutes is None:
            raise RuntimeError(f"Parameter search failed to produce a result for {stock}.")

        test_panel = subset_by_minutes(best_panel, test_minutes)
        test_result = evaluate_params(test_panel, best_result["threshold_quantile"])

        train_buy_summary = summarize_side(best_result["buy_cmp"], "BUY", "train")
        train_sell_summary = summarize_side(best_result["sell_cmp"], "SELL", "train")
        test_buy_summary = summarize_side(test_result["buy_cmp"], "BUY", "test")
        test_sell_summary = summarize_side(test_result["sell_cmp"], "SELL", "test")
        side_summary = pd.concat(
            [train_buy_summary, train_sell_summary, test_buy_summary, test_sell_summary],
            ignore_index=True,
        )

        overall_summary = pd.DataFrame(
            [
                {
                    "split": "train",
                    "stock": stock,
                    "trades": len(best_result["pooled_cmp"]),
                    "average_improvement": best_result["pooled_cmp"]["improvement"].mean(),
                    "median_improvement": best_result["pooled_cmp"]["improvement"].median(),
                    "hit_rate": (best_result["pooled_cmp"]["improvement"] > 0).mean(),
                    "average_improvement_bps": best_result["pooled_cmp"]["improvement_bps"].mean(),
                },
                {
                    "split": "test",
                    "stock": stock,
                    "trades": len(test_result["pooled_cmp"]),
                    "average_improvement": test_result["pooled_cmp"]["improvement"].mean(),
                    "median_improvement": test_result["pooled_cmp"]["improvement"].median(),
                    "hit_rate": (test_result["pooled_cmp"]["improvement"] > 0).mean(),
                    "average_improvement_bps": test_result["pooled_cmp"]["improvement_bps"].mean(),
                },
            ]
        )

        tuning_df = pd.DataFrame(tuning_rows).sort_values(
            ["train_pooled_average_improvement_bps", "lookback_seconds", "threshold_quantile"],
            ascending=[False, True, True],
        )

        avg_bar_path, hist_path = make_test_plots(
            side_summary[side_summary["split"] == "test"],
            test_result["pooled_cmp"],
            stock,
        )

        tuning_df.to_csv(STEP89_OUTPUT_DIR / f"train_grid_search_{stock}.csv", index=False)
        side_summary.to_csv(STEP89_OUTPUT_DIR / f"train_test_side_summary_{stock}.csv", index=False)
        overall_summary.to_csv(STEP89_OUTPUT_DIR / f"train_test_overall_summary_{stock}.csv", index=False)
        test_result["buy_cmp"].to_csv(STEP89_OUTPUT_DIR / f"test_buy_strategy_vs_benchmark_{stock}.csv", index=False)
        test_result["sell_cmp"].to_csv(STEP89_OUTPUT_DIR / f"test_sell_strategy_vs_benchmark_{stock}.csv", index=False)

        all_tuning.append(tuning_df)
        all_side_summary.append(side_summary)
        all_overall_summary.append(overall_summary)
        all_test_buy.append(test_result["buy_cmp"])
        all_test_sell.append(test_result["sell_cmp"])

        print("STEP 8 SELL SIDE")
        print("=" * 80)
        print(f"{stock}: sell logic implemented:")
        print("signal_t < -theta -> execute sell at t+1 best bid; otherwise force sell at minute end best bid.")
        print()
        print("STEP 9 TRAIN / TEST SPLIT")
        print("=" * 80)
        print(f"target_stock = {stock}")
        print(f"train_ratio = {TRAIN_RATIO}")
        print(f"train_minutes = {len(train_minutes)}, test_minutes = {len(test_minutes)}")
        print(f"train_time_range = {train_minutes[0]} -> {train_minutes[-1]}")
        print(f"test_time_range = {test_minutes[0]} -> {test_minutes[-1]}")
        print()
        print("Best train parameters:")
        print(
            pd.DataFrame(
                [
                    {
                        "stock": stock,
                        "lookback_seconds": best_result["lookback_seconds"],
                        "threshold_quantile": best_result["threshold_quantile"],
                        "theta": best_result["theta"],
                        "train_pooled_average_improvement_bps": best_result["objective_bps"],
                    }
                ]
            ).to_string(index=False)
        )
        print()
        print("Test side summary:")
        print(side_summary[side_summary["split"] == "test"].to_string(index=False))
        print()
        print("Test overall summary:")
        print(overall_summary[overall_summary["split"] == "test"].to_string(index=False))
        print()
        print("Saved files:")
        print(STEP89_OUTPUT_DIR / f"train_grid_search_{stock}.csv")
        print(STEP89_OUTPUT_DIR / f"train_test_side_summary_{stock}.csv")
        print(STEP89_OUTPUT_DIR / f"train_test_overall_summary_{stock}.csv")
        print(STEP89_OUTPUT_DIR / f"test_buy_strategy_vs_benchmark_{stock}.csv")
        print(STEP89_OUTPUT_DIR / f"test_sell_strategy_vs_benchmark_{stock}.csv")
        print(avg_bar_path)
        print(hist_path)
        print()

    pd.concat(all_tuning, ignore_index=True).to_csv(STEP89_OUTPUT_DIR / "train_grid_search_all.csv", index=False)
    pd.concat(all_side_summary, ignore_index=True).to_csv(STEP89_OUTPUT_DIR / "train_test_side_summary_all.csv", index=False)
    pd.concat(all_overall_summary, ignore_index=True).to_csv(STEP89_OUTPUT_DIR / "train_test_overall_summary_all.csv", index=False)
    pd.concat(all_test_buy, ignore_index=True).to_csv(STEP89_OUTPUT_DIR / "test_buy_strategy_vs_benchmark_all.csv", index=False)
    pd.concat(all_test_sell, ignore_index=True).to_csv(STEP89_OUTPUT_DIR / "test_sell_strategy_vs_benchmark_all.csv", index=False)
    print("Combined saved files:")
    print(STEP89_OUTPUT_DIR / "train_grid_search_all.csv")
    print(STEP89_OUTPUT_DIR / "train_test_side_summary_all.csv")
    print(STEP89_OUTPUT_DIR / "train_test_overall_summary_all.csv")
    print(STEP89_OUTPUT_DIR / "test_buy_strategy_vs_benchmark_all.csv")
    print(STEP89_OUTPUT_DIR / "test_sell_strategy_vs_benchmark_all.csv")


if __name__ == "__main__":
    main()
