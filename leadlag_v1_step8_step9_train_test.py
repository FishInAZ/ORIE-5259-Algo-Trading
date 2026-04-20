from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from leadlag_v1_step1_preprocessing import Step1Config, prepare_step1_panel
from leadlag_v1_step3_step4 import build_signal_panel, compute_equal_weight_signal


OUTPUT_DIR = Path("leadlag_v1_outputs")
STEP89_OUTPUT_DIR = OUTPUT_DIR / "step8_step9_train_test"
BACKTEST_OUTPUT_DIR = Path("backtest")
LOOKBACK_GRID = [1, 2, 3, 4, 5, 7, 10, 15]
THRESHOLD_QUANTILE_GRID = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
SELL_MAX_WAIT_GRID = [3, 5, 10, 15, 20, 30, 45, 59]
TRAIN_RATIO = 0.70


def prepare_target_panel(lookback_seconds: int, stock: str) -> pd.DataFrame:
    config = Step1Config()
    aligned_panel, mid_matrix = prepare_step1_panel(config)
    _, signals = compute_equal_weight_signal(mid_matrix, lookback_seconds)
    panel = build_signal_panel(aligned_panel, signals)
    panel = panel[panel["stock"] == stock].copy()
    panel["prev_best_bid"] = panel["best_bid"].shift(1)
    size_sum = panel["bid_size"] + panel["ask_size"]
    panel["book_imbalance"] = ((panel["bid_size"] - panel["ask_size"]) / size_sum.where(size_sum != 0)).fillna(0.0)
    panel["sell_confirm"] = (
        (panel["best_bid"] <= panel["prev_best_bid"]) & (panel["book_imbalance"] <= 0)
    ).fillna(False)
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


def build_strategy(
    panel: pd.DataFrame,
    side: str,
    theta: float,
    max_wait_seconds: int | None = None,
) -> pd.DataFrame:
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
            if max_wait_seconds is not None and group.loc[idx, "second_in_minute"] > max_wait_seconds:
                break

            signal_t = group.loc[idx, "signal"]
            passes_sell_confirmation = side != "SELL" or bool(group.loc[idx, "sell_confirm"])
            if pd.notna(signal_t) and trigger_condition(signal_t) and passes_sell_confirmation:
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
            if max_wait_seconds is not None:
                fallback_candidates = group[group["second_in_minute"] >= max_wait_seconds]
                exec_row = fallback_candidates.iloc[0] if not fallback_candidates.empty else group.iloc[-1]
            else:
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


def evaluate_side_params(
    panel: pd.DataFrame,
    side: str,
    theta_quantile: float | None = None,
    theta: float | None = None,
    max_wait_seconds: int | None = None,
) -> dict:
    if theta is None:
        if theta_quantile is None:
            raise ValueError("Either theta_quantile or theta must be provided.")
        theta = panel["signal"].quantile(theta_quantile)

    benchmark = build_benchmark(panel, side)
    strategy = build_strategy(panel, side, theta, max_wait_seconds=max_wait_seconds)
    comparison = compare_strategy_vs_benchmark(strategy, benchmark, side)

    return {
        "theta": theta,
        "benchmark": benchmark,
        "strategy": strategy,
        "cmp": comparison,
        "objective_bps": comparison["improvement_bps"].mean(),
        "max_wait_seconds": max_wait_seconds,
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
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tuning = []
    all_side_summary = []
    all_overall_summary = []
    all_test_buy = []
    all_test_sell = []

    for stock in ["AMZN", "GOOG", "INTC", "MSFT"]:
        tuning_rows = []
        best_buy_result = None
        best_sell_result = None
        best_buy_panel = None
        best_sell_panel = None
        train_minutes = None
        test_minutes = None

        for lookback_seconds in LOOKBACK_GRID:
            panel = prepare_target_panel(lookback_seconds, stock)
            if train_minutes is None or test_minutes is None:
                train_minutes, test_minutes = split_train_test_minutes(panel)

            train_panel = subset_by_minutes(panel, train_minutes)

            for theta_quantile in THRESHOLD_QUANTILE_GRID:
                buy_result = evaluate_side_params(train_panel, "BUY", theta_quantile=theta_quantile)
                tuning_rows.append(
                    {
                        "stock": stock,
                        "side": "BUY",
                        "lookback_seconds": lookback_seconds,
                        "threshold_quantile": theta_quantile,
                        "theta": buy_result["theta"],
                        "max_wait_seconds": "",
                        "train_average_improvement_bps": buy_result["objective_bps"],
                    }
                )
                if best_buy_result is None or buy_result["objective_bps"] > best_buy_result["objective_bps"]:
                    best_buy_result = buy_result | {
                        "lookback_seconds": lookback_seconds,
                        "threshold_quantile": theta_quantile,
                    }
                    best_buy_panel = panel

            for theta_quantile, sell_max_wait_seconds in product(THRESHOLD_QUANTILE_GRID, SELL_MAX_WAIT_GRID):
                sell_result = evaluate_side_params(
                    train_panel,
                    "SELL",
                    theta_quantile=theta_quantile,
                    max_wait_seconds=sell_max_wait_seconds,
                )
                tuning_rows.append(
                    {
                        "stock": stock,
                        "side": "SELL",
                        "lookback_seconds": lookback_seconds,
                        "threshold_quantile": theta_quantile,
                        "theta": sell_result["theta"],
                        "max_wait_seconds": sell_max_wait_seconds,
                        "train_average_improvement_bps": sell_result["objective_bps"],
                    }
                )
                if best_sell_result is None or sell_result["objective_bps"] > best_sell_result["objective_bps"]:
                    best_sell_result = sell_result | {
                        "lookback_seconds": lookback_seconds,
                        "threshold_quantile": theta_quantile,
                    }
                    best_sell_panel = panel

        if (
            best_buy_result is None
            or best_sell_result is None
            or best_buy_panel is None
            or best_sell_panel is None
            or train_minutes is None
            or test_minutes is None
        ):
            raise RuntimeError(f"Parameter search failed to produce a result for {stock}.")

        train_buy_panel = subset_by_minutes(best_buy_panel, train_minutes)
        train_sell_panel = subset_by_minutes(best_sell_panel, train_minutes)
        test_buy_panel = subset_by_minutes(best_buy_panel, test_minutes)
        test_sell_panel = subset_by_minutes(best_sell_panel, test_minutes)

        train_buy_eval = evaluate_side_params(
            train_buy_panel,
            "BUY",
            theta=best_buy_result["theta"],
        )
        train_sell_eval = evaluate_side_params(
            train_sell_panel,
            "SELL",
            theta=best_sell_result["theta"],
            max_wait_seconds=best_sell_result["max_wait_seconds"],
        )
        test_buy_eval = evaluate_side_params(
            test_buy_panel,
            "BUY",
            theta=best_buy_result["theta"],
        )
        test_sell_eval = evaluate_side_params(
            test_sell_panel,
            "SELL",
            theta=best_sell_result["theta"],
            max_wait_seconds=best_sell_result["max_wait_seconds"],
        )

        test_pooled_cmp = pd.concat(
            [
                test_buy_eval["cmp"].assign(side_eval="BUY"),
                test_sell_eval["cmp"].assign(side_eval="SELL"),
            ],
            ignore_index=True,
        )
        train_pooled_cmp = pd.concat(
            [
                train_buy_eval["cmp"].assign(side_eval="BUY"),
                train_sell_eval["cmp"].assign(side_eval="SELL"),
            ],
            ignore_index=True,
        )

        train_buy_summary = summarize_side(train_buy_eval["cmp"], "BUY", "train")
        train_sell_summary = summarize_side(train_sell_eval["cmp"], "SELL", "train")
        test_buy_summary = summarize_side(test_buy_eval["cmp"], "BUY", "test")
        test_sell_summary = summarize_side(test_sell_eval["cmp"], "SELL", "test")
        side_summary = pd.concat(
            [train_buy_summary, train_sell_summary, test_buy_summary, test_sell_summary],
            ignore_index=True,
        )

        overall_summary = pd.DataFrame(
            [
                {
                    "split": "train",
                    "stock": stock,
                    "trades": len(train_pooled_cmp),
                    "average_improvement": train_pooled_cmp["improvement"].mean(),
                    "median_improvement": train_pooled_cmp["improvement"].median(),
                    "hit_rate": (train_pooled_cmp["improvement"] > 0).mean(),
                    "average_improvement_bps": train_pooled_cmp["improvement_bps"].mean(),
                },
                {
                    "split": "test",
                    "stock": stock,
                    "trades": len(test_pooled_cmp),
                    "average_improvement": test_pooled_cmp["improvement"].mean(),
                    "median_improvement": test_pooled_cmp["improvement"].median(),
                    "hit_rate": (test_pooled_cmp["improvement"] > 0).mean(),
                    "average_improvement_bps": test_pooled_cmp["improvement_bps"].mean(),
                },
            ]
        )

        tuning_df = pd.DataFrame(tuning_rows).sort_values(
            ["side", "train_average_improvement_bps", "lookback_seconds", "threshold_quantile", "max_wait_seconds"],
            ascending=[True, False, True, True, True],
        )

        avg_bar_path, hist_path = make_test_plots(
            side_summary[side_summary["split"] == "test"],
            test_pooled_cmp,
            stock,
        )

        tuning_df.to_csv(STEP89_OUTPUT_DIR / f"train_grid_search_{stock}.csv", index=False)
        side_summary.to_csv(STEP89_OUTPUT_DIR / f"train_test_side_summary_{stock}.csv", index=False)
        overall_summary.to_csv(STEP89_OUTPUT_DIR / f"train_test_overall_summary_{stock}.csv", index=False)
        test_buy_eval["cmp"].to_csv(STEP89_OUTPUT_DIR / f"test_buy_strategy_vs_benchmark_{stock}.csv", index=False)
        test_sell_eval["cmp"].to_csv(STEP89_OUTPUT_DIR / f"test_sell_strategy_vs_benchmark_{stock}.csv", index=False)

        all_tuning.append(tuning_df)
        all_side_summary.append(side_summary)
        all_overall_summary.append(overall_summary)
        all_test_buy.append(test_buy_eval["cmp"])
        all_test_sell.append(test_sell_eval["cmp"])

        print("STEP 8 SELL SIDE")
        print("=" * 80)
        print(f"{stock}: sell logic implemented:")
        print("signal_t < -theta -> execute sell at t+1 best bid; if no trigger arrives early enough, force sell by the trained max_wait_seconds cutoff.")
        print()
        print("STEP 9 TRAIN / TEST SPLIT")
        print("=" * 80)
        print(f"target_stock = {stock}")
        print(f"train_ratio = {TRAIN_RATIO}")
        print(f"train_minutes = {len(train_minutes)}, test_minutes = {len(test_minutes)}")
        print(f"train_time_range = {train_minutes[0]} -> {train_minutes[-1]}")
        print(f"test_time_range = {test_minutes[0]} -> {test_minutes[-1]}")
        print()
        print("Best BUY train parameters:")
        print(
            pd.DataFrame(
                [
                    {
                        "stock": stock,
                        "side": "BUY",
                        "lookback_seconds": best_buy_result["lookback_seconds"],
                        "threshold_quantile": best_buy_result["threshold_quantile"],
                        "theta": best_buy_result["theta"],
                        "train_average_improvement_bps": train_buy_eval["objective_bps"],
                    }
                ]
            ).to_string(index=False)
        )
        print()
        print("Best SELL train parameters:")
        print(
            pd.DataFrame(
                [
                    {
                        "stock": stock,
                        "side": "SELL",
                        "lookback_seconds": best_sell_result["lookback_seconds"],
                        "threshold_quantile": best_sell_result["threshold_quantile"],
                        "theta": best_sell_result["theta"],
                        "max_wait_seconds": best_sell_result["max_wait_seconds"],
                        "train_average_improvement_bps": train_sell_eval["objective_bps"],
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

    combined_side_summary = pd.concat(all_side_summary, ignore_index=True)
    combined_side_summary[
        (combined_side_summary["split"] == "test") & (combined_side_summary["side"] == "BUY")
    ][["stock", "average_improvement"]].rename(
        columns={"stock": "stock_name", "average_improvement": "avg_improvement_per_trade"}
    ).to_csv(BACKTEST_OUTPUT_DIR / "leadlag_buy_executions.csv", index=False)
    combined_side_summary[
        (combined_side_summary["split"] == "test") & (combined_side_summary["side"] == "SELL")
    ][["stock", "average_improvement"]].rename(
        columns={"stock": "stock_name", "average_improvement": "avg_improvement_per_trade"}
    ).to_csv(BACKTEST_OUTPUT_DIR / "leadlag_sell_executions.csv", index=False)

    print("Combined saved files:")
    print(STEP89_OUTPUT_DIR / "train_grid_search_all.csv")
    print(STEP89_OUTPUT_DIR / "train_test_side_summary_all.csv")
    print(STEP89_OUTPUT_DIR / "train_test_overall_summary_all.csv")
    print(STEP89_OUTPUT_DIR / "test_buy_strategy_vs_benchmark_all.csv")
    print(STEP89_OUTPUT_DIR / "test_sell_strategy_vs_benchmark_all.csv")
    print(BACKTEST_OUTPUT_DIR / "leadlag_buy_executions.csv")
    print(BACKTEST_OUTPUT_DIR / "leadlag_sell_executions.csv")


if __name__ == "__main__":
    main()
