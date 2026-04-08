from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


STOCK_FILES: Dict[str, str] = {
    "AMZN": "AMZN_5levels_train.csv",
    "GOOG": "GOOG_5levels_train.csv",
    "INTC": "INTC_5levels_train.csv",
    "MSFT": "MSFT_5levels_train.csv",
}

BASE_COLUMNS = ["BidPrice_1", "AskPrice_1", "BidSize_1", "AskSize_1"]


@dataclass(frozen=True)
class Step1Config:
    data_dir: Path
    session_date: str = "2024-01-01"
    resample_freq: str = "1s"


def load_single_stock(path: Path, session_date: str) -> pd.DataFrame:
    """Load one stock and keep the best quote fields needed for v1."""
    df = pd.read_csv(path, usecols=["Time", *BASE_COLUMNS]).copy()
    df["timestamp"] = pd.to_datetime(
        session_date + " " + df["Time"],
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # If multiple updates land in the same event timestamp, keep the latest book state.
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df["mid_price"] = (df["BidPrice_1"] + df["AskPrice_1"]) / 2.0

    keep_cols = ["timestamp", "BidPrice_1", "AskPrice_1", "BidSize_1", "AskSize_1", "mid_price"]
    return df[keep_cols].rename(
        columns={
            "BidPrice_1": "best_bid",
            "AskPrice_1": "best_ask",
            "BidSize_1": "bid_size",
            "AskSize_1": "ask_size",
        }
    )


def resample_to_seconds(stock_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Convert event data to second bars by taking the last valid quote in each second.
    Missing seconds are filled with the most recent valid book.
    """
    second_df = (
        stock_df.set_index("timestamp")
        .resample(freq)
        .last()
        .ffill()
        .dropna(subset=["best_bid", "best_ask"])
    )
    second_df.index.name = "timestamp"
    return second_df.reset_index()


def build_common_second_grid(second_level_data: Dict[str, pd.DataFrame], freq: str) -> pd.DatetimeIndex:
    starts = [df["timestamp"].min() for df in second_level_data.values()]
    ends = [df["timestamp"].max() for df in second_level_data.values()]

    common_start = max(starts).floor(freq)
    common_end = min(ends).floor(freq)

    if common_start > common_end:
        raise ValueError("The four stocks do not share an overlapping time range.")

    return pd.date_range(start=common_start, end=common_end, freq=freq, name="timestamp")


def align_to_common_grid(second_level_data: Dict[str, pd.DataFrame], grid: pd.DatetimeIndex) -> pd.DataFrame:
    aligned_frames = []

    for symbol, df in second_level_data.items():
        aligned = (
            df.set_index("timestamp")
            .reindex(grid)
            .ffill()
            .assign(stock=symbol)
            .reset_index()
        )
        aligned_frames.append(aligned)

    panel = pd.concat(aligned_frames, ignore_index=True)
    panel = panel[["timestamp", "stock", "best_bid", "best_ask", "bid_size", "ask_size", "mid_price"]]
    return panel.sort_values(["timestamp", "stock"]).reset_index(drop=True)


def pivot_mid_prices(aligned_panel: pd.DataFrame) -> pd.DataFrame:
    """Wide mid-price matrix, one row per second and one column per stock."""
    return (
        aligned_panel.pivot(index="timestamp", columns="stock", values="mid_price")
        .sort_index()
        .rename_axis(columns=None)
    )


def prepare_step1_panel(config: Step1Config, symbols: Iterable[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = list(symbols or STOCK_FILES.keys())

    raw_data = {
        symbol: load_single_stock(config.data_dir / STOCK_FILES[symbol], config.session_date)
        for symbol in symbols
    }
    second_level_data = {
        symbol: resample_to_seconds(df, config.resample_freq)
        for symbol, df in raw_data.items()
    }
    grid = build_common_second_grid(second_level_data, config.resample_freq)
    aligned_panel = align_to_common_grid(second_level_data, grid)
    mid_matrix = pivot_mid_prices(aligned_panel)
    return aligned_panel, mid_matrix


if __name__ == "__main__":
    cfg = Step1Config(data_dir=Path("."))
    aligned_panel, mid_matrix = prepare_step1_panel(cfg)

    print("Aligned panel shape:", aligned_panel.shape)
    print("Mid-price matrix shape:", mid_matrix.shape)
    print()
    print("Aligned panel sample:")
    print(aligned_panel.head(8).to_string(index=False))
    print()
    print("Mid-price matrix sample:")
    print(mid_matrix.head(5).to_string())
