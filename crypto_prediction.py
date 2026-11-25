"""Explore daily crypto prices with ARIMA and a separate logistic-map demo."""

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


OHLC = ["Open", "High", "Low", "Close"]
ARIMA_ORDER = (1, 1, 1)
MIN_MODEL_ROWS = 30


def sample_history():
    """Generate deterministic daily OHLC bars, including weekends, for the offline demo."""
    rng = np.random.default_rng(73)
    shocks = rng.normal(size=180)
    volatility = np.where(np.arange(179) < 135, 0.009, 0.02)
    log_returns = volatility * (shocks[1:] + 0.45 * shocks[:-1]) + 0.0004
    close = 40_000 * np.exp(np.r_[0.0, np.cumsum(log_returns)])
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.008, len(close))),
            "Low": np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.008, len(close))),
            "Close": close,
        },
        index=pd.date_range("2024-01-01", periods=len(close), freq="D", name="Date"),
    )


def clean_history(frame):
    """Validate and sort one daily series; don't fill gaps with invented prices."""
    if frame is None or frame.empty:
        raise ValueError("No price bars were returned.")
    if missing := set(OHLC) - set(frame.columns):
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}.")
    bars = frame.loc[:, OHLC].apply(pd.to_numeric, errors="coerce")
    dates = pd.to_datetime(bars.index, format="ISO8601", errors="coerce", utc=True).normalize()
    if dates.isna().any() or dates.has_duplicates:
        raise ValueError("Price bars must have valid, unique dates in UTC.")
    bars.index = dates.tz_localize(None)
    bars = bars.sort_index()
    if not np.isfinite(bars.to_numpy()).all() or (bars <= 0).any().any():
        raise ValueError("OHLC prices must be positive and finite.")
    if (bars["High"] < bars[["Open", "Close", "Low"]].max(axis=1)).any() or (
        bars["Low"] > bars[["Open", "Close"]].min(axis=1)
    ).any():
        raise ValueError("Each bar's high and low must bound its open and close.")
    days = bars.index.to_numpy(dtype="datetime64[D]")
    if np.any(np.diff(days) != np.timedelta64(1, "D")):
        raise ValueError("Expected consecutive daily bars, including weekends; check for missing dates.")
    return bars


def download_history(ticker, start, end):
    from yfinance import download

    return clean_history(download(
        ticker, start=start, end=end, interval="1d", auto_adjust=False,
        progress=False, threads=False, multi_level_index=False,
    ))


def generate_chaos(x0, r, n):
    """Generate a unitless logistic-map trajectory, independent of prices."""
    if n < 1 or not 0 < x0 < 1 or not 0 < r <= 4:
        raise ValueError("Use n >= 1, 0 < x0 < 1, and 0 < r <= 4.")
    values = np.empty(n)
    values[0] = x0
    for i in range(1, n):
        values[i] = r * values[i - 1] * (1 - values[i - 1])
    return values


def calculate_atr(bars, window=14):
    """Average the latest true ranges, without wrapping the last close to day one."""
    if window < 1 or len(bars) < window:
        raise ValueError("ATR needs a positive window and at least that many bars.")
    previous_close = bars["Close"].shift()
    ranges = pd.concat([
        bars["High"] - bars["Low"],
        (bars["High"] - previous_close).abs(),
        (bars["Low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return float(ranges.iloc[-window:].mean())


def forecast_crypto_prices(close, steps=5):
    if len(close) < MIN_MODEL_ROWS or steps < 1:
        raise ValueError(f"ARIMA needs at least {MIN_MODEL_ROWS} closes and a positive horizon.")
    values = np.asarray(close, dtype=float)
    scale = values[-1]
    if np.all(values == scale):
        return np.full(steps, scale)
    # Fit on price levels in units of the last available close for numerical scaling.
    model = ARIMA(values / scale, order=ARIMA_ORDER).fit()
    return np.asarray(model.forecast(steps=steps)) * scale


@dataclass(frozen=True)
class HoldoutResult:
    dates: pd.Index
    actual: np.ndarray
    arima: np.ndarray
    last_close: np.ndarray


def evaluate_holdout(close, length):
    """Score a final block from a single origin; no holdout closes are used to fit."""
    if length < 1 or len(close) - length < MIN_MODEL_ROWS:
        raise ValueError(f"Leave at least {MIN_MODEL_ROWS} closes before the holdout.")
    train, test = close.iloc[:-length], close.iloc[-length:]
    return HoldoutResult(
        dates=test.index,
        actual=test.to_numpy(),
        arima=forecast_crypto_prices(train, steps=length),
        last_close=np.full(length, train.iloc[-1]),
    )


def save_plot(path, close, forecast, trajectory, perturbed):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    recent = close.iloc[-45:]
    figure, (price_ax, map_ax) = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained")
    price_ax.plot(np.arange(1 - len(recent), 1), recent, color="#2554a4", label="Input closes")
    price_ax.axvline(0, color="#777777", alpha=0.65, linewidth=1)
    price_ax.plot(np.arange(len(forecast) + 1), np.r_[recent.iloc[-1], forecast],
                  "o--", color="#b84d16", label="ARIMA forecast", markersize=4)
    price_ax.set(title="ARIMA on daily closes", xlabel="Day relative to last close", ylabel="Price")
    price_ax.legend(loc="best")

    map_ax.plot(trajectory, color="#2554a4", label="x₀ = 0.41", alpha=0.85)
    map_ax.plot(perturbed, color="#b84d16", label="x₀ = 0.410001", alpha=0.75, linestyle="--")
    map_ax.set(title="Independent logistic map (r = 3.9)", xlabel="Iteration",
               ylabel="Unitless state", ylim=(0, 1.05))
    map_ax.legend(loc="lower left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", action="store_true", help="Use the offline sample (default).")
    source.add_argument("--csv", type=Path, help="Read Date, Open, High, Low, and Close columns.")
    source.add_argument("--ticker", help="Download historical daily bars for one ticker, such as BTC-USD.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(timezone.utc).date(),
                        help="Exclusive download end date (UTC).")
    parser.add_argument("--steps", type=positive_int, default=5, help="Number of subsequent daily bars.")
    parser.add_argument("--atr-window", type=positive_int, default=14)
    parser.add_argument("--holdout", type=positive_int, help="Score a final block against the last training close.")
    parser.add_argument("--plot", type=Path, help="Write a PNG visualization.")
    args = parser.parse_args(argv)

    try:
        if args.csv:
            bars = clean_history(pd.read_csv(args.csv, index_col="Date"))
            label = f"CSV ({args.csv})"
        elif args.ticker:
            ticker = args.ticker.strip().upper()
            if not ticker or any(c.isspace() or c == "," for c in ticker):
                raise ValueError("Supply a single ticker such as BTC-USD.")
            if args.start >= args.end:
                raise ValueError("--end must be after --start.")
            bars = download_history(ticker, args.start, args.end)
            label = f"Yahoo Finance historical daily bars ({ticker})"
        else:
            bars = sample_history()
            label = "synthetic OHLC sample (not Bitcoin market data)"

        close = bars["Close"]
        atr = calculate_atr(bars, args.atr_window)
        score = evaluate_holdout(close, args.holdout) if args.holdout else None
        forecast = forecast_crypto_prices(close, args.steps)
        trajectory = generate_chaos(0.41, 3.9, 45)
        if args.plot:
            save_plot(args.plot, close, forecast, trajectory, generate_chaos(0.410001, 3.9, 45))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Source: {label}")
    print(f"Bars: {len(bars)} ({bars.index[0]:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d} UTC)")
    print(f"Recent ATR ({args.atr_window} bars; simple mean): {atr:.2f} ({100 * atr / close.iloc[-1]:.2f}% of last close)")
    print(f"ARIMA{ARIMA_ORDER} forecast for {args.steps} daily observations after the last bar:")
    print("  " + ", ".join(f"{value:.2f}" for value in forecast))
    print("Independent logistic map (unitless, not a price forecast); first five values:")
    print("  " + ", ".join(f"{value:.4f}" for value in trajectory[:5]))
    if score is not None:
        print(f"Fixed-origin holdout: {score.dates[0]:%Y-%m-%d} to {score.dates[-1]:%Y-%m-%d}")
        for name, predicted in [("ARIMA", score.arima), ("Last training close", score.last_close)]:
            errors = score.actual - predicted
            print(f"  {name}: MAE={np.abs(errors).mean():.2f}, RMSE={np.sqrt(np.square(errors).mean()):.2f}")
    if args.plot:
        print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()