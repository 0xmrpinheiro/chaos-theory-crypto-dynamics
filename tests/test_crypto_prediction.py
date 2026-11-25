import subprocess
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_prediction import (
    calculate_atr,
    clean_history,
    download_history,
    evaluate_holdout,
    forecast_crypto_prices,
    generate_chaos,
    sample_history,
)


ROOT = Path(__file__).resolve().parents[1]


class CryptoExperimentTests(TestCase):
    def test_atr_uses_latest_bars_and_captures_both_gaps(self):
        bars = pd.DataFrame({"High": [11, 12, 25, 11], "Low": [9, 10, 20, 8], "Close": [10, 11, 24, 9]})
        self.assertAlmostEqual(calculate_atr(bars, 2), 15)
        self.assertAlmostEqual(calculate_atr(bars, 4), 8.5)
        self.assertAlmostEqual(calculate_atr(bars.iloc[:1], 1), 2)
        with self.assertRaises(ValueError):
            calculate_atr(bars, 5)

    def test_validates_daily_sequence_including_weekends(self):
        bars = sample_history().iloc[:10]
        cleaned = clean_history(bars.iloc[::-1])
        pd.testing.assert_frame_equal(cleaned, bars, check_freq=False)
        self.assertIn(pd.Timestamp("2024-01-06"), cleaned.index)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            clean_history(bars.drop("2024-01-06"))
        with self.assertRaisesRegex(ValueError, "unique dates"):
            clean_history(pd.concat([bars, bars.iloc[:1]]))

    def test_historical_download_uses_daily_single_ticker_prices(self):
        bars = sample_history()
        start, end = date(2024, 1, 1), date(2024, 6, 29)
        with patch("yfinance.download", return_value=bars) as download:
            downloaded = download_history("BTC-USD", start, end)
        pd.testing.assert_frame_equal(downloaded, bars, check_freq=False)
        download.assert_called_once_with(
            "BTC-USD", start=start, end=end, interval="1d", auto_adjust=False,
            progress=False, threads=False, multi_level_index=False,
        )

    def test_holdout_fits_on_training_data_only(self):
        close = sample_history()["Close"]
        with patch("crypto_prediction.forecast_crypto_prices", return_value=np.arange(5)) as fit:
            result = evaluate_holdout(close, 5)
        pd.testing.assert_series_equal(fit.call_args.args[0], close.iloc[:-5])
        self.assertEqual(fit.call_args.kwargs["steps"], 5)
        np.testing.assert_array_equal(result.actual, close.iloc[-5:])
        np.testing.assert_array_equal(result.last_close, np.full(5, close.iloc[-6]))
        pd.testing.assert_index_equal(result.dates, close.index[-5:])

    def test_flat_series_returns_flat_forecast(self):
        np.testing.assert_array_equal(forecast_crypto_prices(np.full(100, 42.0)), np.full(5, 42.0))

    def test_logistic_map_uses_only_its_own_state(self):
        x = generate_chaos(0.41, 3.9, 45)
        y = generate_chaos(0.410001, 3.9, 45)
        np.testing.assert_allclose(x[1:], 3.9 * x[:-1] * (1 - x[:-1]))
        self.assertGreater(np.max(np.abs(x - y)), 0.02)

    def test_offline_demo_writes_plot_and_labels_independent_outputs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "crypto-demo.png"
            result = self.run_cli("--sample", "--holdout", "20", "--plot", str(path))
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("synthetic OHLC sample (not Bitcoin market data)", result)
        self.assertIn("ARIMA(1, 1, 1) forecast", result)
        self.assertIn("Independent logistic map (unitless, not a price forecast)", result)
        self.assertIn("Last training close: MAE=", result)
        self.assertNotIn("Trading Instruction", result)

    def test_csv_uses_the_supplied_input(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            sample_history().to_csv(path)
            result = self.run_cli("--csv", str(path), "--holdout", "20")
        self.assertIn("Source: CSV", result)
        self.assertIn("Fixed-origin holdout: 2024-06-09 to 2024-06-28", result)

    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "crypto_prediction.py"), *arguments], cwd=ROOT,
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout