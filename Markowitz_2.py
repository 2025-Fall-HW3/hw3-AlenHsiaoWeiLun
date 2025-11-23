"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=120, gamma=2, top_n=4):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma
        self.top_n = max(2, min(top_n, len(price.columns) - 1))

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            0.0, index=self.price.index, columns=self.price.columns, dtype=float
        )
        self.portfolio_weights.columns.name = "Symbol"

        short_lb, long_lb = 63, 252
        trend_lb = 200
        start_idx = max(self.lookback, long_lb + 5, short_lb + 5, trend_lb + 5)
        defensive = ["XLP", "XLV", "XLU"]

        price_short = self.price[assets].pct_change(short_lb).fillna(0.0)
        price_long = self.price[assets].pct_change(long_lb).fillna(0.0)
        rolling_ret = (
            self.returns[assets].rolling(self.lookback).mean().fillna(0.0)
        )
        rolling_vol = (
            self.returns[assets].rolling(self.lookback).std().replace(0, np.nan)
        )
        rolling_sharpe = (rolling_ret / rolling_vol).fillna(0.0)
        spy_prices = self.price[self.exclude]
        spy_returns = self.returns[self.exclude]

        for i in range(start_idx, len(self.price)):
            date = self.price.index[i]
            history = self.returns[assets].iloc[i - self.lookback : i]
            if history.empty:
                continue

            vol = history.std(ddof=1).replace(0, np.nan)
            if vol.isna().all():
                continue
            vol = vol.fillna(vol.median()).fillna(1.0)
            inv_vol = 1.0 / vol

            momentum = (
                0.55 * price_short.iloc[i]
                + 0.35 * price_long.iloc[i]
                + 0.10 * rolling_sharpe.iloc[i]
            ).fillna(0.0)
            momentum = momentum.replace([np.inf, -np.inf], 0.0)
            positive = momentum[momentum > 0]

            if positive.empty:
                candidates = pd.Index(defensive).intersection(assets)
                if len(candidates) == 0:
                    candidates = assets
                momentum_focus = pd.Series(1.0, index=candidates)
            else:
                ranked = positive.sort_values(ascending=False)
                candidates = ranked.head(self.top_n).index
                momentum_focus = positive.loc[candidates]
                if momentum_focus.sum() == 0:
                    momentum_focus = pd.Series(1.0, index=candidates)

            momentum_focus = momentum_focus / momentum_focus.sum()

            rp_weights = inv_vol.loc[candidates].replace(
                [np.inf, -np.inf], 0.0
            )
            if rp_weights.sum() == 0:
                rp_weights = pd.Series(1.0, index=candidates)
            rp_weights = rp_weights / rp_weights.sum()

            combined = 0.68 * momentum_focus + 0.32 * rp_weights
            combined = combined / combined.sum()

            regime = self._regime_adjustment(i, spy_prices, spy_returns)
            def_share = float(np.clip(0.15 + 0.45 * (1 - regime), 0.15, 0.6))
            off_share = 1.0 - def_share

            row_weights = pd.Series(0.0, index=self.price.columns, dtype=float)
            row_weights.loc[candidates] += combined.values * off_share

            def_tickers = [t for t in defensive if t in assets]
            if def_tickers:
                def_vol = inv_vol.loc[def_tickers].replace(
                    [np.inf, -np.inf], 0.0
                )
                if def_vol.sum() == 0:
                    def_vol = pd.Series(1.0, index=def_tickers)
                def_vol = def_vol / def_vol.sum()
                row_weights.loc[def_tickers] += def_share * def_vol.values

            cov = history.cov().fillna(0.0)
            cov_matrix = cov.reindex(index=assets, columns=assets).to_numpy()
            w_vec = row_weights.loc[assets].to_numpy()
            port_var = float(np.dot(w_vec, cov_matrix @ w_vec))
            port_vol = np.sqrt(max(port_var, 0.0))

            target_annual_vol = 0.08 + 0.08 * regime
            daily_target = target_annual_vol / np.sqrt(252)
            if port_vol > 0:
                vol_scale = min(1.0, daily_target / (port_vol + 1e-8))
            else:
                vol_scale = 1.0

            core_scale = 0.3 + 0.7 * regime
            total_scale = core_scale * vol_scale

            row_weights *= total_scale
            row_weights[self.exclude] = 0.0

            self.portfolio_weights.loc[date, self.price.columns] = (
                row_weights.values
            )

        self.portfolio_weights.loc[:, self.exclude] = 0.0

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)
    
    def _regime_adjustment(self, idx, spy_prices, spy_returns):
        """Determine how much capital to deploy based on market regime."""
        if idx <= 0:
            return 0.5

        short_ret = spy_prices.iloc[idx] / spy_prices.iloc[max(idx - 63, 0)] - 1
        long_ret = spy_prices.iloc[idx] / spy_prices.iloc[max(idx - 252, 0)] - 1
        vol = spy_returns.iloc[max(idx - 63, 0) : idx].std()

        ma_window = spy_prices.iloc[max(idx - 200, 0) : idx]
        if len(ma_window) < 50:
            trend_on = True
        else:
            trend_on = spy_prices.iloc[idx] >= ma_window.mean()

        trailing_prices = spy_prices.iloc[: idx + 1]
        if len(trailing_prices) > 0:
            rolling_max = trailing_prices.max()
            drawdown = (
                spy_prices.iloc[idx] / rolling_max - 1 if rolling_max > 0 else 0.0
            )
        else:
            drawdown = 0.0

        score = 0.0
        score += 0.5 * np.tanh(short_ret * 5)
        score += 0.3 * np.tanh(long_ret * 3)
        score += 0.2 * (1 if trend_on else -1)
        score -= 0.25 * np.tanh(max(vol - 0.015, 0) * 40)
        if drawdown < -0.15:
            score -= 0.4
        elif drawdown < -0.08:
            score -= 0.2

        base = 0.5 + 0.35 * score
        cap = min(1.0, 3.0 / (self.gamma + 1.0))
        return float(np.clip(base, 0.1, cap))

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
