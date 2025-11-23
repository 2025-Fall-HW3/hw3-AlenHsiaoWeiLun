"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
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

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust=False)
    df[asset] = raw["Adj Close"]

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        assets = df.columns[df.columns != self.exclude]
        if len(assets) == 0:
            raise ValueError("No assets remain after excluding benchmark symbol.")

        equal_weight = np.full(len(assets), 1.0 / len(assets))
        self.portfolio_weights = pd.DataFrame(
            0.0, index=df.index, columns=df.columns, dtype=float
        )
        self.portfolio_weights.columns.name = "Symbol"
        self.portfolio_weights.loc[:, assets] = equal_weight
        self.portfolio_weights.loc[:, self.exclude] = 0.0

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            0.0, index=df.index, columns=df.columns, dtype=float
        )
        self.portfolio_weights.columns.name = "Symbol"

        """
        TODO: Complete Task 2 Below
        """
        eps = 1e-8
        asset_returns = df_returns[assets]

        for i in range(self.lookback + 1, len(df)):
            window = asset_returns.iloc[i - self.lookback : i]
            if window.empty:
                continue

            sigma = window.std(ddof=1)
            inv_sigma = 1.0 / sigma.replace(0, np.nan)
            inv_sigma = inv_sigma.replace([np.inf, -np.inf], np.nan)

            if inv_sigma.isna().all():
                weights = pd.Series(1.0 / len(assets), index=assets)
            else:
                inv_sigma = inv_sigma.fillna(0.0)
                scale = inv_sigma.sum()
                if scale <= eps:
                    weights = pd.Series(1.0 / len(assets), index=assets)
                else:
                    weights = inv_sigma / scale

            self.portfolio_weights.loc[df.index[i], assets] = weights.reindex(
                assets
            ).values

        self.portfolio_weights.loc[:, self.exclude] = 0.0
        """
        TODO: Complete Task 2 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            0.0, index=df.index, columns=df.columns, dtype=float
        )
        self.portfolio_weights.columns.name = "Symbol"

        asset_returns = df_returns[assets]

        for i in range(self.lookback + 1, len(df)):
            R_n = asset_returns.iloc[i - self.lookback : i]
            if R_n.empty:
                continue
            weights = self.mv_opt(R_n, self.gamma)
            self.portfolio_weights.loc[df.index[i], assets] = weights

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        # numerical safety: symmetrize + jitter
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma = Sigma + 1e-8 * np.eye(n)

        try:
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.start()
                with gp.Model(env=env, name="portfolio") as model:
                    w = model.addVars(n, lb=0.0, ub=1.0, name="w")

                    ret_term = gp.quicksum(mu[i] * w[i] for i in range(n))
                    risk_term = gp.quicksum(
                        Sigma[i, j] * w[i] * w[j]
                        for i in range(n)
                        for j in range(n)
                    )
                    model.setObjective(
                        ret_term - 0.5 * gamma * risk_term, gp.GRB.MAXIMIZE
                    )

                    model.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0)

                    model.optimize()

                    if model.status == gp.GRB.OPTIMAL:
                        return np.array([w[i].X for i in range(n)])
        except gp.GurobiError:
            pass
        except Exception:
            pass

        return self._fallback_solution(mu, Sigma, gamma)

    def _fallback_solution(self, mu, Sigma, gamma):
        n = len(mu)
        if gamma <= 0:
            weights = np.zeros(n)
            weights[np.argmax(mu)] = 1.0
            return weights

        try:
            Sigma_inv = np.linalg.pinv(Sigma)
            raw = Sigma_inv @ mu / max(gamma, 1e-8)
            raw = np.clip(raw, 0, None)
            total = raw.sum()
            if total <= 0:
                return np.full(n, 1.0 / n)
            return raw / total
        except np.linalg.LinAlgError:
            return np.full(n, 1.0 / n)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
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
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
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

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)
