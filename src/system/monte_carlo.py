"""
Monte Carlo Simulation Engine

- GBM (Geometric Brownian Motion) path simulation (10,000 paths)
- Block bootstrap for Sharpe ratio p-value
- Strategy percentile ranking against MC distribution
- Confidence interval bands
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """
    Monte Carlo simulation for strategy validation.

    Usage:
        mc = MonteCarloEngine(n_simulations=10000)
        gbm = mc.simulate_gbm(historical_returns)
        bootstrap = mc.block_bootstrap_sharpe(historical_returns, observed_sharpe)
        ranking = mc.strategy_percentile(strategy_equity, gbm_paths)
    """

    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(seed)

    def simulate_gbm(self, returns: pd.Series,
                     n_days: Optional[int] = None,
                     initial_value: float = 100000) -> Dict:
        """
        Simulate n_simulations GBM price paths.

        dS = μ·S·dt + σ·S·dW

        Args:
            returns: historical daily returns to calibrate μ and σ
            n_days: path length (default: len(returns))
            initial_value: starting portfolio value

        Returns:
            Dict with paths array, quantiles, final_values, mu, sigma.
        """
        mu = returns.mean()
        sigma = returns.std()
        n_days = n_days or len(returns)

        logger.info(f"Simulating {self.n_simulations} GBM paths "
                     f"(μ={mu:.6f}, σ={sigma:.6f}, T={n_days} days)")

        dt = 1.0  # daily
        paths = np.zeros((self.n_simulations, n_days + 1))
        paths[:, 0] = initial_value

        # Vectorized GBM: S(t+1) = S(t) * exp((μ - σ²/2)dt + σ√dt·Z)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        Z = self.rng.standard_normal((self.n_simulations, n_days))

        for t in range(n_days):
            paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])

        # Quantile bands
        quantiles = {}
        for q in [5, 25, 50, 75, 95]:
            quantiles[q] = np.percentile(paths, q, axis=0)

        final_values = paths[:, -1]

        return {
            'paths': paths,
            'quantiles': quantiles,
            'final_values': final_values,
            'mu': mu,
            'sigma': sigma,
            'n_days': n_days,
            'initial_value': initial_value,
        }

    def block_bootstrap_sharpe(self, returns: pd.Series,
                                observed_sharpe: float,
                                block_size: int = 21,
                                annualization: float = 252) -> Dict:
        """
        Block bootstrap to test if observed Sharpe could be due to luck.

        Resamples return series in blocks (preserving autocorrelation),
        computes Sharpe for each bootstrap sample, and calculates p-value.

        Args:
            returns: historical daily returns
            observed_sharpe: the Sharpe ratio to test
            block_size: block length in days (default 21 ≈ 1 month)
            annualization: √252 for daily data

        Returns:
            Dict with bootstrapped_sharpes, p_value, confidence_interval.
        """
        n = len(returns)
        returns_arr = returns.values

        logger.info(f"Block bootstrap: {self.n_simulations} samples, "
                     f"block_size={block_size}")

        n_blocks = int(np.ceil(n / block_size))
        bootstrapped_sharpes = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            # Sample random block starting indices
            block_starts = self.rng.randint(0, n - block_size + 1, size=n_blocks)
            # Concatenate blocks
            sample = np.concatenate([
                returns_arr[s:s + block_size] for s in block_starts
            ])[:n]

            # Compute Sharpe
            sample_mean = sample.mean()
            sample_std = sample.std()
            if sample_std > 0:
                bootstrapped_sharpes[i] = (sample_mean / sample_std) * np.sqrt(annualization)
            else:
                bootstrapped_sharpes[i] = 0.0

        # p-value: fraction of bootstrap Sharpes >= observed
        p_value = np.mean(bootstrapped_sharpes >= observed_sharpe)

        # 95% CI for Sharpe
        ci_lower = np.percentile(bootstrapped_sharpes, 2.5)
        ci_upper = np.percentile(bootstrapped_sharpes, 97.5)

        logger.info(f"Observed Sharpe: {observed_sharpe:.3f}, "
                     f"Bootstrap p-value: {p_value:.4f}, "
                     f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        return {
            'bootstrapped_sharpes': bootstrapped_sharpes,
            'observed_sharpe': observed_sharpe,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_bootstrap_sharpe': bootstrapped_sharpes.mean(),
            'std_bootstrap_sharpe': bootstrapped_sharpes.std(),
        }

    def strategy_percentile(self, strategy_equity: pd.Series,
                             gbm_result: Dict) -> Dict:
        """
        Rank strategy final equity against MC distribution.

        Args:
            strategy_equity: actual equity curve
            gbm_result: output from simulate_gbm()

        Returns:
            Dict with percentile ranking and comparison stats.
        """
        strategy_final = strategy_equity.iloc[-1]
        mc_finals = gbm_result['final_values']

        percentile = np.mean(mc_finals <= strategy_final) * 100

        logger.info(f"Strategy final value: ${strategy_final:,.0f}, "
                     f"MC percentile: {percentile:.1f}th")

        return {
            'strategy_final_value': strategy_final,
            'mc_median_final': np.median(mc_finals),
            'mc_mean_final': np.mean(mc_finals),
            'percentile': percentile,
            'beats_random_pct': percentile,
            'mc_5th_pct': np.percentile(mc_finals, 5),
            'mc_95th_pct': np.percentile(mc_finals, 95),
        }

    def run_full_simulation(self, returns: pd.Series,
                             equity: pd.Series,
                             sharpe: float) -> Dict:
        """
        Convenience: run all MC analyses in one call.
        """
        gbm = self.simulate_gbm(returns, initial_value=equity.iloc[0])
        bootstrap = self.block_bootstrap_sharpe(returns, sharpe)
        ranking = self.strategy_percentile(equity, gbm)

        return {
            'gbm': gbm,
            'bootstrap': bootstrap,
            'ranking': ranking,
        }
