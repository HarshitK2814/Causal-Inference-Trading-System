"""
Statistical Tests for Backtest Validation

Publication-grade statistical validation:
1. Deflated Sharpe Ratio (DSR) — Bailey et al. (2014)
2. Probability of Backtest Overfitting (PBO) — Bailey et al. (2017)
3. Minimum Track Record Length (MinTRL)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from scipy import stats as sp_stats
import logging
import warnings

logger = logging.getLogger(__name__)


def deflated_sharpe_ratio(observed_sharpe: float,
                           n_observations: int,
                           n_trials: int,
                           skewness: float = 0.0,
                           kurtosis: float = 3.0,
                           risk_free_rate: float = 0.0) -> Dict:
    """
    Deflated Sharpe Ratio — Bailey & López de Prado (2014).

    Adjusts Sharpe for the number of parameter combinations / strategies
    tested (data-snooping correction).

    The expected maximum Sharpe under the null (all strategies have SR=0):
        E[max(SR)] ≈ √(2·ln(N)) · (1 - γ/(2·ln(N))) + γ/√(2·ln(N))

    where N = n_trials, γ = Euler-Mascheroni constant ≈ 0.5772

    Then DSR tests: H₀: SR* ≤ E[max(SR)]
    using the sampling distribution of Sharpe corrected for
    skewness and kurtosis (Lo, 2002).

    Args:
        observed_sharpe: the Sharpe ratio to test
        n_observations: number of return observations
        n_trials: number of independent strategy variants tested
        skewness: skewness of returns (γ₃)
        kurtosis: excess kurtosis of returns (γ₄, normal = 0)

    Returns:
        Dict with dsr, p_value, expected_max_sharpe, is_significant.
    """
    if n_trials <= 1:
        # No multiple-testing correction needed
        return {
            'dsr': observed_sharpe,
            'p_value': 1.0 - sp_stats.norm.cdf(observed_sharpe * np.sqrt(n_observations)),
            'expected_max_sharpe': 0.0,
            'is_significant': observed_sharpe > 0,
            'n_trials': n_trials,
        }

    euler_mascheroni = 0.5772156649

    # Expected maximum Sharpe under null (Bonferroni-like)
    ln_n = np.log(n_trials)
    if ln_n <= 0:
        ln_n = 1e-8
    e_max_sr = np.sqrt(2.0 * ln_n) * (1.0 - euler_mascheroni / (2.0 * ln_n)) + \
               euler_mascheroni / np.sqrt(2.0 * ln_n)

    # Standard error of Sharpe (Lo, 2002, adjusted for non-normality)
    # Var(SR) ≈ (1/T) * (1 + 0.5·SR² - γ₃·SR + (γ₄/4)·SR²)
    sr = observed_sharpe
    se_sr = np.sqrt(
        (1.0 / n_observations) *
        (1.0 + 0.5 * sr ** 2 - skewness * sr + (kurtosis / 4.0) * sr ** 2)
    )

    if se_sr == 0:
        se_sr = 1e-8

    # DSR test statistic
    dsr_stat = (observed_sharpe - e_max_sr) / se_sr

    # p-value (one-sided: is SR significantly > expected max?)
    p_value = 1.0 - sp_stats.norm.cdf(dsr_stat)

    significant = p_value < 0.05

    logger.info(f"DSR: observed SR={observed_sharpe:.3f}, "
                 f"E[max(SR)]={e_max_sr:.3f} (from {n_trials} trials), "
                 f"DSR stat={dsr_stat:.3f}, p={p_value:.4f}, "
                 f"{'✓ significant' if significant else '✗ NOT significant'}")

    return {
        'dsr': dsr_stat,
        'p_value': p_value,
        'expected_max_sharpe': e_max_sr,
        'is_significant': significant,
        'observed_sharpe': observed_sharpe,
        'n_trials': n_trials,
        'n_observations': n_observations,
        'se_sharpe': se_sr,
    }


def probability_of_backtest_overfitting(returns_matrix: np.ndarray,
                                          n_partitions: int = 16) -> Dict:
    """
    Probability of Backtest Overfitting (PBO) — Bailey et al. (2017).

    Uses combinatorial cross-validation:
    1. Split return series into S sub-samples
    2. For each combination of S/2 sub-samples (train), hold out rest (test)
    3. Select best strategy IS, measure its OOS rank
    4. PBO = fraction of combinations where best-IS strategy has negative OOS Sharpe

    Args:
        returns_matrix: (T × N) array where each column is a strategy variant
        n_partitions: number of time partitions S (must be even)

    Returns:
        Dict with pbo, logit_distribution, n_combinations.
    """
    T, N = returns_matrix.shape
    if N < 2:
        return {'pbo': 0.0, 'note': 'Need at least 2 strategy variants'}

    if n_partitions % 2 != 0:
        n_partitions += 1

    partition_size = T // n_partitions
    if partition_size < 5:
        n_partitions = max(2, T // 5)
        if n_partitions % 2 != 0:
            n_partitions = max(2, n_partitions - 1)
        partition_size = T // n_partitions

    logger.info(f"PBO: {N} strategies, {n_partitions} partitions, "
                 f"{partition_size} obs/partition")

    from itertools import combinations

    partition_indices = []
    for i in range(n_partitions):
        start = i * partition_size
        end = min(start + partition_size, T)
        partition_indices.append(list(range(start, end)))

    half = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), half))

    # Limit combinations for computational feasibility
    max_combos = min(len(all_combos), 500)
    if len(all_combos) > max_combos:
        rng = np.random.RandomState(42)
        combo_idx = rng.choice(len(all_combos), max_combos, replace=False)
        all_combos = [all_combos[i] for i in combo_idx]

    n_overfit = 0
    logits = []

    for combo in all_combos:
        test_partitions = set(range(n_partitions)) - set(combo)

        is_idx = []
        for p in combo:
            is_idx.extend(partition_indices[p])
        oos_idx = []
        for p in test_partitions:
            oos_idx.extend(partition_indices[p])

        if not is_idx or not oos_idx:
            continue

        # Compute IS Sharpe for each strategy
        is_sharpes = np.zeros(N)
        oos_sharpes = np.zeros(N)
        for j in range(N):
            is_ret = returns_matrix[is_idx, j]
            oos_ret = returns_matrix[oos_idx, j]
            is_std = is_ret.std()
            oos_std = oos_ret.std()
            is_sharpes[j] = is_ret.mean() / is_std * np.sqrt(252) if is_std > 0 else 0
            oos_sharpes[j] = oos_ret.mean() / oos_std * np.sqrt(252) if oos_std > 0 else 0

        best_is = np.argmax(is_sharpes)
        oos_of_best = oos_sharpes[best_is]

        if oos_of_best < 0:
            n_overfit += 1

        # Logit: rank of best-IS strategy OOS
        oos_rank = np.sum(oos_sharpes <= oos_of_best) / N
        if 0 < oos_rank < 1:
            logits.append(np.log(oos_rank / (1 - oos_rank)))

    pbo = n_overfit / len(all_combos) if all_combos else 0

    logger.info(f"PBO = {pbo:.2%} ({n_overfit}/{len(all_combos)} combinations overfit)")

    return {
        'pbo': pbo,
        'n_overfit': n_overfit,
        'n_combinations': len(all_combos),
        'logit_distribution': np.array(logits),
        'is_overfit': pbo > 0.50,
    }


def minimum_track_record_length(observed_sharpe: float,
                                  skewness: float = 0.0,
                                  kurtosis: float = 3.0,
                                  confidence: float = 0.95) -> Dict:
    """
    Minimum Track Record Length (MinTRL).

    How many observations are needed to confirm Sharpe ≠ 0 at given confidence?

    MinTRL = 1 + (1 - γ₃·SR + ((γ₄-1)/4)·SR²) / SR²

    multiplied by the squared critical value z²(α).

    Args:
        observed_sharpe: annualised Sharpe ratio
        skewness: return skewness (γ₃)
        kurtosis: return kurtosis (γ₄, normal=3)
        confidence: confidence level (default 0.95)

    Returns:
        Dict with min_trl (in daily observations), min_years.
    """
    if abs(observed_sharpe) < 1e-8:
        return {'min_trl': float('inf'), 'min_years': float('inf'),
                'observed_sharpe': observed_sharpe}

    # Convert annualised Sharpe to daily
    sr_daily = observed_sharpe / np.sqrt(252)

    z = sp_stats.norm.ppf(confidence)

    # MinTRL formula (Bailey & López de Prado)
    excess_kurtosis = kurtosis - 3.0
    numer = 1.0 - skewness * sr_daily + (excess_kurtosis / 4.0) * sr_daily ** 2
    min_trl = (1.0 + numer / (sr_daily ** 2)) * z ** 2

    min_trl = max(1, int(np.ceil(min_trl)))
    min_years = min_trl / 252

    logger.info(f"MinTRL: need {min_trl} daily obs ({min_years:.1f} years) "
                 f"to confirm SR={observed_sharpe:.3f} at {confidence:.0%} confidence")

    return {
        'min_trl': min_trl,
        'min_years': min_years,
        'observed_sharpe': observed_sharpe,
        'confidence': confidence,
        'z_critical': z,
    }


def run_all_tests(returns: pd.Series,
                   observed_sharpe: float,
                   n_trials: int = 1,
                   returns_matrix: Optional[np.ndarray] = None) -> Dict:
    """
    Convenience: run all statistical validation tests.

    Args:
        returns: daily returns series
        observed_sharpe: annualised Sharpe
        n_trials: number of strategy variants tested
        returns_matrix: (T×N) matrix for PBO (optional)

    Returns:
        Dict with dsr, pbo, min_trl results.
    """
    skew = float(returns.skew())
    kurt = float(returns.kurtosis()) + 3  # scipy uses excess, we need raw

    results = {
        'dsr': deflated_sharpe_ratio(
            observed_sharpe, len(returns), n_trials, skew, kurt - 3),
        'min_trl': minimum_track_record_length(observed_sharpe, skew, kurt),
    }

    if returns_matrix is not None and returns_matrix.shape[1] >= 2:
        results['pbo'] = probability_of_backtest_overfitting(returns_matrix)
    else:
        results['pbo'] = {'pbo': None, 'note': 'No strategy variants provided'}

    return results
