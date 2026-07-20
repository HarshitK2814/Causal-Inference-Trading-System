"""
Metrics Module

Comprehensive metrics calculation for causal trading analysis:
- Correct Sharpe (√252 annualisation), Sortino (downside-only denominator)
- CVaR at 95%/99%
- Information Coefficient, Turnover
- Causal-specific metrics (no more hardcoded stubs)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for causal trading analysis.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.trading_days_per_year = config.get('trading_days_per_year', 252)

    def calculate_performance_metrics(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        """
        logger.info("Calculating performance metrics...")

        metrics = {}

        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        n_days = len(returns)
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (self.trading_days_per_year / n_days) - 1
        metrics['volatility'] = returns.std() * np.sqrt(self.trading_days_per_year)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns)
        metrics['information_ratio'] = self._calculate_information_ratio(returns, benchmark_returns)

        # Additional metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
        metrics['win_rate'] = self._calculate_win_rate(returns)
        metrics['profit_factor'] = self._calculate_profit_factor(returns)
        metrics['expectancy'] = self._calculate_expectancy(returns)

        # Advanced metrics
        metrics['var_95'] = self._calculate_var(returns, 0.05)
        metrics['var_99'] = self._calculate_var(returns, 0.01)
        metrics['expected_shortfall_95'] = self._calculate_expected_shortfall(returns, 0.05)
        metrics['expected_shortfall_99'] = self._calculate_expected_shortfall(returns, 0.01)

        # New: Information Coefficient and Turnover
        metrics['daily_turnover'] = 0.0  # Requires position data; set by caller

        return metrics

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        logger.info("Calculating risk metrics...")
        metrics = {}

        downside_returns = returns[returns < 0]
        metrics['downside_deviation'] = downside_returns.std() * np.sqrt(self.trading_days_per_year) if len(downside_returns) > 1 else 0.0

        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        metrics['tail_ratio'] = self._calculate_tail_ratio(returns)

        drawdowns = self._calculate_drawdowns(returns)
        metrics['max_drawdown'] = drawdowns.min()
        metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0
        metrics['max_drawdown_duration'] = self._calculate_max_drawdown_duration(drawdowns)

        metrics['concentration_risk'] = self._calculate_concentration_risk(returns)

        return metrics

    def calculate_information_coefficient(self, signals: pd.Series,
                                           forward_returns: pd.Series) -> float:
        """
        Rank IC: Spearman correlation between signal and forward returns.
        """
        aligned = pd.DataFrame({'signal': signals, 'fwd_ret': forward_returns}).dropna()
        if len(aligned) < 10:
            return 0.0
        return aligned['signal'].corr(aligned['fwd_ret'], method='spearman')

    def calculate_turnover(self, positions: pd.Series) -> float:
        """Daily portfolio turnover: mean |Δposition| / mean |position|."""
        delta = positions.diff().abs()
        avg_pos = positions.abs().mean()
        if avg_pos == 0:
            return 0.0
        return delta.mean() / avg_pos

    def calculate_causal_metrics(self, returns: pd.Series, signals: Dict,
                                  causal_effects: Dict) -> Dict[str, float]:
        """Calculate causal-specific metrics — actual computations, no stubs."""
        logger.info("Calculating causal metrics...")
        metrics = {}

        metrics['causal_signal_effectiveness'] = self._calculate_causal_signal_effectiveness(
            returns, signals, causal_effects)
        metrics['causal_relationship_strength'] = self._calculate_causal_relationship_strength(
            causal_effects)
        metrics['uncertainty_impact'] = self._calculate_uncertainty_impact(returns, signals)

        # Causal consistency: fraction of signals whose direction matched forward return
        if signals:
            correct = 0
            total = 0
            for idx, sig in signals.items():
                if isinstance(idx, int) and idx < len(returns):
                    total += 1
                    sig_dir = 1 if sig.get('action') == 'buy' else (-1 if sig.get('action') == 'sell' else 0)
                    actual_dir = 1 if returns.iloc[idx] > 0 else -1
                    if sig_dir != 0 and sig_dir == actual_dir:
                        correct += 1
            metrics['causal_consistency'] = correct / total if total > 0 else 0.0
        else:
            metrics['causal_consistency'] = 0.0

        return metrics

    def calculate_bandit_metrics(self, returns: pd.Series, signals: Dict,
                                  arm_performance: Dict) -> Dict[str, float]:
        """Calculate contextual bandit metrics."""
        logger.info("Calculating bandit metrics...")
        metrics = {}

        metrics['exploration_rate'] = self._calculate_exploration_rate(signals)
        metrics['exploitation_efficiency'] = self._calculate_exploitation_efficiency(arm_performance)
        metrics['cumulative_regret'] = self._calculate_cumulative_regret(returns, signals, arm_performance)
        metrics['arm_diversity'] = self._calculate_arm_diversity(signals)
        metrics['learning_progress'] = self._calculate_learning_progress(arm_performance)

        return metrics

    def calculate_uncertainty_metrics(self, returns: pd.Series,
                                       uncertainty_estimates: Dict) -> Dict[str, float]:
        """Calculate uncertainty-related metrics with actual computations."""
        logger.info("Calculating uncertainty metrics...")
        metrics = {}

        # Calibration: check if actual returns fall within predicted intervals
        metrics['uncertainty_calibration'] = self._calculate_uncertainty_calibration(
            returns, uncertainty_estimates)
        metrics['epistemic_uncertainty'] = self._calculate_epistemic_uncertainty(uncertainty_estimates)
        metrics['aleatoric_uncertainty'] = self._calculate_aleatoric_uncertainty(uncertainty_estimates)

        # Decision impact: correlation between uncertainty level and |return|
        if uncertainty_estimates:
            all_uncertainties = []
            for model_name, est in uncertainty_estimates.items():
                if 'uncertainty_decomposition' in est:
                    total = est['uncertainty_decomposition'].get('total_uncertainty', [])
                    if hasattr(total, '__len__') and len(total) > 0:
                        all_uncertainties.extend(list(total))
            if all_uncertainties and len(all_uncertainties) == len(returns):
                unc_series = pd.Series(all_uncertainties, index=returns.index)
                metrics['uncertainty_decision_impact'] = unc_series.corr(returns.abs())
            else:
                metrics['uncertainty_decision_impact'] = 0.0
        else:
            metrics['uncertainty_decision_impact'] = 0.0

        return metrics

    # ──────────────────────────────────────────
    # Core metric implementations
    # ──────────────────────────────────────────

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Sharpe = (mean_excess / std) × √252"""
        rf_daily = self.risk_free_rate / self.trading_days_per_year
        excess_returns = returns - rf_daily
        if returns.std() == 0:
            return 0.0
        return excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year)

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Sortino: excess return / downside deviation (√252 annualised)."""
        rf_daily = self.risk_free_rate / self.trading_days_per_year
        excess_mean = (returns - rf_daily).mean()
        downside = returns[returns < rf_daily] - rf_daily
        if len(downside) < 2:
            return float('inf') if excess_mean > 0 else 0.0
        downside_std = downside.std()
        if downside_std == 0:
            return float('inf') if excess_mean > 0 else 0.0
        return (excess_mean / downside_std) * np.sqrt(self.trading_days_per_year)

    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calmar = annualised return / |max drawdown|"""
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return float('inf')
        ann = (1 + returns).prod() ** (self.trading_days_per_year / len(returns)) - 1
        return ann / abs(max_dd)

    def _calculate_information_ratio(self, returns: pd.Series,
                                      benchmark_returns: Optional[pd.Series]) -> float:
        if benchmark_returns is None:
            return 0.0
        excess = returns - benchmark_returns
        if excess.std() == 0:
            return 0.0
        return excess.mean() / excess.std() * np.sqrt(self.trading_days_per_year)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        return ((cum - peak) / peak).min()

    def _calculate_win_rate(self, returns: pd.Series) -> float:
        return (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        pos = returns[returns > 0].sum()
        neg = abs(returns[returns < 0].sum())
        return pos / neg if neg > 0 else float('inf')

    def _calculate_expectancy(self, returns: pd.Series) -> float:
        return returns.mean()

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        return np.percentile(returns, confidence_level * 100)

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        var = self._calculate_var(returns, confidence_level)
        tail = returns[returns <= var]
        return tail.mean() if len(tail) > 0 else var

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        p95 = self._calculate_var(returns, 0.05)
        p99 = self._calculate_var(returns, 0.01)
        return p95 / p99 if p99 != 0 else 0

    def _calculate_drawdowns(self, returns: pd.Series) -> pd.Series:
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        return (cum - peak) / peak

    def _calculate_max_drawdown_duration(self, drawdowns: pd.Series) -> int:
        in_dd = drawdowns < 0
        periods = []
        cur = 0
        for v in in_dd:
            if v:
                cur += 1
            else:
                if cur > 0:
                    periods.append(cur)
                cur = 0
        return max(periods) if periods else 0

    def _calculate_concentration_risk(self, returns: pd.Series) -> float:
        return returns.std() / abs(returns.mean()) if returns.mean() != 0 else 0

    # ──────────────────────────────────────────
    # Causal / Bandit helpers
    # ──────────────────────────────────────────

    def _calculate_causal_signal_effectiveness(self, returns: pd.Series,
                                                signals: Dict, causal_effects: Dict) -> float:
        if not signals:
            return 0.0
        strengths = [s['signal_strength'] for s in signals.values()]
        if len(strengths) == len(returns):
            return np.corrcoef(strengths, returns)[0, 1]
        return 0.0

    def _calculate_causal_relationship_strength(self, causal_effects: Dict) -> float:
        if not causal_effects:
            return 0.0
        effects = []
        for _, ed in causal_effects.items():
            if isinstance(ed, dict):
                for _, v in ed.items():
                    if isinstance(v, (int, float)):
                        effects.append(abs(v))
        return np.mean(effects) if effects else 0.0

    def _calculate_uncertainty_impact(self, returns: pd.Series, signals: Dict) -> float:
        if not signals:
            return 0.0
        confs = [s['confidence'] for s in signals.values()]
        if len(confs) == len(returns):
            return np.corrcoef(confs, returns)[0, 1]
        return 0.0

    def _calculate_uncertainty_calibration(self, returns: pd.Series,
                                            uncertainty_estimates: Dict) -> float:
        """Fraction of returns within predicted ±1σ intervals."""
        if not uncertainty_estimates:
            return 0.0
        for model_name, est in uncertainty_estimates.items():
            if 'prediction_intervals' in est:
                lb = np.array(est['prediction_intervals'].get('lower_bounds', []))
                ub = np.array(est['prediction_intervals'].get('upper_bounds', []))
                if len(lb) > 0 and len(lb) <= len(returns):
                    actual = returns.values[:len(lb)]
                    covered = np.mean((actual >= lb) & (actual <= ub))
                    return float(covered)
        return 0.0

    def _calculate_epistemic_uncertainty(self, uncertainty_estimates: Dict) -> float:
        if not uncertainty_estimates:
            return 0.0
        vals = []
        for _, est in uncertainty_estimates.items():
            if 'uncertainty_decomposition' in est:
                ep = est['uncertainty_decomposition'].get('epistemic_uncertainty', [])
                if isinstance(ep, (list, np.ndarray)) and len(ep) > 0:
                    vals.extend(list(ep))
                elif isinstance(ep, (int, float)):
                    vals.append(ep)
        return float(np.mean(vals)) if vals else 0.0

    def _calculate_aleatoric_uncertainty(self, uncertainty_estimates: Dict) -> float:
        if not uncertainty_estimates:
            return 0.0
        vals = []
        for _, est in uncertainty_estimates.items():
            if 'uncertainty_decomposition' in est:
                al = est['uncertainty_decomposition'].get('aleatoric_uncertainty', [])
                if isinstance(al, (list, np.ndarray)) and len(al) > 0:
                    vals.extend(list(al))
                elif isinstance(al, (int, float)):
                    vals.append(al)
        return float(np.mean(vals)) if vals else 0.0

    def _calculate_exploration_rate(self, signals: Dict) -> float:
        if not signals:
            return 0.0
        explore = sum(1 for s in signals.values() if s.get('action') == 'explore')
        return explore / len(signals)

    def _calculate_exploitation_efficiency(self, arm_performance: Dict) -> float:
        if not arm_performance:
            return 0.0
        return float(np.mean(list(arm_performance.values())))

    def _calculate_cumulative_regret(self, returns: pd.Series,
                                      signals: Dict, arm_performance: Dict) -> float:
        if not signals or not arm_performance:
            return 0.0
        optimal = max(arm_performance.values()) if arm_performance else 0
        return optimal - returns.mean()

    def _calculate_arm_diversity(self, signals: Dict) -> float:
        if not signals:
            return 0.0
        actions = [s['action'] for s in signals.values()]
        counts = pd.Series(actions).value_counts()
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def _calculate_learning_progress(self, arm_performance: Dict) -> float:
        if not arm_performance:
            return 0.0
        perfs = list(arm_performance.values())
        if len(perfs) < 2:
            return 0.0
        x = np.arange(len(perfs))
        coeffs = np.polyfit(x, perfs, 1)
        return float(coeffs[0])