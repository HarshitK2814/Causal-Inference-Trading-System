"""
Regime Detection Module

HMM-based market regime identification:
- Gaussian HMM with 2-3 states (bull/bear/sideways)
- Regime-conditional statistics
- Transition probability matrix
- Integration with backtester for regime-aware strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import warnings

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Hidden Markov Model regime detection.

    Usage:
        detector = RegimeDetector(n_regimes=3)
        result = detector.fit(returns, volatility)
        current = detector.current_regime()
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.regime_labels: Optional[np.ndarray] = None
        self.regime_probs: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, returns: pd.Series,
            volatility: Optional[pd.Series] = None) -> Dict:
        """
        Fit Gaussian HMM on returns (+ optional volatility).

        Args:
            returns: daily return series
            volatility: optional rolling volatility series

        Returns:
            Dict with regimes, statistics, transition matrix.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — using manual GMM fallback")
            return self._gmm_fallback(returns)

        # Prepare features
        if volatility is not None:
            X = np.column_stack([returns.values, volatility.values])
        else:
            X = returns.values.reshape(-1, 1)

        # Remove NaN
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]

        logger.info(f"Fitting {self.n_regimes}-state Gaussian HMM on "
                     f"{len(X_clean)} observations")

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
            tol=1e-4,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_clean)

        self.model = model
        self.regime_labels = np.full(len(returns), -1, dtype=int)
        self.regime_labels[mask] = model.predict(X_clean)
        self.regime_probs = np.zeros((len(returns), self.n_regimes))
        self.regime_probs[mask] = model.predict_proba(X_clean)
        self._fitted = True

        # Sort regimes by mean return (0=bear, 1=sideways, 2=bull)
        regime_means = []
        for r in range(self.n_regimes):
            r_mask = self.regime_labels == r
            regime_means.append(returns[r_mask].mean())
        sort_order = np.argsort(regime_means)
        label_map = {old: new for new, old in enumerate(sort_order)}
        self.regime_labels = np.array([label_map.get(l, l) for l in self.regime_labels])

        # Compute statistics
        stats = self._compute_regime_stats(returns)
        trans_matrix = model.transmat_

        # Reorder transition matrix
        trans_matrix = trans_matrix[sort_order][:, sort_order]

        regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'} if self.n_regimes == 3 else \
                       {0: 'Bear', 1: 'Bull'}

        logger.info("Regime statistics:")
        for r, name in regime_names.items():
            if r in stats:
                s = stats[r]
                logger.info(f"  {name}: μ={s['mean_return']:.5f}, "
                             f"σ={s['volatility']:.5f}, "
                             f"duration={s['avg_duration']:.0f} days, "
                             f"freq={s['frequency']:.1%}")

        # AIC: estimate number of free parameters
        n_features = X_clean.shape[1]
        n_params = (self.n_regimes * n_features +
                    self.n_regimes * n_features * (n_features + 1) // 2 +
                    self.n_regimes * (self.n_regimes - 1) +
                    self.n_regimes - 1)
        ll = model.score(X_clean) * len(X_clean)
        aic_val = -2 * ll + 2 * n_params

        return {
            'regime_labels': self.regime_labels,
            'regime_probs': self.regime_probs,
            'regime_stats': stats,
            'transition_matrix': trans_matrix,
            'regime_names': regime_names,
            'log_likelihood': model.score(X_clean),
            'aic': aic_val,
        }

    def current_regime(self) -> int:
        """Return the most recent regime label."""
        if self.regime_labels is None:
            raise ValueError("Model not fitted yet")
        return int(self.regime_labels[-1])

    def current_regime_probability(self) -> np.ndarray:
        """Return probability distribution over regimes for most recent bar."""
        if self.regime_probs is None:
            raise ValueError("Model not fitted yet")
        return self.regime_probs[-1]

    def _compute_regime_stats(self, returns: pd.Series) -> Dict:
        """Per-regime statistics."""
        stats = {}
        for r in range(self.n_regimes):
            mask = self.regime_labels == r
            r_returns = returns[mask]
            if len(r_returns) == 0:
                continue

            # Compute average duration (consecutive bars in regime)
            durations = []
            cur = 0
            for v in mask:
                if v:
                    cur += 1
                else:
                    if cur > 0:
                        durations.append(cur)
                    cur = 0
            if cur > 0:
                durations.append(cur)

            stats[r] = {
                'mean_return': r_returns.mean(),
                'volatility': r_returns.std(),
                'sharpe': r_returns.mean() / r_returns.std() * np.sqrt(252) if r_returns.std() > 0 else 0,
                'max_drawdown': self._simple_max_dd(r_returns),
                'frequency': mask.mean(),
                'n_observations': int(mask.sum()),
                'avg_duration': np.mean(durations) if durations else 0,
                'skewness': r_returns.skew(),
            }
        return stats

    @staticmethod
    def _simple_max_dd(returns: pd.Series) -> float:
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        return dd.min() if len(dd) > 0 else 0.0

    def _gmm_fallback(self, returns: pd.Series) -> Dict:
        """Fallback using sklearn GMM when hmmlearn unavailable."""
        from sklearn.mixture import GaussianMixture

        X = returns.dropna().values.reshape(-1, 1)
        gm = GaussianMixture(n_components=self.n_regimes,
                              random_state=self.random_state, max_iter=200)
        gm.fit(X)
        labels = gm.predict(X)
        probs = gm.predict_proba(X)

        self.regime_labels = labels
        self.regime_probs = probs
        self._fitted = True

        stats = self._compute_regime_stats(returns.dropna())

        return {
            'regime_labels': labels,
            'regime_probs': probs,
            'regime_stats': stats,
            'transition_matrix': None,
            'regime_names': {i: f'Regime_{i}' for i in range(self.n_regimes)},
            'note': 'GMM fallback (no temporal dynamics)',
        }
