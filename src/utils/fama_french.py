"""
Fama-French Factor Analysis Module

Downloads FF3/FF5 factors from Ken French's data library,
runs OLS regressions with Newey-West standard errors,
and computes rolling alpha / factor attribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import os
import warnings

logger = logging.getLogger(__name__)

# Ken French CSV URLs (hosted at Dartmouth)
FF3_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
           "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip")
FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
           "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")


def _download_ff_factors(url: str, cache_dir: str = "data") -> pd.DataFrame:
    """Download and parse Fama-French factor CSV from Ken French's site."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(url).replace(".zip", ".csv"))

    if os.path.exists(cache_file):
        logger.info(f"Loading cached FF factors from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    try:
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                            start="2000-01-01")[0]
        ff = ff / 100.0  # Convert from percent
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
        ff.to_csv(cache_file)
        logger.info(f"Downloaded FF factors → {cache_file}")
        return ff
    except Exception:
        pass

    try:
        import io
        import zipfile
        import urllib.request
        logger.info(f"Downloading FF factors from {url}")
        response = urllib.request.urlopen(url, timeout=30)
        zf = zipfile.ZipFile(io.BytesIO(response.read()))
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            lines = f.read().decode('utf-8').split('\n')

        # Find data start (skip header lines)
        start_idx = 0
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) >= 4 and parts[0].strip().isdigit():
                start_idx = i
                break

        data_lines = []
        for line in lines[start_idx:]:
            parts = line.strip().split(',')
            if len(parts) >= 4 and len(parts[0].strip()) == 8:
                data_lines.append(parts)

        df = pd.DataFrame(data_lines, columns=['Date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
        df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%Y%m%d')
        df.set_index('Date', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce') / 100.0
        df.dropna(inplace=True)
        df.to_csv(cache_file)
        logger.info(f"Downloaded FF factors → {cache_file}")
        return df
    except Exception as e:
        logger.warning(f"Could not download FF factors: {e}")
        return pd.DataFrame()


class FamaFrenchAnalyzer:
    """
    Run Fama-French 3-factor regressions on strategy returns.

    R_strategy - R_f = α + β₁(R_m - R_f) + β₂·SMB + β₃·HML + ε
    """

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        self.ff_factors: Optional[pd.DataFrame] = None

    def load_factors(self) -> pd.DataFrame:
        """Load FF3 factors (downloads if needed)."""
        if self.ff_factors is not None and not self.ff_factors.empty:
            return self.ff_factors
        self.ff_factors = _download_ff_factors(FF3_URL, self.cache_dir)
        return self.ff_factors

    def run_regression(self, strategy_returns: pd.Series,
                       n_lags: int = 5) -> Dict:
        """
        Run FF3 regression with Newey-West standard errors.

        Args:
            strategy_returns: daily returns (DatetimeIndex)
            n_lags: Newey-West lag truncation

        Returns:
            Dict with alpha, betas, t-stats, p-values, R².
        """
        import statsmodels.api as sm

        ff = self.load_factors()
        if ff.empty:
            logger.warning("FF factors unavailable — returning empty results")
            return self._empty_result()

        # Align dates
        combined = pd.DataFrame({'strategy': strategy_returns})
        combined = combined.join(ff, how='inner')
        combined.dropna(inplace=True)

        if len(combined) < 30:
            logger.warning(f"Only {len(combined)} overlapping observations — need ≥30")
            return self._empty_result()

        # Excess returns
        y = combined['strategy'] - combined['RF']
        X = combined[['Mkt-RF', 'SMB', 'HML']]
        X = sm.add_constant(X)

        # OLS with Newey-West HAC standard errors
        model = sm.OLS(y, X).fit(cov_type='HAC',
                                  cov_kwds={'maxlags': n_lags})

        alpha_daily = model.params['const']
        alpha_annual = alpha_daily * 252

        result = {
            'alpha_daily': alpha_daily,
            'alpha_annualized': alpha_annual,
            'alpha_tstat': model.tvalues['const'],
            'alpha_pvalue': model.pvalues['const'],
            'beta_market': model.params['Mkt-RF'],
            'beta_smb': model.params['SMB'],
            'beta_hml': model.params['HML'],
            'tstat_market': model.tvalues['Mkt-RF'],
            'tstat_smb': model.tvalues['SMB'],
            'tstat_hml': model.tvalues['HML'],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'n_observations': len(combined),
            'model': model,
        }

        logger.info(f"FF3 Alpha: {alpha_annual:.4f} (annualized), "
                     f"t={model.tvalues['const']:.2f}, p={model.pvalues['const']:.4f}")

        return result

    def rolling_alpha(self, strategy_returns: pd.Series,
                      window: int = 252) -> pd.Series:
        """Compute rolling FF3 alpha."""
        import statsmodels.api as sm

        ff = self.load_factors()
        if ff.empty:
            return pd.Series(dtype=float)

        combined = pd.DataFrame({'strategy': strategy_returns}).join(ff, how='inner').dropna()
        alphas = pd.Series(index=combined.index[window:], dtype=float)

        for i in range(window, len(combined)):
            window_data = combined.iloc[i - window:i]
            y = window_data['strategy'] - window_data['RF']
            X = sm.add_constant(window_data[['Mkt-RF', 'SMB', 'HML']])
            try:
                model = sm.OLS(y, X).fit()
                alphas.iloc[i - window] = model.params['const'] * 252
            except Exception:
                alphas.iloc[i - window] = np.nan

        return alphas

    @staticmethod
    def _empty_result() -> Dict:
        return {k: 0.0 for k in [
            'alpha_daily', 'alpha_annualized', 'alpha_tstat', 'alpha_pvalue',
            'beta_market', 'beta_smb', 'beta_hml', 'tstat_market',
            'tstat_smb', 'tstat_hml', 'r_squared', 'adj_r_squared',
            'n_observations'
        ]}
