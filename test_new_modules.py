"""Quick functional test of all new modules."""
import numpy as np
import pandas as pd
from src.system.backtesting import BacktestingEngine
from src.system.monte_carlo import MonteCarloEngine
from src.utils.statistical_tests import deflated_sharpe_ratio, minimum_track_record_length

# Generate synthetic OHLCV
np.random.seed(42)
n = 500
dates = pd.bdate_range('2020-01-01', periods=n)
price = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
data = pd.DataFrame({
    'open': price * (1 + np.random.normal(0, 0.002, n)),
    'high': price * (1 + abs(np.random.normal(0, 0.005, n))),
    'low': price * (1 - abs(np.random.normal(0, 0.005, n))),
    'close': price,
    'volume': np.random.randint(1000000, 5000000, n).astype(float),
}, index=dates)

# Simple signal
sma20 = data['close'].rolling(20).mean()
sma50 = data['close'].rolling(50).mean()
signal = pd.Series(0, index=dates)
signal[sma20 > sma50] = 1
signal[sma20 <= sma50] = -1
signal = signal.shift(1).fillna(0)

# Backtest
engine = BacktestingEngine({'initial_capital': 100000})
results = engine.run_backtest_from_signal_series(data, signal)
perf = results['performance_metrics']
ta = results['trade_analysis']
print(f"Backtest: Return={perf['total_return']:.2%}, Sharpe={perf['sharpe_ratio']:.3f}, MaxDD={perf['max_drawdown']:.2%}")
print(f"Trades: {ta['total_trades']}, Win Rate: {ta['win_rate']:.1%}")

# Monte Carlo
mc = MonteCarloEngine(n_simulations=1000, seed=42)
returns = results['returns_series']
equity = results['equity_series']
gbm = mc.simulate_gbm(returns, initial_value=100000)
bootstrap = mc.block_bootstrap_sharpe(returns, perf['sharpe_ratio'])
ranking = mc.strategy_percentile(equity, gbm)
print(f"MC: percentile={ranking['percentile']:.1f}th, bootstrap p={bootstrap['p_value']:.4f}")

# DSR
dsr = deflated_sharpe_ratio(perf['sharpe_ratio'], len(returns), n_trials=10,
                             skewness=float(returns.skew()), kurtosis=float(returns.kurtosis()))
print(f"DSR: p={dsr['p_value']:.4f}, significant={dsr['is_significant']}")

# MinTRL
mtrl = minimum_track_record_length(perf['sharpe_ratio'])
print(f"MinTRL: {mtrl['min_trl']} days ({mtrl['min_years']:.1f} years)")

# Regime detection
from src.models.regime_detection import RegimeDetector
rd = RegimeDetector(n_regimes=2)
regime_result = rd.fit(returns)
print(f"Regime detection: {len(set(regime_result['regime_labels']))} regimes detected")
print(f"Current regime: {rd.current_regime()}")

print("\n=== ALL TESTS PASSED ===")
