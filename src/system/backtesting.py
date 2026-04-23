"""
Backtesting Module — Event-Driven Engine

Publication-grade backtesting with:
- Event-driven architecture (Bar → Signal → Order → Fill)
- Walk-forward validation (rolling/expanding window)
- Kelly Criterion position sizing
- Realistic execution (slippage, commission, market impact)
- Round-trip trade lifecycle tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import warnings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Event Types
# ──────────────────────────────────────────────

class EventType(Enum):
    BAR = "bar"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


class SignalDirection(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class BarEvent:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_index: int


@dataclass
class SignalEvent:
    timestamp: datetime
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0


@dataclass
class OrderEvent:
    timestamp: datetime
    direction: SignalDirection
    quantity: int
    order_type: str = "market"
    limit_price: Optional[float] = None


@dataclass
class FillEvent:
    timestamp: datetime
    direction: SignalDirection
    quantity: int
    fill_price: float
    commission: float
    slippage_cost: float


@dataclass
class Trade:
    """Round-trip trade record."""
    entry_time: datetime
    entry_price: float
    direction: SignalDirection
    quantity: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    return_pct: float = 0.0
    holding_bars: int = 0
    is_open: bool = True


# ──────────────────────────────────────────────
# Kelly Criterion Position Sizer
# ──────────────────────────────────────────────

class KellyPositionSizer:
    """
    Kelly Criterion with fractional scaling.
    f* = (p * b - q) / b
    where p = win prob, b = win/loss ratio, q = 1-p
    """

    def __init__(self, kelly_fraction: float = 0.5,
                 max_position_pct: float = 0.25,
                 min_trades_for_kelly: int = 30):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_trades_for_kelly = min_trades_for_kelly
        self._trade_results: List[float] = []

    def update(self, trade_pnl: float):
        """Record a completed trade result."""
        self._trade_results.append(trade_pnl)

    def calculate_position_size(self, capital: float, price: float,
                                signal_strength: float = 1.0) -> int:
        """Return number of shares to trade."""
        if len(self._trade_results) < self.min_trades_for_kelly:
            # Fall back to fixed fractional (using max_position_pct)
            raw_dollars = capital * self.max_position_pct * abs(signal_strength)
        else:
            wins = [r for r in self._trade_results if r > 0]
            losses = [r for r in self._trade_results if r <= 0]
            if not wins or not losses:
                raw_dollars = capital * self.max_position_pct * abs(signal_strength)
            else:
                p = len(wins) / len(self._trade_results)
                q = 1.0 - p
                b = np.mean(wins) / abs(np.mean(losses))
                kelly_f = (p * b - q) / b if b > 0 else 0.0
                kelly_f = max(0.0, kelly_f) * self.kelly_fraction
                raw_dollars = capital * kelly_f * abs(signal_strength)

        # Cap at max position percentage
        raw_dollars = min(raw_dollars, capital * self.max_position_pct)
        shares = int(raw_dollars / price) if price > 0 else 0
        return max(0, shares)


# ──────────────────────────────────────────────
# Execution Model
# ──────────────────────────────────────────────

class ExecutionModel:
    """Realistic execution with slippage and commission."""

    def __init__(self, commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 market_impact_factor: float = 0.1):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.market_impact_factor = market_impact_factor

    def execute(self, order: OrderEvent, bar: BarEvent) -> FillEvent:
        """Simulate order execution against current bar."""
        base_price = bar.close

        # Slippage: proportional + sqrt(volume fraction) market impact
        volume_fraction = order.quantity / max(bar.volume, 1)
        impact = self.slippage_rate + self.market_impact_factor * np.sqrt(volume_fraction)

        if order.direction == SignalDirection.BUY:
            fill_price = base_price * (1.0 + impact)
        else:
            fill_price = base_price * (1.0 - impact)

        slippage_cost = abs(fill_price - base_price) * order.quantity
        commission = fill_price * order.quantity * self.commission_rate

        return FillEvent(
            timestamp=bar.timestamp,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage_cost,
        )


# ──────────────────────────────────────────────
# Portfolio Tracker
# ──────────────────────────────────────────────

class Portfolio:
    """Tracks cash, positions, equity curve, and round-trip trades."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0  # net shares held
        self.trades: List[Trade] = []
        self._open_trade: Optional[Trade] = None

        # Time series
        self.equity_curve: List[Dict] = []

    @property
    def closed_trades(self) -> List[Trade]:
        return [t for t in self.trades if not t.is_open]

    def process_fill(self, fill: FillEvent):
        """Update portfolio state from a fill event."""
        cost = fill.fill_price * fill.quantity
        total_cost = fill.commission + fill.slippage_cost

        if fill.direction == SignalDirection.BUY:
            self.cash -= (cost + total_cost)
            self.position += fill.quantity
            if self._open_trade is None:
                self._open_trade = Trade(
                    entry_time=fill.timestamp,
                    entry_price=fill.fill_price,
                    direction=fill.direction,
                    quantity=fill.quantity,
                )
        elif fill.direction == SignalDirection.SELL:
            self.cash += (cost - total_cost)
            self.position -= fill.quantity
            if self._open_trade is not None and self.position <= 0:
                self._close_trade(fill)

    def _close_trade(self, fill: FillEvent):
        """Close the current open trade."""
        if self._open_trade is None:
            return
        t = self._open_trade
        t.exit_time = fill.timestamp
        t.exit_price = fill.fill_price
        t.is_open = False
        if t.direction == SignalDirection.BUY:
            t.pnl = (t.exit_price - t.entry_price) * t.quantity
        else:
            t.pnl = (t.entry_price - t.exit_price) * t.quantity
        t.return_pct = t.pnl / (t.entry_price * t.quantity) if t.entry_price > 0 else 0
        self.trades.append(t)
        self._open_trade = None

    def snapshot(self, bar: BarEvent):
        """Record equity at current bar."""
        position_value = self.position * bar.close
        total_equity = self.cash + position_value
        self.equity_curve.append({
            'timestamp': bar.timestamp,
            'equity': total_equity,
            'cash': self.cash,
            'position': self.position,
            'position_value': position_value,
            'close_price': bar.close,
        })

    def get_equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self.equity_curve)
        return df.set_index('timestamp')['equity']

    def get_returns_series(self) -> pd.Series:
        eq = self.get_equity_series()
        return eq.pct_change().dropna()

    def get_trade_analysis(self) -> Dict:
        closed = self.closed_trades
        if not closed:
            return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                    'win_rate': 0, 'avg_pnl': 0, 'avg_return': 0,
                    'best_trade': 0, 'worst_trade': 0, 'avg_holding_bars': 0,
                    'profit_factor': 0}
        pnls = [t.pnl for t in closed]
        rets = [t.return_pct for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            'total_trades': len(closed),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(closed),
            'avg_pnl': np.mean(pnls),
            'avg_return': np.mean(rets),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'avg_holding_bars': np.mean([t.holding_bars for t in closed]),
            'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        }


# ──────────────────────────────────────────────
# Main Backtesting Engine
# ──────────────────────────────────────────────

class BacktestingEngine:
    """
    Event-driven backtesting engine with walk-forward validation.

    Usage:
        engine = BacktestingEngine(config)
        results = engine.run_backtest(data, signals, start_date, end_date)

    Or with walk-forward:
        results = engine.run_walk_forward(data, signal_generator_fn,
                                          n_splits=5, train_pct=0.7)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission_rate = config.get('commission_rate', 0.001)
        self.slippage_rate = config.get('slippage_rate', 0.0005)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.kelly_fraction = config.get('kelly_fraction', 0.5)
        self.max_position_pct = config.get('max_position_pct', 0.25)

        self.execution_model = ExecutionModel(
            commission_rate=self.commission_rate,
            slippage_rate=self.slippage_rate,
            market_impact_factor=config.get('market_impact_factor', 0.1),
        )

    # ── Public API (backward-compatible) ──

    def run_backtest(self, data: pd.DataFrame, signals: Dict,
                     start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest on data with pre-computed signals dict.

        Args:
            data: OHLCV DataFrame (DatetimeIndex, columns: Open/High/Low/Close/Volume)
            signals: {bar_index: {'action': 'buy'|'sell'|'hold',
                       'signal_strength': float, 'confidence': float}}
            start_date: optional filter
            end_date: optional filter

        Returns:
            Dict with portfolio_values, performance_metrics, risk_metrics,
            drawdown_analysis, trade_analysis, summary.
        """
        logger.info(f"Running backtest (event-driven engine)")

        # Normalize columns
        data = self._normalize_columns(data.copy())

        # Date filter
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]

        # Convert legacy signals dict to SignalEvent list
        signal_events = self._legacy_signals_to_events(signals, data)

        # Run event loop
        portfolio = self._run_event_loop(data, signal_events)

        return self._compile_results(portfolio)

    def run_backtest_from_signal_series(self, data: pd.DataFrame,
                                        signal_series: pd.Series) -> Dict:
        """
        Run backtest from a signal Series (+1 buy, -1 sell, 0 hold).
        """
        data = self._normalize_columns(data.copy())
        signal_events = {}
        for i, (ts, sig) in enumerate(signal_series.items()):
            if sig > 0:
                direction = SignalDirection.BUY
            elif sig < 0:
                direction = SignalDirection.SELL
            else:
                direction = SignalDirection.HOLD
            signal_events[i] = SignalEvent(
                timestamp=ts, direction=direction,
                strength=abs(sig), confidence=0.8,
            )
        portfolio = self._run_event_loop(data, signal_events)
        return self._compile_results(portfolio)

    def run_walk_forward(self, data: pd.DataFrame,
                         signal_generator: Callable,
                         n_splits: int = 5,
                         train_pct: float = 0.7,
                         expanding: bool = False) -> Dict:
        """
        Walk-forward validation.

        Args:
            data: full OHLCV DataFrame
            signal_generator: fn(train_data, test_data) -> signal_series for test period
            n_splits: number of walk-forward folds
            train_pct: fraction used for training in each fold
            expanding: if True, training window expands; else rolls

        Returns:
            Combined results across all OOS folds.
        """
        logger.info(f"Walk-forward validation: {n_splits} splits, "
                     f"{'expanding' if expanding else 'rolling'} window")

        data = self._normalize_columns(data.copy())
        n = len(data)
        fold_size = n // n_splits

        all_oos_equity = []
        all_oos_returns = []
        fold_results = []

        for fold in range(n_splits):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n)

            if expanding:
                train_start = 0
            else:
                train_len = int(fold_size * train_pct / (1 - train_pct))
                train_start = max(0, test_start - train_len)

            if test_start <= train_start:
                continue

            train_data = data.iloc[train_start:test_start]
            test_data = data.iloc[test_start:test_end]

            if len(train_data) < 30 or len(test_data) < 10:
                continue

            logger.info(f"  Fold {fold+1}/{n_splits}: "
                        f"train [{train_data.index[0].date()} → {train_data.index[-1].date()}], "
                        f"test [{test_data.index[0].date()} → {test_data.index[-1].date()}]")

            try:
                signal_series = signal_generator(train_data, test_data)
                fold_result = self.run_backtest_from_signal_series(test_data, signal_series)
                fold_results.append(fold_result)
                oos_returns = fold_result.get('returns_series', pd.Series(dtype=float))
                all_oos_returns.append(oos_returns)
            except Exception as e:
                logger.warning(f"  Fold {fold+1} failed: {e}")
                continue

        if not all_oos_returns:
            logger.error("All walk-forward folds failed")
            return {'error': 'All folds failed'}

        combined_returns = pd.concat(all_oos_returns)
        combined_equity = (1 + combined_returns).cumprod() * self.initial_capital

        return {
            'combined_returns': combined_returns,
            'combined_equity': combined_equity,
            'fold_results': fold_results,
            'n_folds': len(fold_results),
            'performance_metrics': self._compute_performance_metrics(combined_returns),
            'risk_metrics': self._compute_risk_metrics(combined_returns),
        }

    # ── Internal Event Loop ──

    def _run_event_loop(self, data: pd.DataFrame,
                        signal_events: Dict[int, SignalEvent]) -> Portfolio:
        """Core event-driven loop: Bar → Signal → Order → Fill."""
        portfolio = Portfolio(self.initial_capital)
        sizer = KellyPositionSizer(
            kelly_fraction=self.kelly_fraction,
            max_position_pct=self.max_position_pct,
        )

        for i, (timestamp, row) in enumerate(data.iterrows()):
            # 1. BAR EVENT
            bar = BarEvent(
                timestamp=timestamp,
                open=row.get('open', row['close']),
                high=row.get('high', row['close']),
                low=row.get('low', row['close']),
                close=row['close'],
                volume=row.get('volume', 0),
                bar_index=i,
            )

            # 2. SIGNAL EVENT
            if i in signal_events:
                sig = signal_events[i]
                if sig.direction == SignalDirection.HOLD:
                    pass
                elif sig.direction == SignalDirection.BUY and portfolio.position <= 0:
                    qty = sizer.calculate_position_size(
                        portfolio.cash, bar.close, sig.strength)
                    if qty > 0:
                        # 3. ORDER EVENT
                        order = OrderEvent(timestamp=timestamp,
                                           direction=SignalDirection.BUY,
                                           quantity=qty)
                        # 4. FILL EVENT
                        fill = self.execution_model.execute(order, bar)
                        if fill.fill_price * fill.quantity + fill.commission <= portfolio.cash:
                            portfolio.process_fill(fill)

                elif sig.direction == SignalDirection.SELL and portfolio.position > 0:
                    qty = min(portfolio.position,
                              sizer.calculate_position_size(
                                  portfolio.cash + portfolio.position * bar.close,
                                  bar.close, sig.strength))
                    if qty > 0:
                        order = OrderEvent(timestamp=timestamp,
                                           direction=SignalDirection.SELL,
                                           quantity=qty)
                        fill = self.execution_model.execute(order, bar)
                        portfolio.process_fill(fill)
                        sizer.update(portfolio.trades[-1].pnl if portfolio.trades else 0)

            # Update holding bars for open trade
            if portfolio._open_trade is not None:
                portfolio._open_trade.holding_bars += 1

            # Snapshot equity
            portfolio.snapshot(bar)

        return portfolio

    # ── Helpers ──

    def _normalize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
        return data

    def _legacy_signals_to_events(self, signals: Dict,
                                   data: pd.DataFrame) -> Dict[int, SignalEvent]:
        """Convert old-style signals dict to SignalEvent dict."""
        events = {}
        for i, (ts, _) in enumerate(data.iterrows()):
            if i in signals:
                s = signals[i]
                action = s.get('action', 'hold')
                if action == 'buy':
                    d = SignalDirection.BUY
                elif action == 'sell':
                    d = SignalDirection.SELL
                else:
                    d = SignalDirection.HOLD
                events[i] = SignalEvent(
                    timestamp=ts, direction=d,
                    strength=s.get('signal_strength', 1.0),
                    confidence=s.get('confidence', 0.5),
                )
        return events

    def _compile_results(self, portfolio: Portfolio) -> Dict:
        """Compile portfolio into backward-compatible results dict."""
        returns = portfolio.get_returns_series()
        equity = portfolio.get_equity_series()

        perf = self._compute_performance_metrics(returns)
        risk = self._compute_risk_metrics(returns)
        dd = self._compute_drawdown_analysis(equity)
        trade_analysis = portfolio.get_trade_analysis()

        portfolio_values = pd.DataFrame(portfolio.equity_curve)

        return {
            'portfolio_values': portfolio_values,
            'equity_series': equity,
            'returns_series': returns,
            'performance_metrics': perf,
            'risk_metrics': risk,
            'drawdown_analysis': dd,
            'trade_analysis': trade_analysis,
            'summary': self._generate_summary(perf, risk, trade_analysis),
        }

    def _compute_performance_metrics(self, returns: pd.Series) -> Dict:
        if returns.empty or len(returns) < 2:
            return {k: 0.0 for k in ['total_return', 'annualized_return',
                    'volatility', 'sharpe_ratio', 'sortino_ratio',
                    'calmar_ratio', 'max_drawdown', 'win_rate', 'profit_factor']}

        total_return = (1 + returns).prod() - 1
        n_days = len(returns)
        ann_return = (1 + total_return) ** (252 / n_days) - 1
        vol = returns.std() * np.sqrt(252)

        # Sharpe (correct annualisation)
        rf_daily = self.risk_free_rate / 252
        excess = returns - rf_daily
        sharpe = (excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Sortino (only negative returns in denominator)
        downside = returns[returns < 0]
        down_std = downside.std() * np.sqrt(252) if len(downside) > 1 else 1e-8
        sortino = (ann_return - self.risk_free_rate) / down_std if down_std > 0 else 0

        # Max drawdown
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = ((cum - peak) / peak)
        max_dd = dd.min()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        pos = returns[returns > 0]
        neg = returns[returns < 0]
        win_rate = len(pos) / len(returns)
        pf = pos.sum() / abs(neg.sum()) if len(neg) > 0 and neg.sum() != 0 else float('inf')

        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': pf,
        }

    def _compute_risk_metrics(self, returns: pd.Series) -> Dict:
        if returns.empty or len(returns) < 2:
            return {k: 0.0 for k in ['var_95', 'var_99', 'es_95', 'es_99',
                    'tail_ratio', 'skewness', 'kurtosis', 'downside_volatility']}

        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        es_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99

        down = returns[returns < 0]
        down_vol = down.std() * np.sqrt(252) if len(down) > 1 else 0

        return {
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99,
            'tail_ratio': var_95 / var_99 if var_99 != 0 else 0,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'downside_volatility': down_vol,
        }

    def _compute_drawdown_analysis(self, equity: pd.Series) -> Dict:
        if equity.empty:
            return {'max_drawdown': 0, 'max_drawdown_duration': 0,
                    'average_drawdown': 0, 'drawdown_series': np.array([])}

        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        dd_arr = dd.values

        # Max drawdown duration (bars)
        in_dd = dd_arr < 0
        durations = []
        cur = 0
        for v in in_dd:
            if v:
                cur += 1
            else:
                if cur > 0:
                    durations.append(cur)
                cur = 0

        return {
            'max_drawdown': dd_arr.min(),
            'max_drawdown_duration': max(durations) if durations else 0,
            'average_drawdown': dd_arr[dd_arr < 0].mean() if (dd_arr < 0).any() else 0,
            'drawdown_series': dd_arr,
        }

    def _generate_summary(self, perf: Dict, risk: Dict, trades: Dict) -> str:
        return f"""
=== Backtesting Summary (Event-Driven Engine) ===

Performance:
- Total Return: {perf['total_return']:.2%}
- Annualized Return: {perf['annualized_return']:.2%}
- Volatility: {perf['volatility']:.2%}
- Sharpe Ratio: {perf['sharpe_ratio']:.3f}
- Sortino Ratio: {perf['sortino_ratio']:.3f}
- Max Drawdown: {perf['max_drawdown']:.2%}

Risk:
- VaR (95%): {risk['var_95']:.2%}
- VaR (99%): {risk['var_99']:.2%}
- CVaR (95%): {risk['es_95']:.2%}
- Skewness: {risk['skewness']:.3f}
- Kurtosis: {risk['kurtosis']:.3f}

Trades:
- Total: {trades['total_trades']}
- Win Rate: {trades['win_rate']:.1%}
- Profit Factor: {trades['profit_factor']:.2f}
- Avg P&L: ${trades['avg_pnl']:.2f}
"""