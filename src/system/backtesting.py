"""
Backtesting Module

This module provides comprehensive backtesting capabilities for causal trading strategies,
including performance metrics, risk analysis, and strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Comprehensive backtesting engine for causal trading strategies.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission_rate = config.get('commission_rate', 0.001)
        self.slippage_rate = config.get('slippage_rate', 0.0005)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        
    def run_backtest(self, data: pd.DataFrame, signals: Dict, 
                    start_date: str, end_date: str) -> Dict:
        """
        Run comprehensive backtesting analysis.
        
        Args:
            data: Historical data DataFrame
            signals: Trading signals dictionary
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Dictionary with backtesting results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        try:
            # Filter data for backtesting period
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            backtest_data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            # Simulate trading
            portfolio_values = self._simulate_trading(backtest_data, signals)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(portfolio_values)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_values)
            
            # Calculate drawdown analysis
            drawdown_analysis = self._calculate_drawdown_analysis(portfolio_values)
            
            # Calculate trade analysis
            trade_analysis = self._analyze_trades(backtest_data, signals)
            
            # Calculate causal strategy metrics
            causal_metrics = self._calculate_causal_metrics(backtest_data, signals)
            
            backtest_results = {
                'portfolio_values': portfolio_values,
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'drawdown_analysis': drawdown_analysis,
                'trade_analysis': trade_analysis,
                'causal_metrics': causal_metrics,
                'summary': self._generate_summary(performance_metrics, risk_metrics)
            }
            
            logger.info("Backtesting completed successfully")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            raise
            
    def _simulate_trading(self, data: pd.DataFrame, signals: Dict) -> pd.DataFrame:
        """
        Simulate trading based on signals.
        
        Args:
            data: Historical data
            signals: Trading signals
            
        Returns:
            DataFrame with portfolio values over time
        """
        logger.info("Simulating trading...")
        
        # Normalize column names to lowercase for consistent access
        data = data.copy()
        data.columns = [col.lower() if isinstance(col, str) else col for col in data.columns]
        
        portfolio_values = []
        current_capital = self.initial_capital
        position = 0  # Current position size
        trades = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            if i in signals:
                signal = signals[i]
                action = signal['action']
                signal_strength = signal['signal_strength']
                confidence = signal['confidence']
                
                # Calculate position size based on signal strength and confidence
                position_size = self._calculate_position_size(
                    signal_strength, confidence, current_capital
                )
                
                # Execute trade
                if action == 'buy' and position <= 0:
                    # Buy signal
                    shares_to_buy = int(position_size / row['close'])
                    if shares_to_buy > 0:
                        cost = shares_to_buy * row['close'] * (1 + self.commission_rate + self.slippage_rate)
                        if cost <= current_capital:
                            current_capital -= cost
                            position += shares_to_buy
                            trades.append({
                                'timestamp': timestamp,
                                'action': 'buy',
                                'shares': shares_to_buy,
                                'price': row['close'],
                                'cost': cost
                            })
                            
                elif action == 'sell' and position > 0:
                    # Sell signal
                    shares_to_sell = min(position, int(position_size / row['close']))
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * row['close'] * (1 - self.commission_rate - self.slippage_rate)
                        current_capital += proceeds
                        position -= shares_to_sell
                        trades.append({
                            'timestamp': timestamp,
                            'action': 'sell',
                            'shares': shares_to_sell,
                            'price': row['close'],
                            'proceeds': proceeds
                        })
                        
                elif action == 'hold':
                    # Hold position
                    pass
                    
            # Calculate portfolio value
            portfolio_value = current_capital + position * row['close']
            portfolio_values.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': current_capital,
                'position': position,
                'position_value': position * row['close']
            })
            
        return pd.DataFrame(portfolio_values)
        
    def _calculate_position_size(self, signal_strength: float, confidence: float, 
                               available_capital: float) -> float:
        """
        Calculate position size based on signal strength and confidence.
        
        Args:
            signal_strength: Strength of trading signal
            confidence: Confidence in signal
            available_capital: Available capital
            
        Returns:
            Position size
        """
        # Base position size
        base_size = available_capital * 0.1  # 10% of capital per trade
        
        # Adjust based on signal strength
        strength_multiplier = abs(signal_strength)
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        # Final position size
        position_size = base_size * strength_multiplier * confidence_multiplier
        
        return min(position_size, available_capital * 0.5)  # Cap at 50% of capital
        
    def _calculate_performance_metrics(self, portfolio_values: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_values: DataFrame with portfolio values over time
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        returns = portfolio_values['portfolio_value'].pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_values['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_values)
        
        # Additional metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
    def _calculate_risk_metrics(self, portfolio_values: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics.
        
        Args:
            portfolio_values: DataFrame with portfolio values over time
            
        Returns:
            Dictionary with risk metrics
        """
        logger.info("Calculating risk metrics...")
        
        returns = portfolio_values['portfolio_value'].pct_change().dropna()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Tail risk metrics
        tail_ratio = self._calculate_tail_ratio(returns)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Downside risk
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99,
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'downside_volatility': downside_volatility
        }
        
    def _calculate_drawdown_analysis(self, portfolio_values: pd.DataFrame) -> Dict:
        """
        Calculate drawdown analysis.
        
        Args:
            portfolio_values: DataFrame with portfolio values over time
            
        Returns:
            Dictionary with drawdown analysis
        """
        logger.info("Calculating drawdown analysis...")
        
        portfolio_values_array = portfolio_values['portfolio_value'].values
        peak = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - peak) / peak
        
        max_drawdown = np.min(drawdown)
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        average_drawdown = np.mean(drawdown[drawdown < 0])
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'average_drawdown': average_drawdown,
            'drawdown_series': drawdown
        }
        
    def _analyze_trades(self, data: pd.DataFrame, signals: Dict) -> Dict:
        """
        Analyze individual trades.
        
        Args:
            data: Historical data
            signals: Trading signals
            
        Returns:
            Dictionary with trade analysis
        """
        logger.info("Analyzing trades...")
        
        # This would analyze individual trades from the simulation
        # For now, return basic structure
        return {
            'total_trades': len(signals),
            'winning_trades': 0,  # Would be calculated from actual trades
            'losing_trades': 0,
            'average_trade_return': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
        
    def _calculate_causal_metrics(self, data: pd.DataFrame, signals: Dict) -> Dict:
        """
        Calculate causal-specific metrics.
        
        Args:
            data: Historical data
            signals: Trading signals
            
        Returns:
            Dictionary with causal metrics
        """
        logger.info("Calculating causal metrics...")
        
        # Causal signal effectiveness
        signal_effectiveness = self._calculate_signal_effectiveness(signals)
        
        # Causal relationship strength
        causal_strength = self._calculate_causal_strength(signals)
        
        # Uncertainty impact
        uncertainty_impact = self._calculate_uncertainty_impact(signals)
        
        return {
            'signal_effectiveness': signal_effectiveness,
            'causal_strength': causal_strength,
            'uncertainty_impact': uncertainty_impact
        }
        
    def _calculate_signal_effectiveness(self, signals: Dict) -> float:
        """
        Calculate effectiveness of causal signals.
        """
        if not signals:
            return 0.0
            
        # Calculate average signal strength
        signal_strengths = [signal['signal_strength'] for signal in signals.values()]
        return np.mean(signal_strengths)
        
    def _calculate_causal_strength(self, signals: Dict) -> float:
        """
        Calculate strength of causal relationships.
        """
        if not signals:
            return 0.0
            
        # Calculate average confidence
        confidences = [signal['confidence'] for signal in signals.values()]
        return np.mean(confidences)
        
    def _calculate_uncertainty_impact(self, signals: Dict) -> float:
        """
        Calculate impact of uncertainty on signals.
        """
        if not signals:
            return 0.0
            
        # Calculate uncertainty impact based on confidence
        confidences = [signal['confidence'] for signal in signals.values()]
        uncertainty_impact = 1 - np.mean(confidences)
        return uncertainty_impact
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
        return returns.mean() / downside_std
        
    def _calculate_calmar_ratio(self, portfolio_values: pd.DataFrame) -> float:
        """Calculate Calmar ratio."""
        max_dd = self._calculate_max_drawdown(portfolio_values)
        if max_dd == 0:
            return float('inf')
        annual_return = (portfolio_values['portfolio_value'].iloc[-1] / self.initial_capital) ** (252 / len(portfolio_values)) - 1
        return annual_return / abs(max_dd)
        
    def _calculate_max_drawdown(self, portfolio_values: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        portfolio_values_array = portfolio_values['portfolio_value'].values
        peak = np.maximum.accumulate(portfolio_values_array)
        drawdown = (portfolio_values_array - peak) / peak
        return np.min(drawdown)
        
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        positive_returns = returns[returns > 0]
        return len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio."""
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        return var_95 / var_99 if var_99 != 0 else 0
        
    def _calculate_max_drawdown_duration(self, drawdown: np.ndarray) -> int:
        """Calculate maximum drawdown duration."""
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
                
        return max(drawdown_periods) if drawdown_periods else 0
        
    def _generate_summary(self, performance_metrics: Dict, risk_metrics: Dict) -> str:
        """
        Generate summary of backtesting results.
        
        Args:
            performance_metrics: Performance metrics
            risk_metrics: Risk metrics
            
        Returns:
            Summary string
        """
        summary = f"""
=== Backtesting Summary ===

Performance:
- Total Return: {performance_metrics['total_return']:.2%}
- Annualized Return: {performance_metrics['annualized_return']:.2%}
- Volatility: {performance_metrics['volatility']:.2%}
- Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}
- Max Drawdown: {performance_metrics['max_drawdown']:.2%}

Risk:
- VaR (95%): {risk_metrics['var_95']:.2%}
- VaR (99%): {risk_metrics['var_99']:.2%}
- Expected Shortfall (95%): {risk_metrics['es_95']:.2%}
- Downside Volatility: {risk_metrics['downside_volatility']:.2%}
        """
        
        return summary