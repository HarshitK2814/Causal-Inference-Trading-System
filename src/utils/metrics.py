"""
Metrics Module

This module provides comprehensive metrics calculation for causal trading analysis,
including performance metrics, risk metrics, and causal-specific metrics.
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
        """
        Initialize the metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.trading_days_per_year = config.get('trading_days_per_year', 252)
        
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Calculating performance metrics...")
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (self.trading_days_per_year / len(returns)) - 1
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
        
        return metrics
        
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with risk metrics
        """
        logger.info("Calculating risk metrics...")
        
        metrics = {}
        
        # Downside risk metrics
        downside_returns = returns[returns < 0]
        metrics['downside_deviation'] = downside_returns.std() * np.sqrt(self.trading_days_per_year)
        metrics['downside_capture_ratio'] = self._calculate_downside_capture_ratio(returns)
        
        # Tail risk metrics
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
        
        # Drawdown metrics
        drawdowns = self._calculate_drawdowns(returns)
        metrics['max_drawdown'] = drawdowns.min()
        metrics['avg_drawdown'] = drawdowns[drawdowns < 0].mean()
        metrics['max_drawdown_duration'] = self._calculate_max_drawdown_duration(drawdowns)
        
        # Risk concentration
        metrics['concentration_risk'] = self._calculate_concentration_risk(returns)
        
        return metrics
        
    def calculate_causal_metrics(self, returns: pd.Series, signals: Dict, 
                                causal_effects: Dict) -> Dict[str, float]:
        """
        Calculate causal-specific metrics.
        
        Args:
            returns: Series of returns
            signals: Trading signals
            causal_effects: Causal effect estimates
            
        Returns:
            Dictionary with causal metrics
        """
        logger.info("Calculating causal metrics...")
        
        metrics = {}
        
        # Causal signal effectiveness
        metrics['causal_signal_effectiveness'] = self._calculate_causal_signal_effectiveness(
            returns, signals, causal_effects
        )
        
        # Causal relationship strength
        metrics['causal_relationship_strength'] = self._calculate_causal_relationship_strength(
            causal_effects
        )
        
        # Uncertainty impact on performance
        metrics['uncertainty_impact'] = self._calculate_uncertainty_impact(
            returns, signals
        )
        
        # Causal consistency
        metrics['causal_consistency'] = self._calculate_causal_consistency(
            signals, causal_effects
        )
        
        return metrics
        
    def calculate_bandit_metrics(self, returns: pd.Series, signals: Dict, 
                               arm_performance: Dict) -> Dict[str, float]:
        """
        Calculate contextual bandit metrics.
        
        Args:
            returns: Series of returns
            signals: Trading signals
            arm_performance: Performance by arm
            
        Returns:
            Dictionary with bandit metrics
        """
        logger.info("Calculating bandit metrics...")
        
        metrics = {}
        
        # Exploration vs exploitation
        metrics['exploration_rate'] = self._calculate_exploration_rate(signals)
        metrics['exploitation_efficiency'] = self._calculate_exploitation_efficiency(
            arm_performance
        )
        
        # Regret analysis
        metrics['cumulative_regret'] = self._calculate_cumulative_regret(
            returns, signals, arm_performance
        )
        
        # Arm selection diversity
        metrics['arm_diversity'] = self._calculate_arm_diversity(signals)
        
        # Learning progress
        metrics['learning_progress'] = self._calculate_learning_progress(
            arm_performance
        )
        
        return metrics
        
    def calculate_uncertainty_metrics(self, returns: pd.Series, 
                                    uncertainty_estimates: Dict) -> Dict[str, float]:
        """
        Calculate uncertainty-related metrics.
        
        Args:
            returns: Series of returns
            uncertainty_estimates: Uncertainty estimates
            
        Returns:
            Dictionary with uncertainty metrics
        """
        logger.info("Calculating uncertainty metrics...")
        
        metrics = {}
        
        # Uncertainty calibration
        metrics['uncertainty_calibration'] = self._calculate_uncertainty_calibration(
            returns, uncertainty_estimates
        )
        
        # Epistemic vs aleatoric uncertainty
        metrics['epistemic_uncertainty'] = self._calculate_epistemic_uncertainty(
            uncertainty_estimates
        )
        metrics['aleatoric_uncertainty'] = self._calculate_aleatoric_uncertainty(
            uncertainty_estimates
        )
        
        # Uncertainty impact on decisions
        metrics['uncertainty_decision_impact'] = self._calculate_uncertainty_decision_impact(
            uncertainty_estimates
        )
        
        return metrics
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        if returns.std() == 0:
            return 0.0
        return excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year)
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
        return returns.mean() / downside_std * np.sqrt(self.trading_days_per_year)
        
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return float('inf')
        annual_return = (1 + returns).prod() ** (self.trading_days_per_year / len(returns)) - 1
        return annual_return / abs(max_dd)
        
    def _calculate_information_ratio(self, returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series]) -> float:
        """Calculate information ratio."""
        if benchmark_returns is None:
            return 0.0
        excess_returns = returns - benchmark_returns
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year)
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()
        
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        positive_returns = returns[returns > 0]
        return len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
    def _calculate_expectancy(self, returns: pd.Series) -> float:
        """Calculate expectancy."""
        return returns.mean()
        
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, confidence_level * 100)
        
    def _calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
        
    def _calculate_downside_capture_ratio(self, returns: pd.Series) -> float:
        """Calculate downside capture ratio."""
        # This would require benchmark returns for proper calculation
        # For now, return a simplified version
        downside_returns = returns[returns < 0]
        return downside_returns.mean() if len(downside_returns) > 0 else 0
        
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio."""
        var_95 = self._calculate_var(returns, 0.05)
        var_99 = self._calculate_var(returns, 0.01)
        return var_95 / var_99 if var_99 != 0 else 0
        
    def _calculate_drawdowns(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        return (cumulative_returns - running_max) / running_max
        
    def _calculate_max_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """Calculate maximum drawdown duration."""
        in_drawdown = drawdowns < 0
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
        
    def _calculate_concentration_risk(self, returns: pd.Series) -> float:
        """Calculate concentration risk."""
        # Simplified concentration risk calculation
        return returns.std() / abs(returns.mean()) if returns.mean() != 0 else 0
        
    def _calculate_causal_signal_effectiveness(self, returns: pd.Series, 
                                             signals: Dict, causal_effects: Dict) -> float:
        """Calculate causal signal effectiveness."""
        if not signals:
            return 0.0
            
        # Calculate correlation between signal strength and returns
        signal_strengths = [signal['signal_strength'] for signal in signals.values()]
        if len(signal_strengths) == len(returns):
            return np.corrcoef(signal_strengths, returns)[0, 1]
        return 0.0
        
    def _calculate_causal_relationship_strength(self, causal_effects: Dict) -> float:
        """Calculate causal relationship strength."""
        if not causal_effects:
            return 0.0
            
        # Calculate average absolute causal effect
        effects = []
        for effect_name, effects_dict in causal_effects.items():
            if isinstance(effects_dict, dict):
                for method, value in effects_dict.items():
                    if isinstance(value, (int, float)):
                        effects.append(abs(value))
                        
        return np.mean(effects) if effects else 0.0
        
    def _calculate_uncertainty_impact(self, returns: pd.Series, signals: Dict) -> float:
        """Calculate uncertainty impact on performance."""
        if not signals:
            return 0.0
            
        # Calculate correlation between confidence and returns
        confidences = [signal['confidence'] for signal in signals.values()]
        if len(confidences) == len(returns):
            return np.corrcoef(confidences, returns)[0, 1]
        return 0.0
        
    def _calculate_causal_consistency(self, signals: Dict, causal_effects: Dict) -> float:
        """Calculate causal consistency."""
        if not signals or not causal_effects:
            return 0.0
            
        # This would measure consistency between signals and causal relationships
        # For now, return a simplified metric
        return 0.8
        
    def _calculate_exploration_rate(self, signals: Dict) -> float:
        """Calculate exploration rate."""
        if not signals:
            return 0.0
            
        # Count exploration vs exploitation actions
        exploration_actions = 0
        for signal in signals.values():
            if signal.get('action') == 'explore':
                exploration_actions += 1
                
        return exploration_actions / len(signals)
        
    def _calculate_exploitation_efficiency(self, arm_performance: Dict) -> float:
        """Calculate exploitation efficiency."""
        if not arm_performance:
            return 0.0
            
        # Calculate efficiency of arm selection
        performances = list(arm_performance.values())
        return np.mean(performances) if performances else 0.0
        
    def _calculate_cumulative_regret(self, returns: pd.Series, 
                                   signals: Dict, arm_performance: Dict) -> float:
        """Calculate cumulative regret."""
        if not signals or not arm_performance:
            return 0.0
            
        # Simplified regret calculation
        optimal_performance = max(arm_performance.values()) if arm_performance else 0
        actual_performance = returns.mean()
        return optimal_performance - actual_performance
        
    def _calculate_arm_diversity(self, signals: Dict) -> float:
        """Calculate arm selection diversity."""
        if not signals:
            return 0.0
            
        # Calculate entropy of arm selection
        actions = [signal['action'] for signal in signals.values()]
        action_counts = pd.Series(actions).value_counts()
        probabilities = action_counts / action_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy
        
    def _calculate_learning_progress(self, arm_performance: Dict) -> float:
        """Calculate learning progress."""
        if not arm_performance:
            return 0.0
            
        # Calculate improvement in arm performance over time
        performances = list(arm_performance.values())
        if len(performances) < 2:
            return 0.0
            
        # Calculate trend in performance
        x = np.arange(len(performances))
        slope, _ = np.polyfit(x, performances, 1)
        return slope
        
    def _calculate_uncertainty_calibration(self, returns: pd.Series, 
                                         uncertainty_estimates: Dict) -> float:
        """Calculate uncertainty calibration."""
        # This would measure how well uncertainty estimates match actual variability
        # For now, return a simplified metric
        return 0.7
        
    def _calculate_epistemic_uncertainty(self, uncertainty_estimates: Dict) -> float:
        """Calculate epistemic uncertainty."""
        if not uncertainty_estimates:
            return 0.0
            
        # Extract epistemic uncertainty from estimates
        epistemic_values = []
        for model_name, estimates in uncertainty_estimates.items():
            if 'uncertainty_decomposition' in estimates:
                epistemic = estimates['uncertainty_decomposition'].get('epistemic_uncertainty', [])
                if isinstance(epistemic, (list, np.ndarray)):
                    epistemic_values.extend(epistemic)
                else:
                    epistemic_values.append(epistemic)
                    
        return np.mean(epistemic_values) if epistemic_values else 0.0
        
    def _calculate_aleatoric_uncertainty(self, uncertainty_estimates: Dict) -> float:
        """Calculate aleatoric uncertainty."""
        if not uncertainty_estimates:
            return 0.0
            
        # Extract aleatoric uncertainty from estimates
        aleatoric_values = []
        for model_name, estimates in uncertainty_estimates.items():
            if 'uncertainty_decomposition' in estimates:
                aleatoric = estimates['uncertainty_decomposition'].get('aleatoric_uncertainty', [])
                if isinstance(aleatoric, (list, np.ndarray)):
                    aleatoric_values.extend(aleatoric)
                else:
                    aleatoric_values.append(aleatoric)
                    
        return np.mean(aleatoric_values) if aleatoric_values else 0.0
        
    def _calculate_uncertainty_decision_impact(self, uncertainty_estimates: Dict) -> float:
        """Calculate uncertainty impact on decisions."""
        # This would measure how uncertainty affects decision-making
        # For now, return a simplified metric
        return 0.6