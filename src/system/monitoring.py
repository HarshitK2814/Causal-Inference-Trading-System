"""
System Monitoring Module

This module provides comprehensive monitoring capabilities for the causal trading system,
including performance tracking, alerting, and system health checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Comprehensive system monitoring for causal trading system.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the system monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alert_thresholds = config.get('alert_thresholds', {})
        self.performance_history = []
        self.alert_history = []
        
    def check_system_health(self, data: pd.DataFrame, signals: Dict, 
                           performance_metrics: Dict) -> Dict:
        """
        Check overall system health.
        
        Args:
            data: Current data
            signals: Trading signals
            performance_metrics: Performance metrics
            
        Returns:
            Dictionary with system health status
        """
        logger.info("Checking system health...")
        
        try:
            health_checks = {
                'data_quality': self._check_data_quality(data),
                'signal_quality': self._check_signal_quality(signals),
                'performance_health': self._check_performance_health(performance_metrics),
                'system_resources': self._check_system_resources(),
                'causal_model_health': self._check_causal_model_health(signals),
                'uncertainty_health': self._check_uncertainty_health(signals)
            }
            
            # Overall health status
            overall_health = self._calculate_overall_health(health_checks)
            health_checks['overall_health'] = overall_health
            
            # Generate alerts if needed
            alerts = self._generate_alerts(health_checks)
            health_checks['alerts'] = alerts
            
            # Update performance history
            self._update_performance_history(performance_metrics)
            
            logger.info("System health check completed")
            return health_checks
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            raise
            
    def _check_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Check data quality metrics.
        
        Args:
            data: Current data
            
        Returns:
            Dictionary with data quality metrics
        """
        logger.info("Checking data quality...")
        
        quality_metrics = {}
        
        # Check for missing values
        missing_percentage = (data.isnull().sum() / len(data)) * 100
        quality_metrics['missing_values'] = {
            'max_missing': missing_percentage.max(),
            'avg_missing': missing_percentage.mean(),
            'columns_with_missing': (missing_percentage > 0).sum()
        }
        
        # Check for outliers
        outlier_counts = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
            outlier_counts[column] = outliers
            
        quality_metrics['outliers'] = {
            'max_outliers': max(outlier_counts.values()) if outlier_counts else 0,
            'total_outliers': sum(outlier_counts.values()),
            'outlier_columns': len([col for col, count in outlier_counts.items() if count > 0])
        }
        
        # Check data freshness
        if 'timestamp' in data.columns:
            latest_timestamp = pd.to_datetime(data['timestamp']).max()
            time_since_update = datetime.now() - latest_timestamp
            quality_metrics['data_freshness'] = {
                'latest_timestamp': latest_timestamp,
                'hours_since_update': time_since_update.total_seconds() / 3600
            }
        else:
            quality_metrics['data_freshness'] = {
                'latest_timestamp': None,
                'hours_since_update': None
            }
            
        # Overall data quality score
        quality_score = self._calculate_data_quality_score(quality_metrics)
        quality_metrics['quality_score'] = quality_score
        
        return quality_metrics
        
    def _check_signal_quality(self, signals: Dict) -> Dict:
        """
        Check trading signal quality.
        
        Args:
            signals: Trading signals
            
        Returns:
            Dictionary with signal quality metrics
        """
        logger.info("Checking signal quality...")
        
        if not signals:
            return {
                'signal_count': 0,
                'quality_score': 0,
                'warnings': ['No signals generated']
            }
            
        signal_metrics = {}
        
        # Signal count and frequency
        signal_count = len(signals)
        signal_metrics['signal_count'] = signal_count
        
        # Signal strength analysis
        signal_strengths = [signal['signal_strength'] for signal in signals.values()]
        signal_metrics['signal_strength'] = {
            'mean': np.mean(signal_strengths),
            'std': np.std(signal_strengths),
            'min': np.min(signal_strengths),
            'max': np.max(signal_strengths)
        }
        
        # Confidence analysis
        confidences = [signal['confidence'] for signal in signals.values()]
        signal_metrics['confidence'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        # Action distribution
        actions = [signal['action'] for signal in signals.values()]
        action_counts = pd.Series(actions).value_counts()
        signal_metrics['action_distribution'] = action_counts.to_dict()
        
        # Signal quality score
        quality_score = self._calculate_signal_quality_score(signal_metrics)
        signal_metrics['quality_score'] = quality_score
        
        return signal_metrics
        
    def _check_performance_health(self, performance_metrics: Dict) -> Dict:
        """
        Check performance health.
        
        Args:
            performance_metrics: Performance metrics
            
        Returns:
            Dictionary with performance health metrics
        """
        logger.info("Checking performance health...")
        
        if not performance_metrics:
            return {
                'health_score': 0,
                'warnings': ['No performance metrics available']
            }
            
        performance_health = {}
        
        # Check key performance indicators
        total_return = performance_metrics.get('total_return', 0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0)
        volatility = performance_metrics.get('volatility', 0)
        
        performance_health['metrics'] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }
        
        # Performance warnings
        warnings = []
        if total_return < -0.1:  # -10% return
            warnings.append('Poor total return')
        if sharpe_ratio < 0.5:
            warnings.append('Low Sharpe ratio')
        if max_drawdown < -0.2:  # -20% drawdown
            warnings.append('High maximum drawdown')
        if volatility > 0.3:  # 30% volatility
            warnings.append('High volatility')
            
        performance_health['warnings'] = warnings
        
        # Performance health score
        health_score = self._calculate_performance_health_score(performance_health)
        performance_health['health_score'] = health_score
        
        return performance_health
        
    def _check_system_resources(self) -> Dict:
        """
        Check system resource usage.
        
        Returns:
            Dictionary with system resource metrics
        """
        logger.info("Checking system resources...")
        
        # This would check actual system resources
        # For now, return mock data
        resource_metrics = {
            'cpu_usage': 45.2,  # Mock CPU usage
            'memory_usage': 67.8,  # Mock memory usage
            'disk_usage': 23.1,  # Mock disk usage
            'network_latency': 12.5  # Mock network latency
        }
        
        # Resource warnings
        warnings = []
        if resource_metrics['cpu_usage'] > 80:
            warnings.append('High CPU usage')
        if resource_metrics['memory_usage'] > 85:
            warnings.append('High memory usage')
        if resource_metrics['disk_usage'] > 90:
            warnings.append('High disk usage')
        if resource_metrics['network_latency'] > 100:
            warnings.append('High network latency')
            
        resource_metrics['warnings'] = warnings
        
        return resource_metrics
        
    def _check_causal_model_health(self, signals: Dict) -> Dict:
        """
        Check causal model health.
        
        Args:
            signals: Trading signals
            
        Returns:
            Dictionary with causal model health metrics
        """
        logger.info("Checking causal model health...")
        
        if not signals:
            return {
                'health_score': 0,
                'warnings': ['No signals to analyze causal model health']
            }
            
        causal_health = {}
        
        # Analyze signal consistency with causal relationships
        signal_consistency = self._analyze_signal_consistency(signals)
        causal_health['signal_consistency'] = signal_consistency
        
        # Check for causal relationship stability
        relationship_stability = self._check_relationship_stability(signals)
        causal_health['relationship_stability'] = relationship_stability
        
        # Causal model health score
        health_score = self._calculate_causal_health_score(causal_health)
        causal_health['health_score'] = health_score
        
        return causal_health
        
    def _check_uncertainty_health(self, signals: Dict) -> Dict:
        """
        Check uncertainty quantification health.
        
        Args:
            signals: Trading signals
            
        Returns:
            Dictionary with uncertainty health metrics
        """
        logger.info("Checking uncertainty health...")
        
        if not signals:
            return {
                'health_score': 0,
                'warnings': ['No signals to analyze uncertainty health']
            }
            
        uncertainty_health = {}
        
        # Analyze uncertainty levels
        confidences = [signal['confidence'] for signal in signals.values()]
        uncertainty_health['confidence_metrics'] = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        # Check for high uncertainty periods
        low_confidence_signals = [conf for conf in confidences if conf < 0.3]
        uncertainty_health['low_confidence_count'] = len(low_confidence_signals)
        uncertainty_health['low_confidence_percentage'] = len(low_confidence_signals) / len(confidences)
        
        # Uncertainty warnings
        warnings = []
        if uncertainty_health['low_confidence_percentage'] > 0.3:
            warnings.append('High percentage of low-confidence signals')
        if np.mean(confidences) < 0.5:
            warnings.append('Low average confidence')
            
        uncertainty_health['warnings'] = warnings
        
        # Uncertainty health score
        health_score = self._calculate_uncertainty_health_score(uncertainty_health)
        uncertainty_health['health_score'] = health_score
        
        return uncertainty_health
        
    def _calculate_overall_health(self, health_checks: Dict) -> Dict:
        """
        Calculate overall system health.
        
        Args:
            health_checks: Individual health check results
            
        Returns:
            Dictionary with overall health status
        """
        # Calculate weighted health score
        weights = {
            'data_quality': 0.25,
            'signal_quality': 0.25,
            'performance_health': 0.20,
            'system_resources': 0.15,
            'causal_model_health': 0.10,
            'uncertainty_health': 0.05
        }
        
        overall_score = 0
        for component, weight in weights.items():
            if component in health_checks:
                component_score = health_checks[component].get('quality_score', 
                                                             health_checks[component].get('health_score', 0))
                overall_score += component_score * weight
                
        # Determine health status
        if overall_score >= 0.8:
            status = 'excellent'
        elif overall_score >= 0.6:
            status = 'good'
        elif overall_score >= 0.4:
            status = 'fair'
        else:
            status = 'poor'
            
        return {
            'overall_score': overall_score,
            'status': status,
            'timestamp': datetime.now()
        }
        
    def _generate_alerts(self, health_checks: Dict) -> List[Dict]:
        """
        Generate alerts based on health checks.
        
        Args:
            health_checks: Health check results
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Check each component for alerts
        for component, metrics in health_checks.items():
            if component == 'overall_health':
                continue
                
            # Check for warnings
            if 'warnings' in metrics and metrics['warnings']:
                for warning in metrics['warnings']:
                    alert = {
                        'component': component,
                        'type': 'warning',
                        'message': warning,
                        'timestamp': datetime.now(),
                        'severity': 'medium'
                    }
                    alerts.append(alert)
                    
            # Check for critical issues
            if component == 'data_quality':
                if metrics.get('missing_values', {}).get('max_missing', 0) > 50:
                    alert = {
                        'component': component,
                        'type': 'critical',
                        'message': 'High percentage of missing values',
                        'timestamp': datetime.now(),
                        'severity': 'high'
                    }
                    alerts.append(alert)
                    
        return alerts
        
    def _update_performance_history(self, performance_metrics: Dict):
        """
        Update performance history.
        
        Args:
            performance_metrics: Current performance metrics
        """
        if performance_metrics:
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': performance_metrics
            })
            
            # Keep only last 100 entries
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
                
    def _calculate_data_quality_score(self, quality_metrics: Dict) -> float:
        """Calculate data quality score."""
        score = 1.0
        
        # Penalize missing values
        max_missing = quality_metrics['missing_values']['max_missing']
        if max_missing > 10:
            score -= 0.3
        elif max_missing > 5:
            score -= 0.1
            
        # Penalize outliers
        max_outliers = quality_metrics['outliers']['max_outliers']
        if max_outliers > 100:
            score -= 0.2
        elif max_outliers > 50:
            score -= 0.1
            
        return max(0.0, score)
        
    def _calculate_signal_quality_score(self, signal_metrics: Dict) -> float:
        """Calculate signal quality score."""
        score = 1.0
        
        # Penalize low confidence
        mean_confidence = signal_metrics['confidence']['mean']
        if mean_confidence < 0.3:
            score -= 0.4
        elif mean_confidence < 0.5:
            score -= 0.2
            
        # Penalize low signal strength
        mean_strength = abs(signal_metrics['signal_strength']['mean'])
        if mean_strength < 0.2:
            score -= 0.3
        elif mean_strength < 0.4:
            score -= 0.1
            
        return max(0.0, score)
        
    def _calculate_performance_health_score(self, performance_health: Dict) -> float:
        """Calculate performance health score."""
        score = 1.0
        
        # Penalize based on warnings
        warning_count = len(performance_health.get('warnings', []))
        score -= warning_count * 0.2
        
        return max(0.0, score)
        
    def _calculate_causal_health_score(self, causal_health: Dict) -> float:
        """Calculate causal model health score."""
        score = 1.0
        
        # This would be more sophisticated in practice
        # For now, return a base score
        return score
        
    def _calculate_uncertainty_health_score(self, uncertainty_health: Dict) -> float:
        """Calculate uncertainty health score."""
        score = 1.0
        
        # Penalize based on warnings
        warning_count = len(uncertainty_health.get('warnings', []))
        score -= warning_count * 0.3
        
        return max(0.0, score)
        
    def _analyze_signal_consistency(self, signals: Dict) -> Dict:
        """Analyze signal consistency."""
        # This would analyze consistency of signals with causal relationships
        return {'consistency_score': 0.8}
        
    def _check_relationship_stability(self, signals: Dict) -> Dict:
        """Check causal relationship stability."""
        # This would check stability of causal relationships over time
        return {'stability_score': 0.7}