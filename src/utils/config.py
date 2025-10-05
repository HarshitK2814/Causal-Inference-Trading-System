"""
Configuration Module

This module handles configuration management for the causal trading system,
including loading, validation, and environment-specific settings.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for the causal trading system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path:
            self.config_path = config_path
            
        if not self.config_path:
            self.config_path = self._find_default_config()
            
        if not self.config_path or not os.path.exists(self.config_path):
            logger.warning("No configuration file found, using default configuration")
            return self._get_default_config()
            
        try:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            elif self.config_path.endswith('.json'):
                with open(self.config_path, 'r') as file:
                    self.config = json.load(file)
            else:
                raise ValueError(f"Unsupported configuration file format: {self.config_path}")
                
            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
            
    def _find_default_config(self) -> Optional[str]:
        """
        Find default configuration file.
        
        Returns:
            Path to default configuration file or None
        """
        possible_paths = [
            'config.yaml',
            'config.yml',
            'config.json',
            'configs/config.yaml',
            'configs/config.yml',
            'configs/config.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        return None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'environment': self.environment,
            'data_pipeline': {
                'batch_size': 1000,
                'max_missing_percentage': 10,
                'outlier_threshold': 3.0
            },
            'data_loader': {
                'api_keys': {},
                'database': {
                    'path': 'data.db'
                }
            },
            'data_validator': {
                'validation_rules': {
                    'max_missing_percentage': 10,
                    'expected_types': {},
                    'outlier_columns': []
                }
            },
            'causal_inference': {
                'causal_method': 'pc',
                'treatments': [],
                'outcomes': []
            },
            'uncertainty_quantification': {
                'uncertainty_methods': ['bootstrap', 'conformal'],
                'bootstrap_samples': 1000,
                'confidence_level': 0.95
            },
            'contextual_bandit': {
                'arms': ['buy', 'sell', 'hold'],
                'algorithm': 'thompson_sampling',
                'context_dim': 10
            },
            'backtesting': {
                'initial_capital': 100000,
                'commission_rate': 0.001,
                'slippage_rate': 0.0005,
                'risk_free_rate': 0.02
            },
            'monitoring': {
                'alert_thresholds': {
                    'max_drawdown': -0.2,
                    'min_sharpe_ratio': 0.5,
                    'max_volatility': 0.3
                }
            }
        }
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set_config(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating configuration...")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate required sections
        required_sections = [
            'data_pipeline',
            'data_loader',
            'data_validator',
            'causal_inference',
            'uncertainty_quantification',
            'contextual_bandit',
            'backtesting',
            'monitoring'
        ]
        
        for section in required_sections:
            if section not in self.config:
                validation_results['errors'].append(f"Missing required section: {section}")
                validation_results['is_valid'] = False
                
        # Validate data pipeline configuration
        if 'data_pipeline' in self.config:
            dp_config = self.config['data_pipeline']
            if 'batch_size' in dp_config and dp_config['batch_size'] <= 0:
                validation_results['errors'].append("batch_size must be positive")
                validation_results['is_valid'] = False
                
        # Validate backtesting configuration
        if 'backtesting' in self.config:
            bt_config = self.config['backtesting']
            if 'initial_capital' in bt_config and bt_config['initial_capital'] <= 0:
                validation_results['errors'].append("initial_capital must be positive")
                validation_results['is_valid'] = False
                
            if 'commission_rate' in bt_config and (bt_config['commission_rate'] < 0 or bt_config['commission_rate'] > 1):
                validation_results['errors'].append("commission_rate must be between 0 and 1")
                validation_results['is_valid'] = False
                
        # Validate contextual bandit configuration
        if 'contextual_bandit' in self.config:
            cb_config = self.config['contextual_bandit']
            if 'arms' in cb_config and not cb_config['arms']:
                validation_results['warnings'].append("No arms defined for contextual bandit")
                
        logger.info(f"Configuration validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
        
    def save_config(self, output_path: str, format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            output_path: Path to save configuration
            format: Output format ('yaml' or 'json')
        """
        try:
            if format.lower() == 'yaml':
                with open(output_path, 'w') as file:
                    yaml.dump(self.config, file, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(output_path, 'w') as file:
                    json.dump(self.config, file, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
            
    def get_environment_config(self) -> Dict[str, Any]:
        """
        Get environment-specific configuration.
        
        Returns:
            Environment-specific configuration
        """
        env_config = self.config.get('environments', {}).get(self.environment, {})
        
        # Merge with base configuration
        merged_config = self.config.copy()
        merged_config.update(env_config)
        
        return merged_config
        
    def load_environment_variables(self):
        """
        Load configuration from environment variables.
        """
        logger.info("Loading configuration from environment variables...")
        
        # Map environment variables to configuration keys
        env_mappings = {
            'INITIAL_CAPITAL': 'backtesting.initial_capital',
            'COMMISSION_RATE': 'backtesting.commission_rate',
            'SLIPPAGE_RATE': 'backtesting.slippage_rate',
            'RISK_FREE_RATE': 'backtesting.risk_free_rate',
            'BATCH_SIZE': 'data_pipeline.batch_size',
            'MAX_MISSING_PERCENTAGE': 'data_pipeline.max_missing_percentage',
            'CAUSAL_METHOD': 'causal_inference.causal_method',
            'BANDIT_ALGORITHM': 'contextual_bandit.algorithm'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Convert to appropriate type
                if config_key.endswith('_rate') or config_key.endswith('_percentage'):
                    value = float(env_value)
                elif config_key.endswith('_size') or config_key.endswith('_capital'):
                    value = int(env_value)
                else:
                    value = env_value
                    
                self.set_config(config_key, value)
                logger.info(f"Set {config_key} = {value} from environment variable {env_var}")
                
    def create_config_template(self, output_path: str):
        """
        Create a configuration template file.
        
        Args:
            output_path: Path to save template
        """
        template_config = {
            'environment': 'development',
            'data_pipeline': {
                'batch_size': 1000,
                'max_missing_percentage': 10,
                'outlier_threshold': 3.0,
                'feature_engineering': {
                    'technical_indicators': True,
                    'lag_features': 5,
                    'rolling_windows': [5, 10, 20]
                }
            },
            'data_loader': {
                'api_keys': {
                    'alpha_vantage': 'your_api_key_here',
                    'quandl': 'your_api_key_here'
                },
                'database': {
                    'path': 'data.db',
                    'type': 'sqlite'
                },
                'data_sources': [
                    {
                        'type': 'yfinance',
                        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                        'period': '1y'
                    }
                ]
            },
            'data_validator': {
                'validation_rules': {
                    'max_missing_percentage': 10,
                    'expected_types': {
                        'close': 'float64',
                        'volume': 'int64'
                    },
                    'outlier_columns': ['close', 'volume']
                }
            },
            'causal_inference': {
                'causal_method': 'pc',
                'treatments': ['market_sentiment', 'news_sentiment'],
                'outcomes': ['price_change', 'volume_change'],
                'confounders': ['market_cap', 'sector', 'volatility']
            },
            'uncertainty_quantification': {
                'uncertainty_methods': ['bootstrap', 'conformal'],
                'bootstrap_samples': 1000,
                'confidence_level': 0.95,
                'cross_validation_folds': 5
            },
            'contextual_bandit': {
                'arms': ['buy', 'sell', 'hold'],
                'algorithm': 'thompson_sampling',
                'context_dim': 10,
                'exploration_rate': 0.1
            },
            'backtesting': {
                'initial_capital': 100000,
                'commission_rate': 0.001,
                'slippage_rate': 0.0005,
                'risk_free_rate': 0.02,
                'rebalance_frequency': 'daily'
            },
            'monitoring': {
                'alert_thresholds': {
                    'max_drawdown': -0.2,
                    'min_sharpe_ratio': 0.5,
                    'max_volatility': 0.3,
                    'min_confidence': 0.3
                },
                'check_interval': 3600,  # seconds
                'alert_channels': ['email', 'slack']
            },
            'environments': {
                'development': {
                    'debug': True,
                    'log_level': 'DEBUG'
                },
                'production': {
                    'debug': False,
                    'log_level': 'INFO'
                }
            }
        }
        
        try:
            with open(output_path, 'w') as file:
                yaml.dump(template_config, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration template created at {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create configuration template: {e}")
            raise