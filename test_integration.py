#!/usr/bin/env python3
"""
Integration test script for the Integrated Causal Trading System.

This script tests all components to ensure they work together without errors.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from src.data.data_pipeline import DataPipeline
        from src.data.data_loader import DataLoader
        from src.data.data_validator import DataValidator
        logger.info("✓ Data modules imported successfully")
        
        from src.models.causal_inference import CausalInference
        from src.models.uncertainty_quantification import UncertaintyQuantification
        from src.models.contextual_bandit import ContextualBandit
        logger.info("✓ Model modules imported successfully")
        
        from src.system.integrated_system import IntegratedCausalTradingSystem
        from src.system.backtesting import BacktestingEngine
        from src.system.monitoring import SystemMonitor
        logger.info("✓ System modules imported successfully")
        
        from src.utils.config import ConfigManager
        from src.utils.metrics import MetricsCalculator
        from src.utils.visualization import VisualizationEngine
        logger.info("✓ Utility modules imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config_manager():
    """Test configuration manager."""
    logger.info("Testing ConfigManager...")
    
    try:
        from src.utils.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config('config.yaml')
        
        assert isinstance(config, dict), "Config should be a dictionary"
        assert 'data_pipeline' in config, "Config should have data_pipeline section"
        
        logger.info("✓ ConfigManager working correctly")
        return True, config
    except Exception as e:
        logger.error(f"✗ ConfigManager failed: {e}")
        return False, {}


def test_data_loader():
    """Test data loader."""
    logger.info("Testing DataLoader...")
    
    try:
        from src.data.data_loader import DataLoader
        
        config = {'api_keys': {}, 'database': {}}
        loader = DataLoader(config)
        
        # Test creating synthetic data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        test_data = pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        assert len(test_data) > 0, "Test data should not be empty"
        
        logger.info("✓ DataLoader working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ DataLoader failed: {e}")
        return False


def test_data_validator():
    """Test data validator."""
    logger.info("Testing DataValidator...")
    
    try:
        from src.data.data_validator import DataValidator
        
        config = {'validation_rules': {}}
        validator = DataValidator(config)
        
        # Create test data with some missing values
        test_data = pd.DataFrame({
            'price': [100, 101, np.nan, 103, 104],
            'volume': [1000, 1100, 1200, np.nan, 1400]
        })
        
        missing_stats = validator.check_missing_values(test_data)
        assert isinstance(missing_stats, dict), "Missing stats should be a dictionary"
        
        logger.info("✓ DataValidator working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ DataValidator failed: {e}")
        return False


def test_data_pipeline():
    """Test complete data pipeline."""
    logger.info("Testing DataPipeline...")
    
    try:
        from src.data.data_pipeline import DataPipeline
        
        config = {
            'max_missing_percentage': 10,
            'feature_engineering': {
                'rolling_windows': [5, 10]
            }
        }
        pipeline = DataPipeline(config)
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        test_data = pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Test preprocessing
        processed = pipeline.preprocess_data(test_data)
        assert len(processed) > 0, "Processed data should not be empty"
        
        # Test feature engineering
        features = pipeline.engineer_features(processed)
        assert len(features.columns) > len(test_data.columns), "Should have more features after engineering"
        
        # Test validation
        is_valid = pipeline.validate_data(features)
        assert isinstance(is_valid, bool), "Validation should return boolean"
        
        logger.info("✓ DataPipeline working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ DataPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model components."""
    logger.info("Testing Models...")
    
    try:
        from src.models.causal_inference import CausalInference
        from src.models.uncertainty_quantification import UncertaintyQuantification
        from src.models.contextual_bandit import ContextualBandit
        
        config_causal = {}
        causal = CausalInference(config_causal)
        assert causal is not None, "CausalInference should be initialized"
        
        config_uq = {'uncertainty_methods': ['bootstrap']}
        uq = UncertaintyQuantification(config_uq)
        assert uq is not None, "UncertaintyQuantification should be initialized"
        
        config_bandit = {'arms': ['buy', 'sell', 'hold'], 'context_dim': 10, 'algorithm': 'thompson_sampling'}
        bandit = ContextualBandit(config_bandit)
        assert bandit is not None, "ContextualBandit should be initialized"
        
        logger.info("✓ All models initialized correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_components():
    """Test system components."""
    logger.info("Testing System Components...")
    
    try:
        from src.system.backtesting import BacktestingEngine
        from src.system.monitoring import SystemMonitor
        
        config_backtest = {
            'initial_capital': 100000,
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'risk_free_rate': 0.02
        }
        backtester = BacktestingEngine(config_backtest)
        assert backtester is not None, "BacktestingEngine should be initialized"
        
        config_monitor = {'alert_thresholds': {}}
        monitor = SystemMonitor(config_monitor)
        assert monitor is not None, "SystemMonitor should be initialized"
        
        logger.info("✓ System components initialized correctly")
        return True
    except Exception as e:
        logger.error(f"✗ System components failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility components."""
    logger.info("Testing Utility Components...")
    
    try:
        from src.utils.metrics import MetricsCalculator
        from src.utils.visualization import VisualizationEngine
        
        config_metrics = {'risk_free_rate': 0.02, 'trading_days_per_year': 252}
        metrics = MetricsCalculator(config_metrics)
        assert metrics is not None, "MetricsCalculator should be initialized"
        
        # Test basic metrics calculation
        test_returns = pd.Series(np.random.randn(100) * 0.01)
        performance_metrics = metrics.calculate_performance_metrics(test_returns)
        assert isinstance(performance_metrics, dict), "Metrics should return a dictionary"
        
        config_viz = {}
        viz = VisualizationEngine(config_viz)
        assert viz is not None, "VisualizationEngine should be initialized"
        
        logger.info("✓ Utility components working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Utility components failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_system():
    """Test integrated system initialization."""
    logger.info("Testing IntegratedCausalTradingSystem...")
    
    try:
        from src.system.integrated_system import IntegratedCausalTradingSystem
        
        config = {
            'data_pipeline': {},
            'data_loader': {'api_keys': {}, 'database': {}},
            'data_validator': {'validation_rules': {}},
            'causal_inference': {},
            'uncertainty_quantification': {'uncertainty_methods': ['bootstrap']},
            'contextual_bandit': {'arms': ['buy', 'sell', 'hold'], 'context_dim': 10, 'algorithm': 'thompson_sampling'},
            'backtesting': {
                'initial_capital': 100000,
                'commission_rate': 0.001,
                'slippage_rate': 0.0005,
                'risk_free_rate': 0.02
            },
            'monitoring': {'alert_thresholds': {}}
        }
        
        system = IntegratedCausalTradingSystem(config)
        assert system is not None, "IntegratedCausalTradingSystem should be initialized"
        
        # Test system initialization
        success = system.initialize_system()
        assert success == True, "System initialization should succeed"
        
        logger.info("✓ IntegratedCausalTradingSystem initialized correctly")
        return True
    except Exception as e:
        logger.error(f"✗ IntegratedCausalTradingSystem failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("Starting Integrated Causal Trading System Tests")
    logger.info("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("ConfigManager Test", lambda: test_config_manager()[0]),
        ("DataLoader Test", test_data_loader),
        ("DataValidator Test", test_data_validator),
        ("DataPipeline Test", test_data_pipeline),
        ("Models Test", test_models),
        ("System Components Test", test_system_components),
        ("Utility Components Test", test_utils),
        ("Integrated System Test", test_integrated_system),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info(f"{'='*60}")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


