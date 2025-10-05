"""
Data Validator Module

This module provides data validation and quality checks for the causal trading system.
Ensures data integrity and completeness of financial datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and integrity for causal trading analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data validator with configuration.
        
        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config
        self.validation_rules = config.get('validation_rules', {})
        
    def check_missing_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing values in the dataset.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with missing value percentages by column
        """
        logger.info("Checking for missing values...")
        
        missing_stats = {}
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            missing_percentage = (missing_count / len(data)) * 100
            missing_stats[column] = missing_percentage
            
        return missing_stats
        
    def check_data_types(self, data: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate data types against expected types.
        
        Args:
            data: DataFrame to validate
            expected_types: Dictionary mapping column names to expected types
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating data types...")
        
        type_validation = {}
        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                type_validation[column] = actual_type == expected_type
            else:
                type_validation[column] = False
                
        return type_validation
        
    def check_outliers(self, data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Detect outliers in specified columns.
        
        Args:
            data: DataFrame to validate
            columns: List of columns to check for outliers
            method: Method for outlier detection ('iqr' or 'zscore')
            
        Returns:
            Dictionary with outlier indices by column
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        outliers = {}
        
        for column in columns:
            if column in data.columns:
                if method == 'iqr':
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                    outlier_mask = z_scores > 3
                    
                outliers[column] = data[outlier_mask].index.tolist()
                
        return outliers
        
    def check_data_consistency(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Check for data consistency issues.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with consistency check results
        """
        logger.info("Checking data consistency...")
        
        consistency_checks = {}
        
        # Check for duplicate rows
        consistency_checks['no_duplicates'] = not data.duplicated().any()
        
        # Check for negative prices (if price columns exist)
        price_columns = [col for col in data.columns if 'price' in col.lower() or 'close' in col.lower()]
        if price_columns:
            consistency_checks['no_negative_prices'] = not (data[price_columns] < 0).any().any()
        
        # Check for future dates (if date columns exist)
        date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            current_date = pd.Timestamp.now()
            future_dates = data[date_columns].max() > current_date
            consistency_checks['no_future_dates'] = not future_dates.any()
            
        return consistency_checks
        
    def validate_trading_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Comprehensive validation for trading data.
        
        Args:
            data: DataFrame with trading data
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Performing comprehensive trading data validation...")
        
        validation_results = {
            'missing_values': self.check_missing_values(data),
            'data_types': self.check_data_types(data, self.validation_rules.get('expected_types', {})),
            'outliers': self.check_outliers(data, self.validation_rules.get('outlier_columns', [])),
            'consistency': self.check_data_consistency(data),
            'is_valid': True
        }
        
        # Determine overall validity
        if any(validation_results['missing_values'].values()) > self.validation_rules.get('max_missing_percentage', 10):
            validation_results['is_valid'] = False
            
        if not all(validation_results['data_types'].values()):
            validation_results['is_valid'] = False
            
        if not all(validation_results['consistency'].values()):
            validation_results['is_valid'] = False
            
        return validation_results
        
    def generate_validation_report(self, validation_results: Dict) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_results: Results from validate_trading_data
            
        Returns:
            Formatted validation report string
        """
        report = "=== Data Validation Report ===\n\n"
        
        # Missing values report
        report += "Missing Values:\n"
        for column, percentage in validation_results['missing_values'].items():
            report += f"  {column}: {percentage:.2f}%\n"
        report += "\n"
        
        # Data types report
        report += "Data Types Validation:\n"
        for column, is_valid in validation_results['data_types'].items():
            status = "✓" if is_valid else "✗"
            report += f"  {column}: {status}\n"
        report += "\n"
        
        # Consistency report
        report += "Consistency Checks:\n"
        for check, passed in validation_results['consistency'].items():
            status = "✓" if passed else "✗"
            report += f"  {check}: {status}\n"
        report += "\n"
        
        # Overall status
        overall_status = "✓ VALID" if validation_results['is_valid'] else "✗ INVALID"
        report += f"Overall Status: {overall_status}\n"
        
        return report