"""
Data Pipeline Module

This module handles the complete data processing pipeline for causal trading analysis.
Includes data ingestion, preprocessing, feature engineering, and data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main data pipeline class for processing financial and market data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing pipeline parameters
        """
        self.config = config
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, data_sources: List[str]) -> pd.DataFrame:
        """
        Load data from multiple sources.
        
        Args:
            data_sources: List of data source paths or APIs
            
        Returns:
            Combined DataFrame
        """
        logger.info("Loading data from sources...")
        # For now, return an empty DataFrame - this should be connected to DataLoader
        from src.data.data_loader import DataLoader
        loader = DataLoader(self.config)
        
        # Convert string paths to dict format expected by DataLoader
        data_source_configs = []
        for source in data_sources:
            if source.endswith('.csv'):
                data_source_configs.append({'type': 'csv', 'file_path': source})
            else:
                # Assume it's a symbol for yfinance
                data_source_configs.append({'type': 'yfinance', 'symbols': [source], 'period': '1y'})
        
        return loader.combine_data_sources(data_source_configs)
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data including cleaning and normalization.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Handle missing values
        data = data.ffill().bfill()
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Sort by index (assuming datetime index)
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        
        return data
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for causal analysis.
        
        Args:
            data: Preprocessed data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        # Store original columns
        original_cols = list(data.select_dtypes(include=[np.number]).columns)
        
        # Add basic technical indicators
        for col in original_cols:
            # Add returns
            data[f'{col}_return'] = data[col].pct_change()
            
            # Add rolling statistics
            for window in self.config.get('feature_engineering', {}).get('rolling_windows', [5, 10, 20]):
                data[f'{col}_ma_{window}'] = data[col].rolling(window=window).mean()
                data[f'{col}_std_{window}'] = data[col].rolling(window=window).std()
        
        # Fill NaN values in features with forward fill (for technical indicators at start)
        # But keep original data intact
        feature_cols = [c for c in data.columns if c not in original_cols]
        for col in feature_cols:
            data[col] = data[col].ffill()
        
        # Drop remaining NaN rows (first few rows that can't be filled)
        data = data.dropna()
        
        return data
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality and completeness.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        logger.info("Validating data quality...")
        
        from src.data.data_validator import DataValidator
        validator = DataValidator(self.config)
        
        # Check for missing values
        missing_stats = validator.check_missing_values(data)
        max_missing = self.config.get('max_missing_percentage', 10)
        
        for col, pct in missing_stats.items():
            if pct > max_missing:
                logger.error(f"Column {col} has {pct}% missing values (max allowed: {max_missing}%)")
                return False
        
        # Check if data is empty
        if len(data) == 0:
            logger.error("Data is empty")
            return False
        
        logger.info("Data validation passed")
        return True
        
    def run_pipeline(self, data_sources: List[str]) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Args:
            data_sources: List of data sources
            
        Returns:
            Processed and validated data
        """
        logger.info("Starting data pipeline...")
        
        # Load data
        raw_data = self.load_data(data_sources)
        
        # Preprocess data
        processed_data = self.preprocess_data(raw_data)
        
        # Engineer features
        features_data = self.engineer_features(processed_data)
        
        # Validate data
        if not self.validate_data(features_data):
            raise ValueError("Data validation failed")
            
        self.processed_data = features_data
        logger.info("Data pipeline completed successfully")
        
        return features_data