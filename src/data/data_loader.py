"""
Data Loader Module

This module handles loading data from various sources including APIs, databases,
and file systems for the causal trading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import requests
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading data from multiple sources for causal trading analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.database_config = config.get('database', {})
        
    def load_from_yfinance(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Load financial data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols
            period: Time period for data
            
        Returns:
            DataFrame with financial data
        """
        logger.info(f"Loading data from Yahoo Finance for symbols: {symbols}")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                data[symbol] = hist
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                
        return data
        
    def load_from_api(self, endpoint: str, params: Dict) -> pd.DataFrame:
        """
        Load data from external API.
        
        Args:
            endpoint: API endpoint URL
            params: API parameters
            
        Returns:
            DataFrame with API data
        """
        logger.info(f"Loading data from API: {endpoint}")
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error loading data from API: {e}")
            return pd.DataFrame()
            
    def load_from_database(self, query: str) -> pd.DataFrame:
        """
        Load data from database.
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with database data
        """
        logger.info("Loading data from database")
        
        try:
            conn = sqlite3.connect(self.database_config.get('path', 'data.db'))
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return pd.DataFrame()
            
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with CSV data
        """
        logger.info(f"Loading data from CSV: {file_path}")
        
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return pd.DataFrame()
            
    def combine_data_sources(self, data_sources: List[Dict]) -> pd.DataFrame:
        """
        Combine data from multiple sources.
        
        Args:
            data_sources: List of data source configurations
            
        Returns:
            Combined DataFrame
        """
        logger.info("Combining data from multiple sources")
        
        combined_data = []
        
        for source in data_sources:
            source_type = source.get('type')
            
            if source_type == 'yfinance':
                data = self.load_from_yfinance(
                    source.get('symbols', []),
                    source.get('period', '1y')
                )
            elif source_type == 'api':
                data = self.load_from_api(
                    source.get('endpoint'),
                    source.get('params', {})
                )
            elif source_type == 'database':
                data = self.load_from_database(source.get('query'))
            elif source_type == 'csv':
                data = self.load_from_csv(source.get('file_path'))
            else:
                logger.warning(f"Unknown data source type: {source_type}")
                continue
                
            if not data.empty:
                combined_data.append(data)
                
        if combined_data:
            return pd.concat(combined_data, ignore_index=True)
        else:
            return pd.DataFrame()