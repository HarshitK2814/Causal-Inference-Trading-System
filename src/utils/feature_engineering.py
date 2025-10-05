"""
Advanced Feature Engineering for Trading Signals
Creates interaction features, polynomial features, and regime indicators
Target: Improve ML accuracy from 44% to 50-52% before deep learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingFeatureEngineer:
    """
    Advanced feature engineering for trading data
    
    Creates 30+ features from basic OHLCV data:
    - Technical indicators (already have some)
    - Interaction features (volume × volatility, RSI × momentum, etc.)
    - Polynomial features (squares, cubes for non-linear patterns)
    - Rolling statistics (10/20/50 day windows)
    - Regime indicators (volatility regimes, trend strength)
    - Lagged features (past values as predictors)
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with OHLCV + basic indicators
        """
        self.data = data.copy()
        self.feature_names = []
        
    def add_interaction_features(self) -> pd.DataFrame:
        """
        Create interaction features between existing indicators
        
        Key interactions that capture trading dynamics:
        - Volume × Volatility: High volume + high volatility = strong moves
        - RSI × Momentum: Overbought momentum = reversal signal
        - MACD × Volume: Confirm trend strength
        """
        logger.info("Adding interaction features...")
        
        df = self.data
        
        # Volume interactions
        if 'volume_zscore' in df.columns and 'volatility' in df.columns:
            df['volume_volatility'] = df['volume_zscore'] * df['volatility']
            self.feature_names.append('volume_volatility')
        
        # RSI interactions
        if 'RSI' in df.columns:
            if 'momentum_5' in df.columns:
                df['rsi_momentum5'] = df['RSI'] * df['momentum_5']
                self.feature_names.append('rsi_momentum5')
            
            if 'momentum_20' in df.columns:
                df['rsi_momentum20'] = df['RSI'] * df['momentum_20']
                self.feature_names.append('rsi_momentum20')
            
            # RSI deviation from 50 (neutral)
            df['rsi_deviation'] = (df['RSI'] - 50) / 50
            self.feature_names.append('rsi_deviation')
        
        # MACD interactions
        if 'MACD' in df.columns and 'volume_zscore' in df.columns:
            df['macd_volume'] = df['MACD'] * df['volume_zscore']
            self.feature_names.append('macd_volume')
        
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            df['macd_signal_diff'] = df['MACD'] - df['MACD_signal']
            self.feature_names.append('macd_signal_diff')
        
        # Momentum interactions
        if 'momentum_5' in df.columns and 'momentum_20' in df.columns:
            df['momentum_ratio'] = df['momentum_5'] / (df['momentum_20'] + 1e-6)
            self.feature_names.append('momentum_ratio')
        
        # Volatility interactions
        if 'volatility' in df.columns and 'ATR' in df.columns:
            df['volatility_atr_ratio'] = df['volatility'] / (df['ATR'] + 1e-6)
            self.feature_names.append('volatility_atr_ratio')
        
        logger.info(f"  Added {len([f for f in self.feature_names if f in df.columns])} interaction features")
        return df
    
    def add_rolling_statistics(self, windows=[10, 20, 50]) -> pd.DataFrame:
        """
        Add rolling statistics to capture longer-term patterns
        
        Args:
            windows: List of lookback periods [10, 20, 50] days
        """
        logger.info(f"Adding rolling statistics (windows: {windows})...")
        
        df = self.data
        price_col = 'Close' if 'Close' in df.columns else 'close'
        
        for window in windows:
            # Rolling mean and std
            df[f'price_sma_{window}'] = df[price_col].rolling(window).mean()
            df[f'price_std_{window}'] = df[price_col].rolling(window).std()
            
            # Price relative to moving average
            df[f'price_to_sma_{window}'] = (df[price_col] / df[f'price_sma_{window}'] - 1)
            
            self.feature_names.extend([
                f'price_sma_{window}',
                f'price_std_{window}',
                f'price_to_sma_{window}'
            ])
            
            # Rolling volume statistics
            if 'Volume' in df.columns:
                df[f'volume_sma_{window}'] = df['Volume'].rolling(window).mean()
                df[f'volume_ratio_{window}'] = df['Volume'] / (df[f'volume_sma_{window}'] + 1e-6)
                self.feature_names.extend([
                    f'volume_sma_{window}',
                    f'volume_ratio_{window}'
                ])
        
        logger.info(f"  Added {len(windows) * 5} rolling features")
        return df
    
    def add_regime_indicators(self) -> pd.DataFrame:
        """
        Identify market regimes (volatility, trend, etc.)
        
        Regimes:
        - Low/Medium/High volatility
        - Uptrend/Downtrend/Sideways
        - High/Low volume
        """
        logger.info("Adding regime indicators...")
        
        df = self.data
        
        # Volatility regime (percentile-based)
        if 'volatility' in df.columns:
            vol_20pct = df['volatility'].quantile(0.20)
            vol_80pct = df['volatility'].quantile(0.80)
            
            df['volatility_regime'] = 1  # Medium
            df.loc[df['volatility'] < vol_20pct, 'volatility_regime'] = 0  # Low
            df.loc[df['volatility'] > vol_80pct, 'volatility_regime'] = 2  # High
            
            self.feature_names.append('volatility_regime')
        
        # Trend regime (based on momentum)
        if 'momentum_20' in df.columns:
            mom_20pct = df['momentum_20'].quantile(0.20)
            mom_80pct = df['momentum_20'].quantile(0.80)
            
            df['trend_regime'] = 1  # Sideways
            df.loc[df['momentum_20'] < mom_20pct, 'trend_regime'] = 0  # Downtrend
            df.loc[df['momentum_20'] > mom_80pct, 'trend_regime'] = 2  # Uptrend
            
            self.feature_names.append('trend_regime')
        
        # Volume regime
        if 'volume_zscore' in df.columns:
            vol_z_20pct = df['volume_zscore'].quantile(0.20)
            vol_z_80pct = df['volume_zscore'].quantile(0.80)
            
            df['volume_regime'] = 1  # Normal
            df.loc[df['volume_zscore'] < vol_z_20pct, 'volume_regime'] = 0  # Low
            df.loc[df['volume_zscore'] > vol_z_80pct, 'volume_regime'] = 2  # High
            
            self.feature_names.append('volume_regime')
        
        # RSI regime (overbought/oversold/neutral)
        if 'RSI' in df.columns:
            df['rsi_regime'] = 1  # Neutral
            df.loc[df['RSI'] < 30, 'rsi_regime'] = 0  # Oversold
            df.loc[df['RSI'] > 70, 'rsi_regime'] = 2  # Overbought
            
            self.feature_names.append('rsi_regime')
        
        logger.info(f"  Added 4 regime indicators")
        return df
    
    def add_lagged_features(self, lags=[1, 2, 3, 5]) -> pd.DataFrame:
        """
        Add lagged features (past values as predictors)
        
        Args:
            lags: List of lag periods [1, 2, 3, 5] days
        """
        logger.info(f"Adding lagged features (lags: {lags})...")
        
        df = self.data
        
        # Lag important features
        important_features = ['returns', 'volume_zscore', 'volatility', 'RSI', 'momentum_5']
        
        lagged_count = 0
        for feature in important_features:
            if feature not in df.columns:
                continue
            
            for lag in lags:
                lag_name = f'{feature}_lag{lag}'
                df[lag_name] = df[feature].shift(lag)
                self.feature_names.append(lag_name)
                lagged_count += 1
        
        logger.info(f"  Added {lagged_count} lagged features")
        return df
    
    def add_polynomial_features(self, degree=2, selected_features=None) -> pd.DataFrame:
        """
        Add polynomial features (squares, cubes) for non-linear patterns
        
        Only for selected important features to avoid explosion
        
        Args:
            degree: 2 for squares, 3 for cubes
            selected_features: List of features to polynomialize
        """
        if selected_features is None:
            selected_features = ['returns', 'volume_zscore', 'volatility', 'RSI', 'momentum_5', 'momentum_20']
        
        logger.info(f"Adding polynomial features (degree={degree})...")
        
        df = self.data
        poly_count = 0
        
        for feature in selected_features:
            if feature not in df.columns:
                continue
            
            # Square
            df[f'{feature}_squared'] = df[feature] ** 2
            self.feature_names.append(f'{feature}_squared')
            poly_count += 1
            
            if degree >= 3:
                # Cube
                df[f'{feature}_cubed'] = df[feature] ** 3
                self.feature_names.append(f'{feature}_cubed')
                poly_count += 1
        
        logger.info(f"  Added {poly_count} polynomial features")
        return df
    
    def engineer_all_features(self) -> Tuple[pd.DataFrame, list]:
        """
        Apply all feature engineering steps
        
        Returns:
            (DataFrame with all features, list of new feature names)
        """
        logger.info("\n" + "="*80)
        logger.info("ADVANCED FEATURE ENGINEERING")
        logger.info("="*80)
        
        initial_features = len(self.data.columns)
        
        # Apply all transformations
        self.data = self.add_interaction_features()
        self.data = self.add_rolling_statistics(windows=[10, 20, 50])
        self.data = self.add_regime_indicators()
        self.data = self.add_lagged_features(lags=[1, 2, 3, 5])
        self.data = self.add_polynomial_features(degree=2)
        
        # Drop NaN values created by rolling/lagging
        self.data = self.data.dropna()
        
        final_features = len(self.data.columns)
        
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("="*80)
        logger.info(f"Initial features:  {initial_features}")
        logger.info(f"Final features:    {final_features}")
        logger.info(f"Added features:    {final_features - initial_features}")
        logger.info(f"Samples remaining: {len(self.data)}")
        logger.info("")
        
        return self.data, self.feature_names


def prepare_features_for_ml(data: pd.DataFrame, target_col='target') -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare features for ML models
    
    Args:
        data: DataFrame with all features
        target_col: Name of target column
    
    Returns:
        (X, y, feature_names)
    """
    # Exclude non-feature columns
    exclude_cols = ['target', 'signal', 'next_return', 'Date', 'date', 
                    'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'open', 'high', 'low', 'close', 'volume', 'adj close']
    
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols].values
    y = data[target_col].values if target_col in data.columns else None
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, feature_cols


if __name__ == "__main__":
    # Test feature engineering
    logger.info("Testing feature engineering module...")
    
    # Create dummy data
    np.random.seed(42)
    n = 1000
    
    dummy_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(n) * 0.02),
        'Volume': np.random.lognormal(10, 0.5, n),
        'returns': np.random.randn(n) * 0.02,
        'volume_zscore': np.random.randn(n),
        'volatility': np.abs(np.random.randn(n) * 0.02),
        'RSI': np.random.uniform(20, 80, n),
        'MACD': np.random.randn(n) * 0.5,
        'MACD_signal': np.random.randn(n) * 0.5,
        'momentum_5': np.random.randn(n) * 0.05,
        'momentum_20': np.random.randn(n) * 0.08,
        'ATR': np.abs(np.random.randn(n) * 2)
    })
    
    # Engineer features
    engineer = TradingFeatureEngineer(dummy_data)
    engineered_data, new_features = engineer.engineer_all_features()
    
    logger.info(f"\n✓ Feature engineering successful!")
    logger.info(f"  Final shape: {engineered_data.shape}")
    logger.info(f"  New features added: {len(new_features)}")
    logger.info(f"\nSample new features:")
    for i, feat in enumerate(new_features[:10]):
        logger.info(f"  {i+1}. {feat}")
