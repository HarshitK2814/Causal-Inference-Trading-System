"""
Uncertainty Quantification Module

This module provides methods for quantifying uncertainty in causal trading models,
including prediction intervals, confidence intervals, and model uncertainty estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)


class UncertaintyQuantification:
    """
    Quantifies uncertainty in causal trading models and predictions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the uncertainty quantification module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.uncertainty_methods = config.get('uncertainty_methods', ['bootstrap', 'conformal'])
        
    def bootstrap_prediction_intervals(self, model, X: np.ndarray, y: np.ndarray, 
                                     n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using bootstrap method.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level for confidence intervals
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        logger.info(f"Calculating bootstrap prediction intervals with {n_bootstrap} samples...")
        
        n_samples = len(X)
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model on bootstrap sample
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_boot, y_boot)
            
            # Make predictions on original data
            pred = model_copy.predict(X)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        # Calculate percentiles
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(predictions, upper_percentile, axis=0)
        
        return lower_bounds, upper_bounds
        
    def conformal_prediction_intervals(self, model, X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using conformal prediction.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            alpha: Significance level
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        logger.info("Calculating conformal prediction intervals...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate residuals on training data
        y_pred_train = model.predict(X_train)
        residuals = np.abs(y_train - y_pred_train)
        
        # Calculate conformal quantile
        conformal_quantile = np.quantile(residuals, 1 - alpha)
        
        # Make predictions on test data
        y_pred_test = model.predict(X_test)
        
        # Calculate intervals
        lower_bounds = y_pred_test - conformal_quantile
        upper_bounds = y_pred_test + conformal_quantile
        
        return lower_bounds, upper_bounds
        
    def model_uncertainty_estimation(self, model, X: np.ndarray, y: np.ndarray,
                                   cv_folds: int = 5) -> Dict[str, float]:
        """
        Estimate model uncertainty using cross-validation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with uncertainty metrics
        """
        logger.info(f"Estimating model uncertainty with {cv_folds}-fold CV...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        
        # Calculate uncertainty metrics
        mean_score = -np.mean(cv_scores)
        std_score = np.std(cv_scores)
        cv_uncertainty = std_score / np.sqrt(cv_folds)
        
        # Calculate prediction variance
        predictions = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            pred = model_copy.predict(X_val)
            predictions.extend(pred)
            
        prediction_variance = np.var(predictions)
        
        return {
            'cv_mean_score': mean_score,
            'cv_std_score': std_score,
            'cv_uncertainty': cv_uncertainty,
            'prediction_variance': prediction_variance,
            'confidence_interval': (mean_score - 1.96 * cv_uncertainty, 
                                  mean_score + 1.96 * cv_uncertainty)
        }
        
    def epistemic_uncertainty(self, model, X: np.ndarray) -> np.ndarray:
        """
        Estimate epistemic (model) uncertainty.
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            Array of epistemic uncertainty values
        """
        logger.info("Calculating epistemic uncertainty...")
        
        if hasattr(model, 'predict_proba'):
            # For probabilistic models
            predictions = model.predict_proba(X)
            epistemic_uncertainty = 1 - np.max(predictions, axis=1)
        else:
            # For ensemble models, use prediction variance
            if hasattr(model, 'estimators_'):
                predictions = []
                for estimator in model.estimators_:
                    pred = estimator.predict(X)
                    predictions.append(pred)
                predictions = np.array(predictions)
                epistemic_uncertainty = np.var(predictions, axis=0)
            else:
                # For single models, use distance-based uncertainty
                # This is a simplified approach
                epistemic_uncertainty = np.ones(len(X)) * 0.1
                
        return epistemic_uncertainty
        
    def aleatoric_uncertainty(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimate aleatoric (data) uncertainty.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            
        Returns:
            Array of aleatoric uncertainty values
        """
        logger.info("Calculating aleatoric uncertainty...")
        
        # Train model
        model.fit(X, y)
        
        # Calculate residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Estimate noise variance (aleatoric uncertainty)
        noise_variance = np.var(residuals)
        aleatoric_uncertainty = np.full(len(X), noise_variance)
        
        return aleatoric_uncertainty
        
    def bayesian_uncertainty(self, model, X: np.ndarray, y: np.ndarray,
                           n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using Bayesian methods.
        
        Args:
            model: Model to use
            X: Feature matrix
            y: Target values
            n_samples: Number of samples for Bayesian inference
            
        Returns:
            Dictionary with Bayesian uncertainty estimates
        """
        logger.info(f"Calculating Bayesian uncertainty with {n_samples} samples...")
        
        # This is a simplified Bayesian approach
        # In practice, use proper Bayesian models like Gaussian Process or Bayesian Neural Networks
        
        # Sample from posterior (simplified)
        posterior_samples = []
        for _ in range(n_samples):
            # Add noise to data for sampling
            X_noisy = X + np.random.normal(0, 0.01, X.shape)
            y_noisy = y + np.random.normal(0, 0.01, y.shape)
            
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_noisy, y_noisy)
            pred = model_copy.predict(X)
            posterior_samples.append(pred)
            
        posterior_samples = np.array(posterior_samples)
        
        # Calculate uncertainty metrics
        mean_predictions = np.mean(posterior_samples, axis=0)
        std_predictions = np.std(posterior_samples, axis=0)
        
        # Calculate credible intervals
        lower_bound = np.percentile(posterior_samples, 2.5, axis=0)
        upper_bound = np.percentile(posterior_samples, 97.5, axis=0)
        
        return {
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'posterior_samples': posterior_samples
        }
        
    def ensemble_uncertainty(self, models: List, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate uncertainty using ensemble of models.
        
        Args:
            models: List of trained models
            X: Feature matrix
            
        Returns:
            Dictionary with ensemble uncertainty estimates
        """
        logger.info(f"Calculating ensemble uncertainty with {len(models)} models...")
        
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        # Calculate ensemble statistics
        mean_predictions = np.mean(predictions, axis=0)
        std_predictions = np.std(predictions, axis=0)
        
        # Calculate prediction intervals
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean_predictions': mean_predictions,
            'std_predictions': std_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'all_predictions': predictions
        }
        
    def uncertainty_decomposition(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose total uncertainty into epistemic and aleatoric components.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with uncertainty decomposition
        """
        logger.info("Decomposing uncertainty into epistemic and aleatoric components...")
        
        # Calculate epistemic uncertainty
        epistemic = self.epistemic_uncertainty(model, X)
        
        # Calculate aleatoric uncertainty
        aleatoric = self.aleatoric_uncertainty(model, X, y)
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        return {
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total,
            'epistemic_ratio': epistemic / total,
            'aleatoric_ratio': aleatoric / total
        }