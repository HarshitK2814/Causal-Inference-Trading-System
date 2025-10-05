"""
Contextual Bandit Module

This module implements contextual bandit algorithms for adaptive trading strategies,
including Thompson Sampling, UCB, and other exploration-exploitation methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


class ContextualBandit:
    """
    Implements contextual bandit algorithms for adaptive trading.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the contextual bandit.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.arms = config.get('arms', [])  # Available trading actions
        self.context_dim = config.get('context_dim', 10)
        self.algorithm = config.get('algorithm', 'thompson_sampling')
        
        # Initialize models for each arm
        self.arm_models = {}
        self.arm_rewards = {}
        self.arm_counts = {}
        
        for arm in self.arms:
            self.arm_models[arm] = self._initialize_model()
            self.arm_rewards[arm] = []
            self.arm_counts[arm] = 0
            
    def _initialize_model(self):
        """
        Initialize model for arm.
        """
        if self.algorithm == 'thompson_sampling':
            return LinearRegression()
        elif self.algorithm == 'ucb':
            return LinearRegression()
        elif self.algorithm == 'epsilon_greedy':
            return RandomForestClassifier(n_estimators=10, random_state=42)
        else:
            return LinearRegression()
            
    def thompson_sampling(self, context: np.ndarray, arm: str) -> float:
        """
        Sample reward from Thompson Sampling posterior.
        
        Args:
            context: Context vector
            arm: Arm to sample from
            
        Returns:
            Sampled reward
        """
        if self.arm_counts[arm] == 0:
            # No data yet, return prior
            return np.random.normal(0, 1)
            
        # Get model predictions and uncertainty
        model = self.arm_models[arm]
        predicted_reward = model.predict(context.reshape(1, -1))[0]
        
        # Estimate uncertainty (simplified)
        if len(self.arm_rewards[arm]) > 1:
            uncertainty = np.std(self.arm_rewards[arm])
        else:
            uncertainty = 1.0
            
        # Sample from posterior
        sampled_reward = np.random.normal(predicted_reward, uncertainty)
        
        return sampled_reward
        
    def upper_confidence_bound(self, context: np.ndarray, arm: str, 
                              confidence_level: float = 0.95) -> float:
        """
        Calculate Upper Confidence Bound for arm.
        
        Args:
            context: Context vector
            arm: Arm to evaluate
            confidence_level: Confidence level for UCB
            
        Returns:
            UCB value
        """
        if self.arm_counts[arm] == 0:
            return float('inf')
            
        # Get model prediction
        model = self.arm_models[arm]
        predicted_reward = model.predict(context.reshape(1, -1))[0]
        
        # Calculate confidence bound
        confidence_width = np.sqrt(2 * np.log(self.arm_counts[arm]) / self.arm_counts[arm])
        ucb_value = predicted_reward + confidence_width
        
        return ucb_value
        
    def epsilon_greedy(self, context: np.ndarray, epsilon: float = 0.1) -> str:
        """
        Select arm using epsilon-greedy strategy.
        
        Args:
            context: Context vector
            epsilon: Exploration probability
            
        Returns:
            Selected arm
        """
        if np.random.random() < epsilon:
            # Explore: choose random arm
            return np.random.choice(self.arms)
        else:
            # Exploit: choose best arm
            arm_rewards = {}
            for arm in self.arms:
                if self.arm_counts[arm] > 0:
                    model = self.arm_models[arm]
                    reward = model.predict(context.reshape(1, -1))[0]
                    arm_rewards[arm] = reward
                else:
                    arm_rewards[arm] = 0
                    
            return max(arm_rewards, key=arm_rewards.get)
            
    def select_arm(self, context: np.ndarray) -> str:
        """
        Select arm based on current algorithm.
        
        Args:
            context: Context vector
            
        Returns:
            Selected arm
        """
        if self.algorithm == 'thompson_sampling':
            return self._thompson_sampling_selection(context)
        elif self.algorithm == 'ucb':
            return self._ucb_selection(context)
        elif self.algorithm == 'epsilon_greedy':
            return self.epsilon_greedy(context)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
    def _thompson_sampling_selection(self, context: np.ndarray) -> str:
        """
        Select arm using Thompson Sampling.
        """
        arm_samples = {}
        for arm in self.arms:
            sample = self.thompson_sampling(context, arm)
            arm_samples[arm] = sample
            
        return max(arm_samples, key=arm_samples.get)
        
    def _ucb_selection(self, context: np.ndarray) -> str:
        """
        Select arm using UCB.
        """
        arm_ucbs = {}
        for arm in self.arms:
            ucb = self.upper_confidence_bound(context, arm)
            arm_ucbs[arm] = ucb
            
        return max(arm_ucbs, key=arm_ucbs.get)
        
    def update_arm(self, arm: str, context: np.ndarray, reward: float):
        """
        Update arm model with new observation.
        
        Args:
            arm: Selected arm
            context: Context vector
            reward: Observed reward
        """
        logger.info(f"Updating arm {arm} with reward {reward}")
        
        # Update counts and rewards
        self.arm_counts[arm] += 1
        self.arm_rewards[arm].append(reward)
        
        # Retrain model for this arm
        if len(self.arm_rewards[arm]) > 1:
            # Prepare training data
            contexts = []
            rewards = []
            
            for i, (ctx, rew) in enumerate(zip(self.arm_rewards[arm], self.arm_rewards[arm])):
                contexts.append(ctx)
                rewards.append(rew)
                
            contexts = np.array(contexts)
            rewards = np.array(rewards)
            
            # Retrain model
            model = self.arm_models[arm]
            model.fit(contexts, rewards)
            
    def get_arm_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all arms.
        
        Returns:
            Dictionary with arm statistics
        """
        stats = {}
        
        for arm in self.arms:
            stats[arm] = {
                'count': self.arm_counts[arm],
                'mean_reward': np.mean(self.arm_rewards[arm]) if self.arm_rewards[arm] else 0,
                'std_reward': np.std(self.arm_rewards[arm]) if len(self.arm_rewards[arm]) > 1 else 0,
                'total_reward': np.sum(self.arm_rewards[arm])
            }
            
        return stats
        
    def regret_analysis(self, optimal_rewards: List[float], 
                       selected_arms: List[str], 
                       actual_rewards: List[float]) -> Dict[str, float]:
        """
        Calculate regret metrics.
        
        Args:
            optimal_rewards: List of optimal rewards for each round
            selected_arms: List of selected arms for each round
            actual_rewards: List of actual rewards received
            
        Returns:
            Dictionary with regret metrics
        """
        logger.info("Calculating regret analysis...")
        
        # Calculate cumulative regret
        cumulative_regret = []
        total_regret = 0
        
        for i, (optimal, actual) in enumerate(zip(optimal_rewards, actual_rewards)):
            regret = optimal - actual
            total_regret += regret
            cumulative_regret.append(total_regret)
            
        # Calculate average regret
        average_regret = total_regret / len(actual_rewards)
        
        # Calculate regret bounds (theoretical)
        if self.algorithm == 'thompson_sampling':
            # Thompson Sampling regret bound: O(sqrt(T log T))
            t = len(actual_rewards)
            theoretical_bound = np.sqrt(t * np.log(t))
        elif self.algorithm == 'ucb':
            # UCB regret bound: O(sqrt(T log T))
            t = len(actual_rewards)
            theoretical_bound = np.sqrt(t * np.log(t))
        else:
            theoretical_bound = None
            
        return {
            'cumulative_regret': cumulative_regret,
            'total_regret': total_regret,
            'average_regret': average_regret,
            'theoretical_bound': theoretical_bound
        }
        
    def adaptive_learning_rate(self, arm: str, context: np.ndarray) -> float:
        """
        Calculate adaptive learning rate for arm.
        
        Args:
            arm: Arm identifier
            context: Context vector
            
        Returns:
            Learning rate
        """
        # Simple adaptive learning rate based on arm performance
        if self.arm_counts[arm] == 0:
            return 1.0
            
        # Calculate performance-based learning rate
        recent_rewards = self.arm_rewards[arm][-10:]  # Last 10 rewards
        if len(recent_rewards) > 1:
            performance = np.mean(recent_rewards)
            # Higher performance = lower learning rate (more exploitation)
            learning_rate = 1.0 / (1.0 + performance)
        else:
            learning_rate = 1.0
            
        return max(0.01, min(1.0, learning_rate))  # Clamp between 0.01 and 1.0
        
    def contextual_linear_bandit(self, context: np.ndarray) -> str:
        """
        Implement contextual linear bandit algorithm.
        
        Args:
            context: Context vector
            
        Returns:
            Selected arm
        """
        logger.info("Running contextual linear bandit...")
        
        # Calculate expected rewards for each arm
        arm_rewards = {}
        
        for arm in self.arms:
            if self.arm_counts[arm] > 0:
                model = self.arm_models[arm]
                expected_reward = model.predict(context.reshape(1, -1))[0]
                
                # Add exploration bonus
                exploration_bonus = np.sqrt(2 * np.log(sum(self.arm_counts.values())) / 
                                         max(1, self.arm_counts[arm]))
                
                arm_rewards[arm] = expected_reward + exploration_bonus
            else:
                arm_rewards[arm] = float('inf')
                
        return max(arm_rewards, key=arm_rewards.get)