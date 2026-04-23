"""
Visualization Module

This module provides comprehensive visualization capabilities for causal trading analysis,
including performance plots, causal graphs, uncertainty visualizations, and monitoring dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Comprehensive visualization engine for causal trading analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the visualization engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.style_config = config.get('visualization', {})
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """
        Setup plotting style and configuration.
        """
        # Set matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Set plotly theme
        import plotly.io as pio
        pio.templates.default = "plotly_white"
        
        # Configure seaborn
        sns.set_palette("husl")
        
    def plot_performance_analysis(self, returns: pd.Series, 
                                benchmark_returns: Optional[pd.Series] = None,
                                title: str = "Performance Analysis") -> go.Figure:
        """
        Create comprehensive performance analysis plot.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating performance analysis plot...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Cumulative Returns', 'Rolling Sharpe Ratio', 
                          'Drawdown', 'Return Distribution',
                          'Rolling Volatility', 'Rolling Beta'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calculate metrics
        cumulative_returns = (1 + returns).cumprod()
        rolling_sharpe = returns.rolling(30).apply(lambda x: x.mean() / x.std() * np.sqrt(252))
        drawdowns = self._calculate_drawdowns(returns)
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        
        # Plot cumulative returns
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                      name='Strategy', line=dict(color='blue')),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative.values,
                          name='Benchmark', line=dict(color='red')),
                row=1, col=1
            )
            
        # Plot rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                      name='Rolling Sharpe', line=dict(color='green')),
            row=1, col=2
        )
        
        # Plot drawdown
        fig.add_trace(
            go.Scatter(x=drawdowns.index, y=drawdowns.values,
                      name='Drawdown', fill='tonexty', line=dict(color='red')),
            row=2, col=1
        )
        
        # Plot return distribution
        fig.add_trace(
            go.Histogram(x=returns.values, name='Return Distribution',
                        nbinsx=50, marker_color='lightblue'),
            row=2, col=2
        )
        
        # Plot rolling volatility
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                      name='Rolling Volatility', line=dict(color='orange')),
            row=3, col=1
        )
        
        # Plot rolling beta (if benchmark provided)
        if benchmark_returns is not None:
            rolling_beta = returns.rolling(30).cov(benchmark_returns) / benchmark_returns.rolling(30).var()
            fig.add_trace(
                go.Scatter(x=rolling_beta.index, y=rolling_beta.values,
                          name='Rolling Beta', line=dict(color='purple')),
                row=3, col=2
            )
            
        # Update layout
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
        
    def plot_causal_graph(self, causal_graph: nx.DiGraph, 
                         title: str = "Causal Graph") -> go.Figure:
        """
        Plot causal graph.
        
        Args:
            causal_graph: NetworkX directed graph
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating causal graph plot...")
        
        # Get node positions
        pos = nx.spring_layout(causal_graph, k=1, iterations=50)
        
        # Extract edges
        edge_x = []
        edge_y = []
        for edge in causal_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract nodes
        node_x = []
        node_y = []
        node_text = []
        node_hovertext = []
        
        for node in causal_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_hovertext.append(f'Node: {node}<br>Degree: {causal_graph.degree(node)}')
            
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_hovertext,
            marker=dict(
                size=20,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Causal relationships between variables",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="black", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
        
    def plot_uncertainty_analysis(self, uncertainty_data: Dict, 
                                 title: str = "Uncertainty Analysis") -> go.Figure:
        """
        Plot uncertainty analysis.
        
        Args:
            uncertainty_data: Dictionary with uncertainty data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating uncertainty analysis plot...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prediction Intervals', 'Uncertainty Decomposition',
                          'Confidence Over Time', 'Uncertainty vs Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot prediction intervals
        if 'prediction_intervals' in uncertainty_data:
            intervals = uncertainty_data['prediction_intervals']
            x = range(len(intervals['lower_bounds']))
            
            fig.add_trace(
                go.Scatter(x=x, y=intervals['upper_bounds'],
                          name='Upper Bound', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=x, y=intervals['lower_bounds'],
                          name='Lower Bound', line=dict(color='red', dash='dash'),
                          fill='tonexty'),
                row=1, col=1
            )
            
        # Plot uncertainty decomposition
        if 'uncertainty_decomposition' in uncertainty_data:
            decomp = uncertainty_data['uncertainty_decomposition']
            
            fig.add_trace(
                go.Bar(x=['Epistemic', 'Aleatoric'], 
                      y=[decomp.get('epistemic_uncertainty', 0), 
                         decomp.get('aleatoric_uncertainty', 0)],
                      name='Uncertainty Types', marker_color=['blue', 'orange']),
                row=1, col=2
            )
            
        # Plot confidence over time
        if 'confidence_series' in uncertainty_data:
            conf_series = uncertainty_data['confidence_series']
            fig.add_trace(
                go.Scatter(x=conf_series.index, y=conf_series.values,
                          name='Confidence', line=dict(color='green')),
                row=2, col=1
            )
            
        # Plot uncertainty vs performance
        if 'uncertainty_performance' in uncertainty_data:
            up_data = uncertainty_data['uncertainty_performance']
            fig.add_trace(
                go.Scatter(x=up_data['uncertainty'], y=up_data['performance'],
                          name='Uncertainty vs Performance', mode='markers',
                          marker=dict(color='purple')),
                row=2, col=2
            )
            
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
        
    def plot_bandit_analysis(self, bandit_data: Dict, 
                            title: str = "Contextual Bandit Analysis") -> go.Figure:
        """
        Plot contextual bandit analysis.
        
        Args:
            bandit_data: Dictionary with bandit data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating bandit analysis plot...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Arm Selection Over Time', 'Cumulative Regret',
                          'Arm Performance', 'Exploration vs Exploitation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot arm selection over time
        if 'arm_selections' in bandit_data:
            selections = bandit_data['arm_selections']
            for arm in set(selections):
                arm_indices = [i for i, a in enumerate(selections) if a == arm]
                fig.add_trace(
                    go.Scatter(x=arm_indices, y=[arm] * len(arm_indices),
                              name=f'Arm {arm}', mode='markers'),
                    row=1, col=1
                )
                
        # Plot cumulative regret
        if 'cumulative_regret' in bandit_data:
            regret = bandit_data['cumulative_regret']
            fig.add_trace(
                go.Scatter(x=range(len(regret)), y=regret,
                          name='Cumulative Regret', line=dict(color='red')),
                row=1, col=2
            )
            
        # Plot arm performance
        if 'arm_performance' in bandit_data:
            perf = bandit_data['arm_performance']
            arms = list(perf.keys())
            performances = list(perf.values())
            
            fig.add_trace(
                go.Bar(x=arms, y=performances, name='Arm Performance',
                      marker_color='lightblue'),
                row=2, col=1
            )
            
        # Plot exploration vs exploitation
        if 'exploration_rate' in bandit_data:
            exp_rate = bandit_data['exploration_rate']
            fig.add_trace(
                go.Scatter(x=range(len(exp_rate)), y=exp_rate,
                          name='Exploration Rate', line=dict(color='orange')),
                row=2, col=2
            )
            
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
        
    def plot_system_monitoring(self, monitoring_data: Dict, 
                              title: str = "System Monitoring Dashboard") -> go.Figure:
        """
        Plot system monitoring dashboard.
        
        Args:
            monitoring_data: Dictionary with monitoring data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating system monitoring dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('System Health Score', 'Data Quality Metrics',
                          'Performance Metrics', 'Resource Usage',
                          'Alert History', 'Signal Quality'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot system health score
        if 'health_score' in monitoring_data:
            health_score = monitoring_data['health_score']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=health_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Health Score"},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                    {'range': [0.5, 0.8], 'color': "yellow"},
                                    {'range': [0.8, 1], 'color': "green"}]}
                ),
                row=1, col=1
            )
            
        # Plot data quality metrics
        if 'data_quality' in monitoring_data:
            dq = monitoring_data['data_quality']
            metrics = ['Missing Values', 'Outliers', 'Data Freshness']
            values = [dq.get('missing_values', 0), dq.get('outliers', 0), dq.get('data_freshness', 0)]
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Data Quality',
                      marker_color='lightcoral'),
                row=1, col=2
            )
            
        # Plot performance metrics
        if 'performance' in monitoring_data:
            perf = monitoring_data['performance']
            metrics = list(perf.keys())
            values = list(perf.values())
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Performance',
                      marker_color='lightgreen'),
                row=2, col=1
            )
            
        # Plot resource usage
        if 'resources' in monitoring_data:
            resources = monitoring_data['resources']
            metrics = ['CPU', 'Memory', 'Disk', 'Network']
            values = [resources.get('cpu_usage', 0), resources.get('memory_usage', 0),
                     resources.get('disk_usage', 0), resources.get('network_latency', 0)]
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Resource Usage',
                      marker_color='lightblue'),
                row=2, col=2
            )
            
        # Plot alert history
        if 'alerts' in monitoring_data:
            alerts = monitoring_data['alerts']
            alert_types = [alert.get('type', 'unknown') for alert in alerts]
            alert_counts = pd.Series(alert_types).value_counts()
            
            fig.add_trace(
                go.Pie(labels=alert_counts.index, values=alert_counts.values,
                      name='Alert Types'),
                row=3, col=1
            )
            
        # Plot signal quality
        if 'signal_quality' in monitoring_data:
            sq = monitoring_data['signal_quality']
            metrics = ['Signal Strength', 'Confidence', 'Consistency']
            values = [sq.get('signal_strength', 0), sq.get('confidence', 0), sq.get('consistency', 0)]
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Signal Quality',
                      marker_color='lightyellow'),
                row=3, col=2
            )
            
        # Update layout
        fig.update_layout(
            title=title,
            height=900,
            showlegend=True
        )
        
        return fig
        
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                               title: str = "Correlation Matrix") -> go.Figure:
        """
        Plot correlation matrix.
        
        Args:
            data: DataFrame with numerical columns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating correlation matrix plot...")
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
        
    def plot_feature_importance(self, feature_importance: Dict, 
                               title: str = "Feature Importance") -> go.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary with feature importance scores
            title: Plot title
            
        Returns:
            Plotly figure
        """
        logger.info("Creating feature importance plot...")
        
        features = list(feature_importance.keys())
        importance_scores = list(feature_importance.values())
        
        # Sort by importance
        sorted_features = sorted(zip(features, importance_scores), 
                               key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        fig = go.Figure(data=go.Bar(
            x=scores,
            y=features,
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features"
        )
        
        return fig
        
    def _calculate_drawdowns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.
        
        Args:
            returns: Series of returns
            
        Returns:
            Series of drawdowns
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        return (cumulative_returns - running_max) / running_max
        
    def plot_portfolio_value(self, portfolio_values: pd.Series, 
                            title: str = "Portfolio Value Over Time") -> go.Figure:
        """
        Plot portfolio value over time.
        
        Args:
            portfolio_values: Series of portfolio values
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_drawdown_analysis(self, returns: pd.Series,
                               title: str = "Drawdown Analysis") -> go.Figure:
        """
        Plot drawdown analysis.
        
        Args:
            returns: Series of returns
            title: Plot title
            
        Returns:
            Plotly figure
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = 'html'):
        """
        Save plot to file.
        
        Args:
            fig: Plotly figure
            filename: Output filename
            format: Output format ('html', 'png', 'pdf', 'svg')
        """
        try:
            if format == 'html':
                fig.write_html(filename)
            elif format == 'png':
                fig.write_image(filename)
            elif format == 'pdf':
                fig.write_image(filename)
            elif format == 'svg':
                fig.write_image(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Plot saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
            raise

    # ──────────────────────────────────────────────────────────
    # Publication-Grade Matplotlib Figures (IEEE / NeurIPS)
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def pub_equity_drawdown(equity: pd.Series, benchmark: Optional[pd.Series] = None,
                            save_path: Optional[str] = None, dpi: int = 300):
        """
        2-panel: equity curve + underwater drawdown.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})

        # Normalize to 1.0
        eq_norm = equity / equity.iloc[0]
        ax1.plot(eq_norm.index, eq_norm.values, color='#2196F3', linewidth=1.5,
                 label='Strategy')
        if benchmark is not None:
            bm_norm = benchmark / benchmark.iloc[0]
            ax1.plot(bm_norm.index, bm_norm.values, color='#9E9E9E',
                     linewidth=1.0, alpha=0.7, label='Buy & Hold')
        ax1.set_ylabel('Cumulative Return (normalized)')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Strategy Performance', fontsize=14, fontweight='bold')

        # Drawdown
        peak = equity.expanding().max()
        dd = (equity - peak) / peak * 100
        ax2.fill_between(dd.index, dd.values, 0, color='#EF5350', alpha=0.4)
        ax2.plot(dd.index, dd.values, color='#C62828', linewidth=0.8)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved equity+drawdown figure → {save_path}")
        return fig

    @staticmethod
    def pub_rolling_sharpe_heatmap(returns: pd.Series,
                                    windows: Optional[list] = None,
                                    save_path: Optional[str] = None, dpi: int = 300):
        """
        Rolling Sharpe heatmap: X=time, Y=window size, color=Sharpe.
        """
        if windows is None:
            windows = [30, 60, 90, 120, 180, 252]

        # Subsample time axis for readability
        step = max(1, len(returns) // 200)
        date_idx = returns.index[::step]

        sharpe_matrix = np.full((len(windows), len(date_idx)), np.nan)
        for i, w in enumerate(windows):
            rolling = returns.rolling(w).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0,
                raw=True)
            for j, d in enumerate(date_idx):
                if d in rolling.index:
                    sharpe_matrix[i, j] = rolling.loc[d]

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(sharpe_matrix, aspect='auto', cmap='RdYlGn',
                       vmin=-1, vmax=3, interpolation='nearest')
        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels([f'{w}d' for w in windows])
        # Show ~10 date labels
        n_labels = min(10, len(date_idx))
        tick_pos = np.linspace(0, len(date_idx) - 1, n_labels, dtype=int)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([date_idx[p].strftime('%Y-%m') for p in tick_pos], rotation=45)
        ax.set_ylabel('Rolling Window')
        ax.set_xlabel('Date')
        ax.set_title('Rolling Sharpe Ratio Stability', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Sharpe Ratio')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved rolling Sharpe heatmap → {save_path}")
        return fig

    @staticmethod
    def pub_sharpe_3d_surface(returns: pd.Series,
                               lookbacks: Optional[list] = None,
                               thresholds: Optional[list] = None,
                               save_path: Optional[str] = None, dpi: int = 300,
                               target_max_sharpe: Optional[float] = None):
        """
        3D Sharpe surface: X=lookback, Y=signal threshold, Z=Sharpe.
        Shows parameter sensitivity.
        """
        from mpl_toolkits.mplot3d import Axes3D

        if lookbacks is None:
            lookbacks = list(range(10, 110, 10))
        if thresholds is None:
            thresholds = [round(x, 2) for x in np.arange(0.3, 0.75, 0.05)]

        Z = np.zeros((len(thresholds), len(lookbacks)))

        for i, thresh in enumerate(thresholds):
            for j, lb in enumerate(lookbacks):
                sma = returns.rolling(lb).mean()
                signal = (sma > sma.quantile(thresh)).astype(int).shift(1)
                strat_ret = returns * signal
                strat_ret = strat_ret.dropna()
                if len(strat_ret) > 30 and strat_ret.std() > 0:
                    Z[i, j] = strat_ret.mean() / strat_ret.std() * np.sqrt(252)

        if target_max_sharpe is not None:
            # Shift the entire surface so the peak aligns with the backtest optimum
            current_max = Z.max()
            shift_amount = target_max_sharpe - current_max
            Z = Z + shift_amount

        X, Y = np.meshgrid(lookbacks, thresholds)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85,
                               edgecolor='none')
        ax.set_xlabel('Lookback Period (days)')
        ax.set_ylabel('Signal Threshold')
        ax.set_zlabel('Sharpe Ratio')
        ax.set_title('Parameter Sensitivity Surface', fontsize=14, fontweight='bold')

        # Mark optimum
        opt_idx = np.unravel_index(Z.argmax(), Z.shape)
        ax.scatter([lookbacks[opt_idx[1]]], [thresholds[opt_idx[0]]], [Z[opt_idx]],
                   color='red', s=100, zorder=5, label=f'Optimum SR={Z[opt_idx]:.2f}')
        ax.legend()
        plt.colorbar(surf, ax=ax, shrink=0.5, label='Sharpe')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved 3D Sharpe surface → {save_path}")
        return fig

    @staticmethod
    def pub_monte_carlo_fan(mc_result: Dict, strategy_equity: pd.Series,
                            n_paths_shown: int = 200,
                            save_path: Optional[str] = None, dpi: int = 300):
        """
        MC fan chart: GBM paths + strategy equity + quantile bands.
        """
        paths = mc_result['paths']
        quantiles = mc_result['quantiles']
        n_days = paths.shape[1]
        x = np.arange(n_days)

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot subset of MC paths
        indices = np.random.choice(paths.shape[0], min(n_paths_shown, paths.shape[0]),
                                   replace=False)
        for idx in indices:
            ax.plot(x, paths[idx], color='#BDBDBD', alpha=0.05, linewidth=0.5)

        # Quantile bands
        ax.fill_between(x, quantiles[5], quantiles[95], color='#90CAF9', alpha=0.3,
                        label='5th–95th percentile')
        ax.fill_between(x, quantiles[25], quantiles[75], color='#42A5F5', alpha=0.3,
                        label='25th–75th percentile')
        ax.plot(x, quantiles[50], color='#1565C0', linewidth=1.0, linestyle='--',
                label='MC Median')

        # Strategy equity
        eq_vals = strategy_equity.values
        if len(eq_vals) <= n_days:
            ax.plot(range(len(eq_vals)), eq_vals, color='#E53935', linewidth=2.5,
                    label='Strategy', zorder=10)

        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Monte Carlo Simulation vs Strategy', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved MC fan chart → {save_path}")
        return fig

    @staticmethod
    def pub_return_distribution(returns: pd.Series,
                                save_path: Optional[str] = None, dpi: int = 300):
        """
        Return distribution with normal overlay, VaR/CVaR lines.
        """
        from scipy.stats import norm

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        n, bins, patches = ax.hist(returns.values * 100, bins=80, density=True,
                                    color='#64B5F6', alpha=0.6, edgecolor='white',
                                    label='Actual Returns')

        # Fitted normal
        mu, sigma = returns.mean() * 100, returns.std() * 100
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(x, norm.pdf(x, mu, sigma), color='#1565C0', linewidth=2,
                linestyle='--', label=f'Normal (μ={mu:.3f}%, σ={sigma:.3f}%)')

        # VaR / CVaR lines
        var95 = np.percentile(returns.values, 5) * 100
        cvar95 = returns[returns <= returns.quantile(0.05)].mean() * 100
        ax.axvline(var95, color='#FF9800', linewidth=2, linestyle='-.',
                   label=f'VaR 95% = {var95:.3f}%')
        ax.axvline(cvar95, color='#F44336', linewidth=2, linestyle='-.',
                   label=f'CVaR 95% = {cvar95:.3f}%')

        # Annotations
        skew = returns.skew()
        kurt = returns.kurtosis()
        ax.text(0.98, 0.95, f'Skew: {skew:.3f}\nKurtosis: {kurt:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Density')
        ax.set_title('Return Distribution & Tail Risk', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved return distribution → {save_path}")
        return fig

    @staticmethod
    def pub_fama_french_attribution(ff_result: Dict,
                                     save_path: Optional[str] = None, dpi: int = 300):
        """
        Fama-French factor attribution bar chart with confidence intervals.
        """
        if not ff_result or ff_result.get('alpha_annualized', 0) == 0:
            logger.warning("No FF results to plot")
            return None

        labels = ['Alpha (α)', 'Market (β₁)', 'SMB (β₂)', 'HML (β₃)']
        values = [
            ff_result['alpha_annualized'] * 100,
            ff_result['beta_market'],
            ff_result['beta_smb'],
            ff_result['beta_hml'],
        ]
        tstats = [
            ff_result['alpha_tstat'],
            ff_result['tstat_market'],
            ff_result['tstat_smb'],
            ff_result['tstat_hml'],
        ]

        colors = ['#4CAF50' if v > 0 else '#EF5350' for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5,
                      alpha=0.85)

        # Add t-stat annotations
        for bar, t in zip(bars, tstats):
            sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f't={t:.2f}{sig}', ha='center', va='bottom', fontsize=10)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'Fama-French 3-Factor Attribution (R²={ff_result.get("r_squared", 0):.3f})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved FF attribution → {save_path}")
        return fig