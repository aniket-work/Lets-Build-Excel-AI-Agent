"""
Advanced Visualization Utilities for Excel AI Agent

This module provides sophisticated visualization generation, customization,
and rendering utilities with support for multiple chart types and themes.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class VisualizationTheme:
    """Advanced theming configuration for visualizations"""
    primary_color: str = "#3b82f6"
    secondary_color: str = "#10b981"
    accent_color: str = "#f59e0b"
    danger_color: str = "#ef4444"
    background_color: str = "rgba(0,0,0,0)"
    grid_color: str = "rgba(255,255,255,0.1)"
    text_color: str = "#e2e8f0"
    font_family: str = "Inter, sans-serif"
    color_palette: List[str] = None
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                self.primary_color,
                self.secondary_color,
                self.accent_color,
                "#8b5cf6",
                "#ec4899",
                "#06b6d4",
                "#84cc16"
            ]


class AdvancedVisualizationEngine:
    """Sophisticated visualization generation engine"""
    
    def __init__(self, theme: Optional[VisualizationTheme] = None):
        self.theme = theme or VisualizationTheme()
        self._base_layout = self._create_base_layout()
    
    def _create_base_layout(self) -> Dict[str, Any]:
        """Create base layout configuration"""
        return {
            'plot_bgcolor': self.theme.background_color,
            'paper_bgcolor': self.theme.background_color,
            'font': {
                'family': self.theme.font_family,
                'color': self.theme.text_color,
                'size': 12
            },
            'title': {
                'font': {
                    'size': 16,
                    'color': self.theme.text_color,
                    'family': self.theme.font_family
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'gridcolor': self.theme.grid_color,
                'zerolinecolor': self.theme.grid_color,
                'color': self.theme.text_color
            },
            'yaxis': {
                'gridcolor': self.theme.grid_color,
                'zerolinecolor': self.theme.grid_color,
                'color': self.theme.text_color
            },
            'colorway': self.theme.color_palette
        }
    
    def create_distribution_analysis(
        self, 
        df: pd.DataFrame, 
        column: str,
        chart_type: str = "histogram"
    ) -> go.Figure:
        """
        Create sophisticated distribution analysis visualization
        
        Args:
            df: DataFrame containing data
            column: Column to analyze
            chart_type: Type of chart (histogram, box, violin)
            
        Returns:
            Plotly figure object
        """
        if chart_type == "histogram":
            fig = px.histogram(
                df,
                x=column,
                title=f"Distribution Analysis: {column}",
                marginal="box",
                color_discrete_sequence=[self.theme.primary_color]
            )
        elif chart_type == "box":
            fig = px.box(
                df,
                y=column,
                title=f"Box Plot Analysis: {column}",
                color_discrete_sequence=[self.theme.primary_color]
            )
        elif chart_type == "violin":
            fig = px.violin(
                df,
                y=column,
                title=f"Violin Plot Analysis: {column}",
                color_discrete_sequence=[self.theme.primary_color]
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        fig.update_layout(**self._base_layout)
        return fig
    
    def create_correlation_matrix(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create advanced correlation matrix visualization
        
        Args:
            df: DataFrame containing numeric data
            columns: Specific columns to include (all numeric if None)
            
        Returns:
            Plotly figure object
        """
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns]
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation analysis")
        
        correlation_matrix = numeric_df.corr()
        
        # Create masked correlation matrix for better visualization
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlation_matrix_masked = correlation_matrix.mask(mask)
        
        fig = px.imshow(
            correlation_matrix_masked,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            labels=dict(color="Correlation"),
            zmin=-1,
            zmax=1
        )
        
        # Add correlation values as text
        fig.update_traces(
            text=correlation_matrix_masked.round(2),
            texttemplate="%{text}",
            textfont={"size": 10, "color": self.theme.text_color}
        )
        
        fig.update_layout(**self._base_layout)
        return fig
    
    def create_time_series_analysis(
        self, 
        df: pd.DataFrame, 
        date_column: str, 
        value_columns: List[str],
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create sophisticated time series visualization
        
        Args:
            df: DataFrame containing time series data
            date_column: Column containing datetime values
            value_columns: Columns to plot as time series
            title: Custom title for the chart
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for i, column in enumerate(value_columns):
            color = self.theme.color_palette[i % len(self.theme.color_palette)]
            
            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df[column],
                mode='lines+markers',
                name=column,
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color)
            ))
        
        fig.update_layout(
            title=title or "Time Series Analysis",
            xaxis_title=date_column,
            yaxis_title="Values",
            hovermode='x unified',
            **self._base_layout
        )
        
        return fig
    
    def create_categorical_analysis(
        self, 
        df: pd.DataFrame, 
        category_column: str, 
        value_column: Optional[str] = None,
        chart_type: str = "bar"
    ) -> go.Figure:
        """
        Create advanced categorical data visualization
        
        Args:
            df: DataFrame containing categorical data
            category_column: Column with categories
            value_column: Column with values (for aggregation)
            chart_type: Type of chart (bar, pie, treemap)
            
        Returns:
            Plotly figure object
        """
        if value_column:
            # Aggregate data by category
            agg_df = df.groupby(category_column)[value_column].sum().reset_index()
            x_data = agg_df[category_column]
            y_data = agg_df[value_column]
            title = f"{value_column} by {category_column}"
        else:
            # Count occurrences
            value_counts = df[category_column].value_counts()
            x_data = value_counts.index
            y_data = value_counts.values
            title = f"Distribution of {category_column}"
        
        if chart_type == "bar":
            fig = px.bar(
                x=x_data,
                y=y_data,
                title=title,
                color_discrete_sequence=[self.theme.primary_color]
            )
        elif chart_type == "pie":
            fig = px.pie(
                values=y_data,
                names=x_data,
                title=title,
                color_discrete_sequence=self.theme.color_palette
            )
        elif chart_type == "treemap":
            fig = px.treemap(
                names=x_data,
                values=y_data,
                title=title,
                color_discrete_sequence=self.theme.color_palette
            )
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        fig.update_layout(**self._base_layout)
        return fig
    
    def create_scatter_analysis(
        self, 
        df: pd.DataFrame, 
        x_column: str, 
        y_column: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create advanced scatter plot with optional dimensions
        
        Args:
            df: DataFrame containing data
            x_column: Column for x-axis
            y_column: Column for y-axis
            color_column: Column for color coding
            size_column: Column for bubble sizes
            title: Custom title
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            title=title or f"{y_column} vs {x_column}",
            color_discrete_sequence=self.theme.color_palette,
            hover_data=[col for col in [color_column, size_column] if col]
        )
        
        # Add trendline if no color grouping
        if not color_column:
            # Calculate correlation
            correlation = df[x_column].corr(df[y_column])
            
            # Add trendline
            z = np.polyfit(df[x_column].dropna(), df[y_column].dropna(), 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=df[x_column],
                y=p(df[x_column]),
                mode='lines',
                name=f'Trend (r={correlation:.2f})',
                line=dict(color=self.theme.accent_color, dash='dash')
            ))
        
        fig.update_layout(**self._base_layout)
        return fig
    
    def create_comparison_chart(
        self, 
        df: pd.DataFrame, 
        categories: List[str], 
        values: List[str],
        chart_type: str = "grouped_bar"
    ) -> go.Figure:
        """
        Create sophisticated comparison visualizations
        
        Args:
            df: DataFrame containing data
            categories: Category columns
            values: Value columns to compare
            chart_type: Type of comparison chart
            
        Returns:
            Plotly figure object
        """
        if chart_type == "grouped_bar":
            fig = go.Figure()
            
            for i, value_col in enumerate(values):
                color = self.theme.color_palette[i % len(self.theme.color_palette)]
                
                fig.add_trace(go.Bar(
                    name=value_col,
                    x=df[categories[0]],
                    y=df[value_col],
                    marker_color=color
                ))
            
            fig.update_layout(
                title=f"Comparison of {', '.join(values)}",
                xaxis_title=categories[0],
                yaxis_title="Values",
                barmode='group',
                **self._base_layout
            )
        
        elif chart_type == "stacked_bar":
            fig = go.Figure()
            
            for i, value_col in enumerate(values):
                color = self.theme.color_palette[i % len(self.theme.color_palette)]
                
                fig.add_trace(go.Bar(
                    name=value_col,
                    x=df[categories[0]],
                    y=df[value_col],
                    marker_color=color
                ))
            
            fig.update_layout(
                title=f"Stacked Comparison of {', '.join(values)}",
                xaxis_title=categories[0],
                yaxis_title="Values",
                barmode='stack',
                **self._base_layout
            )
        
        elif chart_type == "radar":
            fig = go.Figure()
            
            for category in df[categories[0]].unique()[:5]:  # Limit to 5 categories
                subset = df[df[categories[0]] == category]
                color = self.theme.color_palette[
                    list(df[categories[0]].unique()).index(category) % len(self.theme.color_palette)
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=subset[values].iloc[0].values if len(subset) > 0 else [0] * len(values),
                    theta=values,
                    fill='toself',
                    name=str(category),
                    line_color=color
                ))
            
            fig.update_layout(
                title=f"Radar Chart Comparison",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        color=self.theme.text_color
                    ),
                    angularaxis=dict(
                        color=self.theme.text_color
                    )
                ),
                **self._base_layout
            )
        
        else:
            raise ValueError(f"Unsupported comparison chart type: {chart_type}")
        
        return fig
    
    def create_heatmap(
        self, 
        df: pd.DataFrame, 
        x_column: str, 
        y_column: str, 
        value_column: str,
        aggregation: str = "mean"
    ) -> go.Figure:
        """
        Create advanced heatmap visualization
        
        Args:
            df: DataFrame containing data
            x_column: Column for x-axis
            y_column: Column for y-axis
            value_column: Column for values
            aggregation: Aggregation method
            
        Returns:
            Plotly figure object
        """
        # Create pivot table for heatmap
        if aggregation == "mean":
            pivot_df = df.pivot_table(
                values=value_column,
                index=y_column,
                columns=x_column,
                aggfunc='mean'
            )
        elif aggregation == "sum":
            pivot_df = df.pivot_table(
                values=value_column,
                index=y_column,
                columns=x_column,
                aggfunc='sum'
            )
        elif aggregation == "count":
            pivot_df = df.pivot_table(
                values=value_column,
                index=y_column,
                columns=x_column,
                aggfunc='count'
            )
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        
        fig = px.imshow(
            pivot_df,
            title=f"Heatmap: {value_column} ({aggregation}) by {x_column} and {y_column}",
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        
        fig.update_layout(**self._base_layout)
        return fig


class StatisticalVisualizationUtils:
    """Utilities for statistical visualization and analysis"""
    
    @staticmethod
    def create_regression_plot(
        df: pd.DataFrame, 
        x_column: str, 
        y_column: str,
        theme: VisualizationTheme
    ) -> go.Figure:
        """Create regression analysis plot with confidence intervals"""
        fig = px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=f"Regression Analysis: {y_column} vs {x_column}",
            trendline="ols",
            color_discrete_sequence=[theme.primary_color]
        )
        
        # Calculate R-squared
        correlation = df[x_column].corr(df[y_column])
        r_squared = correlation ** 2
        
        # Add annotation with R-squared
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"R² = {r_squared:.3f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=theme.primary_color,
            borderwidth=1
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family=theme.font_family
        )
        
        return fig
    
    @staticmethod
    def create_distribution_comparison(
        df: pd.DataFrame, 
        column: str, 
        group_column: str,
        theme: VisualizationTheme
    ) -> go.Figure:
        """Create overlapping distribution comparison"""
        fig = go.Figure()
        
        groups = df[group_column].unique()
        
        for i, group in enumerate(groups):
            group_data = df[df[group_column] == group][column]
            color = theme.color_palette[i % len(theme.color_palette)]
            
            fig.add_trace(go.Histogram(
                x=group_data,
                name=str(group),
                opacity=0.7,
                marker_color=color,
                nbinsx=30
            ))
        
        fig.update_layout(
            title=f"Distribution Comparison: {column} by {group_column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            barmode='overlay',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family=theme.font_family
        )
        
        return fig


def create_dashboard_layout(figures: List[go.Figure], titles: List[str]) -> go.Figure:
    """
    Create a dashboard layout with multiple visualizations
    
    Args:
        figures: List of Plotly figures
        titles: List of titles for each figure
        
    Returns:
        Combined dashboard figure
    """
    n_figures = len(figures)
    
    if n_figures <= 2:
        rows, cols = 1, n_figures
    elif n_figures <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 3, 2
    
    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles[:n_figures],
        specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
    )
    
    # Add figures to subplots
    for i, source_fig in enumerate(figures[:n_figures]):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        for trace in source_fig.data:
            fig.add_trace(trace, row=row, col=col)
    
    fig.update_layout(
        title="Excel Data Analysis Dashboard",
        showlegend=True,
        height=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
