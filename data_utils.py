"""
Data generation, cleaning, and feature engineering utilities
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_sample_sales_data(
    start_date: str = '2021-01-01',
    periods: int = 36,
    freq: str = 'M',
    base_sales: float = 10000,
    growth_rate: float = 0.03,
    seasonality_strength: float = 0.2,
    noise_level: float = 0.1,
    add_anomalies: bool = True
) -> pd.DataFrame:
    """
    Generate realistic sales data with trend, seasonality, and noise.
    
    Parameters:
    - start_date: Start date for the time series
    - periods: Number of time periods to generate
    - freq: Frequency ('D' daily, 'W' weekly, 'M' monthly)
    - base_sales: Base sales amount
    - growth_rate: Monthly growth rate (0.03 = 3% growth)
    - seasonality_strength: Strength of seasonal component (0-1)
    - noise_level: Random noise level (0-1)
    - add_anomalies: Add occasional sales spikes/drops
    
    Returns:
    - DataFrame with date and sales columns
    """
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate trend component
    trend = base_sales * (1 + growth_rate) ** np.arange(periods)
    
    # Generate seasonal component (annual seasonality)
    if freq == 'M':
        seasonal_period = 12
    elif freq == 'W':
        seasonal_period = 52
    else:  # Daily
        seasonal_period = 365
    
    seasonal = seasonality_strength * base_sales * np.sin(2 * np.pi * np.arange(periods) / seasonal_period)
    
    # Generate random noise
    noise = noise_level * base_sales * np.random.randn(periods)
    
    # Combine components
    sales = trend + seasonal + noise
    
    # Add occasional anomalies (spikes or drops)
    if add_anomalies:
        anomaly_indices = np.random.choice(periods, size=max(1, periods // 12), replace=False)
        for idx in anomaly_indices:
            if np.random.rand() > 0.5:
                sales[idx] *= 1.5  # Spike
            else:
                sales[idx] *= 0.7  # Drop
    
    # Ensure non-negative sales
    sales = np.maximum(sales, 0)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })
    
    return df


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sales data: handle missing values, outliers, and format dates.
    
    Parameters:
    - df: DataFrame with 'date' and 'sales' columns
    
    Returns:
    - Cleaned DataFrame
    """
    df = df.copy()
    
    # Convert date to datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicate dates (keep last)
    df = df.drop_duplicates(subset=['date'], keep='last')
    
    # Handle missing values
    df['sales'] = df['sales'].fillna(df['sales'].median())
    
    # Remove extreme outliers (beyond 3 standard deviations)
    mean_sales = df['sales'].mean()
    std_sales = df['sales'].std()
    df['sales'] = df['sales'].clip(
        lower=max(0, mean_sales - 3 * std_sales),
        upper=mean_sales + 3 * std_sales
    )
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for forecasting.
    
    Parameters:
    - df: DataFrame with 'date' and 'sales' columns
    
    Returns:
    - DataFrame with additional features
    """
    df = df.copy()
    
    # Extract time components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # Cyclical encoding for month (captures seasonality)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lagged features
    for lag in [1, 3, 6, 12]:
        df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df[f'sales_rolling_mean_{window}'] = df['sales'].rolling(window=window, min_periods=1).mean()
        df[f'sales_rolling_std_{window}'] = df['sales'].rolling(window=window, min_periods=1).std()
    
    # Growth rate
    df['sales_pct_change'] = df['sales'].pct_change()
    
    return df


def split_train_test(df: pd.DataFrame, test_size: int = 12) -> tuple:
    """
    Split data into train and test sets.
    
    Parameters:
    - df: DataFrame with time series data
    - test_size: Number of periods for test set
    
    Returns:
    - train_df, test_df
    """
    split_idx = len(df) - test_size
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def load_or_generate_data(filepath: Optional[str] = None, generate_new: bool = False) -> pd.DataFrame:
    """
    Load data from file or generate sample data.
    
    Parameters:
    - filepath: Path to CSV file
    - generate_new: Force generation of new sample data
    
    Returns:
    - DataFrame with sales data
    """
    if filepath and not generate_new:
        try:
            df = pd.read_csv(filepath)
            df = clean_sales_data(df)
            return df
        except FileNotFoundError:
            pass
    
    # Generate sample data
    df = generate_sample_sales_data(
        start_date='2021-01-01',
        periods=48,
        freq='M',
        base_sales=10000,
        growth_rate=0.025,
        seasonality_strength=0.3,
        noise_level=0.1,
        add_anomalies=True
    )
    
    return df
