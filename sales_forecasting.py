import os
from datetime import datetime
import numpy as np
import pandas as pd


def load_or_generate_data(path: str) -> pd.DataFrame:
    """Load sales data if available; else generate a synthetic monthly dataset."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Expect columns: date, sales
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    # Generate 36 months of synthetic data with mild seasonality
    rng = pd.date_range(start='2022-01-01', periods=36, freq='MS')
    base = 100 + np.linspace(0, 20, len(rng))
    season = 10 * np.sin(np.arange(len(rng)) * 2 * np.pi / 12)
    noise = np.random.normal(0, 5, size=len(rng))
    sales = base + season + noise
    return pd.DataFrame({'date': rng, 'sales': sales})


def moving_average(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def make_forecast(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    df = df.copy()
    df['ma3'] = moving_average(df['sales'], window=3)
    last_date = df['date'].iloc[-1]
    last_ma = df['ma3'].iloc[-1]
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
    # Simple baseline: repeat last moving average value
    forecast_values = np.full(shape=horizon, fill_value=last_ma)
    return pd.DataFrame({'date': future_dates, 'forecast': forecast_values})


def evaluate_baseline(df: pd.DataFrame) -> float:
    """Compute RMSE of MA(3) one-step baseline on historical data."""
    df = df.copy()
    df['ma3'] = moving_average(df['sales'], window=3)
    # Shift MA to form prediction for next period
    df['pred'] = df['ma3'].shift(1)
    valid = df.dropna()
    rmse = np.sqrt(np.mean((valid['sales'] - valid['pred']) ** 2))
    return float(rmse)


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'sales.csv')
    df = load_or_generate_data(data_path)
    rmse = evaluate_baseline(df)
    fc = make_forecast(df, horizon=12)
    print('Historical samples:', len(df))
    print('Baseline RMSE (MA3 one-step):', round(rmse, 2))
    print('\nNext 12 months forecast (baseline):')
    print(fc.head(12).to_string(index=False))
    # Optional plotting if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(df['date'], df['sales'], label='Sales')
        plt.plot(df['date'], moving_average(df['sales'], 3), label='MA3')
        plt.plot(fc['date'], fc['forecast'], label='Forecast')
        plt.legend(); plt.tight_layout(); plt.show()
    except Exception as e:
        # Plotting is optional; keep CLI clean
        pass


if __name__ == '__main__':
    main()
