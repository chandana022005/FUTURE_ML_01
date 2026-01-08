"""
Forecasting models: Linear Regression, ARIMA, Exponential Smoothing, Prophet
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    """Calculate forecasting metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'R²': round(r2, 3)
    }


class LinearRegressionForecaster:
    """Simple Linear Regression with time-based features."""
    
    def __init__(self):
        self.model = LinearRegression()
        self.name = "Linear Regression"
    
    def fit(self, df):
        """Fit the model using engineered features."""
        self.train_df = df.copy()
        
        # Use numeric time index
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['sales'].values
        
        self.model.fit(X, y)
        
    def predict(self, steps=12):
        """Generate future predictions."""
        last_idx = len(self.train_df)
        future_X = np.arange(last_idx, last_idx + steps).reshape(-1, 1)
        predictions = self.model.predict(future_X)
        
        # Create future dates
        last_date = self.train_df['date'].iloc[-1]
        freq = pd.infer_freq(self.train_df['date'])
        if freq is None:
            freq = 'M'
        
        future_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': predictions
        })
        
        return forecast_df
    
    def evaluate(self, test_df):
        """Evaluate on test set."""
        train_len = len(self.train_df)
        test_len = len(test_df)
        
        X_test = np.arange(train_len, train_len + test_len).reshape(-1, 1)
        predictions = self.model.predict(X_test)
        
        return calculate_metrics(test_df['sales'].values, predictions)


class ARIMAForecaster:
    """ARIMA time series model."""
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.name = "ARIMA"
    
    def fit(self, df):
        """Fit ARIMA model."""
        self.train_df = df.copy()
        
        try:
            self.model = ARIMA(df['sales'].values, order=self.order)
            self.fitted_model = self.model.fit()
        except Exception as e:
            # Fallback to simpler model
            self.order = (1, 0, 0)
            self.model = ARIMA(df['sales'].values, order=self.order)
            self.fitted_model = self.model.fit()
    
    def predict(self, steps=12):
        """Generate future predictions."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        predictions = self.fitted_model.forecast(steps=steps)
        
        # Create future dates
        last_date = self.train_df['date'].iloc[-1]
        freq = pd.infer_freq(self.train_df['date'])
        if freq is None:
            freq = 'M'
        
        future_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': predictions
        })
        
        return forecast_df
    
    def evaluate(self, test_df):
        """Evaluate on test set."""
        test_len = len(test_df)
        predictions = self.fitted_model.forecast(steps=test_len)
        
        return calculate_metrics(test_df['sales'].values, predictions)


class ExponentialSmoothingForecaster:
    """Exponential Smoothing (Holt-Winters) model."""
    
    def __init__(self, seasonal_periods=12):
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
        self.name = "Exponential Smoothing"
    
    def fit(self, df):
        """Fit Exponential Smoothing model."""
        self.train_df = df.copy()
        
        try:
            self.model = ExponentialSmoothing(
                df['sales'].values,
                seasonal_periods=self.seasonal_periods,
                trend='add',
                seasonal='add'
            )
            self.fitted_model = self.model.fit()
        except Exception as e:
            # Fallback to simpler model without seasonality
            self.model = ExponentialSmoothing(
                df['sales'].values,
                trend='add',
                seasonal=None
            )
            self.fitted_model = self.model.fit()
    
    def predict(self, steps=12):
        """Generate future predictions."""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet")
        
        predictions = self.fitted_model.forecast(steps=steps)
        
        # Create future dates
        last_date = self.train_df['date'].iloc[-1]
        freq = pd.infer_freq(self.train_df['date'])
        if freq is None:
            freq = 'M'
        
        future_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': predictions
        })
        
        return forecast_df
    
    def evaluate(self, test_df):
        """Evaluate on test set."""
        test_len = len(test_df)
        predictions = self.fitted_model.forecast(steps=test_len)
        
        return calculate_metrics(test_df['sales'].values, predictions)


class ProphetForecaster:
    """Facebook Prophet model."""
    
    def __init__(self):
        self.model = None
        self.name = "Prophet"
    
    def fit(self, df):
        """Fit Prophet model."""
        try:
            from prophet import Prophet
            
            self.train_df = df.copy()
            
            # Prophet requires 'ds' and 'y' columns
            prophet_df = pd.DataFrame({
                'ds': df['date'],
                'y': df['sales']
            })
            
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            self.model.fit(prophet_df)
            
        except ImportError:
            raise ImportError("Prophet not installed. Install with: pip install prophet")
    
    def predict(self, steps=12):
        """Generate future predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Create future dataframe
        freq = pd.infer_freq(self.train_df['date'])
        if freq is None:
            freq = 'M'
        
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self.model.predict(future)
        
        # Extract only future predictions
        forecast_df = forecast[['ds', 'yhat']].tail(steps).copy()
        forecast_df.columns = ['date', 'forecast']
        forecast_df = forecast_df.reset_index(drop=True)
        
        return forecast_df
    
    def evaluate(self, test_df):
        """Evaluate on test set."""
        # Create dataframe with train + test dates
        all_dates = pd.concat([
            self.train_df[['date']],
            test_df[['date']]
        ], ignore_index=True)
        
        future = pd.DataFrame({'ds': all_dates['date']})
        forecast = self.model.predict(future)
        
        # Extract test period predictions
        test_predictions = forecast['yhat'].tail(len(test_df)).values
        
        return calculate_metrics(test_df['sales'].values, test_predictions)


def get_available_models():
    """Return list of available forecasting models."""
    models = [
        LinearRegressionForecaster,
        ARIMAForecaster,
        ExponentialSmoothingForecaster
    ]
    
    # Check if Prophet is available
    try:
        from prophet import Prophet
        models.append(ProphetForecaster)
    except ImportError:
        pass
    
    return models
