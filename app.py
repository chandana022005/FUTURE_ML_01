"""
Sales Forecasting Application - Streamlit UI
Professional forecasting tool with multiple models and business insights
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Import custom modules
from data_utils import (
    load_or_generate_data,
    clean_sales_data,
    engineer_features,
    split_train_test,
    generate_sample_sales_data
)
from models import (
    LinearRegressionForecaster,
    ARIMAForecaster,
    ExponentialSmoothingForecaster,
    ProphetForecaster,
    get_available_models
)

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 20px;
    }
    h3 {
        color: #34495e;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def load_data(uploaded_file=None, use_sample=False):
    """Load data from file or generate sample."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = clean_sales_data(df)
        return df
    elif use_sample:
        return load_or_generate_data(generate_new=True)
    else:
        # Try to load existing sample
        sample_path = os.path.join(os.path.dirname(__file__), 'sample_sales_data.csv')
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            df = clean_sales_data(df)
            return df
        return load_or_generate_data(generate_new=True)


def plot_historical_data(df):
    """Create interactive plot of historical sales."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sales'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Add trend line
    z = np.polyfit(range(len(df)), df['sales'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=p(range(len(df))),
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Historical Sales Data with Trend',
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_forecast(train_df, test_df, forecast_df, model_name):
    """Create interactive forecast plot."""
    fig = go.Figure()
    
    # Historical training data
    fig.add_trace(go.Scatter(
        x=train_df['date'],
        y=train_df['sales'],
        mode='lines',
        name='Training Data',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Test data (if available)
    if test_df is not None and len(test_df) > 0:
        fig.add_trace(go.Scatter(
            x=test_df['date'],
            y=test_df['sales'],
            mode='lines+markers',
            name='Actual Test Data',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title=f'Sales Forecast - {model_name}',
        xaxis_title='Date',
        yaxis_title='Sales',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_model_comparison(results):
    """Create bar chart comparing model performance."""
    models = list(results.keys())
    mae_values = [results[m]['MAE'] for m in models]
    rmse_values = [results[m]['RMSE'] for m in models]
    
    fig = go.Figure(data=[
        go.Bar(name='MAE', x=models, y=mae_values, marker_color='#1f77b4'),
        go.Bar(name='RMSE', x=models, y=rmse_values, marker_color='#ff7f0e')
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Error',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig


def generate_business_insights(df, forecast_df, metrics):
    """Generate business insights from forecast."""
    insights = []
    
    # Calculate key statistics
    last_actual = df['sales'].iloc[-1]
    avg_forecast = forecast_df['forecast'].mean()
    forecast_trend = forecast_df['forecast'].iloc[-1] - forecast_df['forecast'].iloc[0]
    
    # Growth analysis
    if avg_forecast > last_actual:
        growth_pct = ((avg_forecast - last_actual) / last_actual) * 100
        insights.append(f"📈 **Expected Growth**: Forecasts indicate an average increase of {growth_pct:.1f}% compared to current sales.")
    else:
        decline_pct = ((last_actual - avg_forecast) / last_actual) * 100
        insights.append(f"📉 **Caution**: Forecasts suggest a potential decline of {decline_pct:.1f}% in average sales.")
    
    # Trend analysis
    if forecast_trend > 0:
        insights.append(f"✅ **Positive Trend**: Sales are projected to grow by ${forecast_trend:,.0f} over the forecast period.")
    else:
        insights.append(f"⚠️ **Declining Trend**: Sales may decrease by ${abs(forecast_trend):,.0f} over the forecast period.")
    
    # Model accuracy
    if 'MAPE' in metrics:
        mape = metrics['MAPE']
        if mape < 10:
            insights.append(f"🎯 **High Accuracy**: Model shows excellent accuracy with {mape:.1f}% error rate.")
        elif mape < 20:
            insights.append(f"✓ **Good Accuracy**: Model demonstrates reliable predictions with {mape:.1f}% error rate.")
        else:
            insights.append(f"⚡ **Moderate Accuracy**: Predictions have {mape:.1f}% error - consider additional data or features.")
    
    # Peak forecast
    max_forecast_idx = forecast_df['forecast'].idxmax()
    max_forecast_date = forecast_df.loc[max_forecast_idx, 'date']
    max_forecast_val = forecast_df.loc[max_forecast_idx, 'forecast']
    insights.append(f"🔝 **Peak Forecast**: Highest sales of ${max_forecast_val:,.0f} expected around {max_forecast_date.strftime('%B %Y')}.")
    
    # Minimum forecast
    min_forecast_idx = forecast_df['forecast'].idxmin()
    min_forecast_date = forecast_df.loc[min_forecast_idx, 'date']
    min_forecast_val = forecast_df.loc[min_forecast_idx, 'forecast']
    insights.append(f"📊 **Lowest Point**: Minimum sales of ${min_forecast_val:,.0f} projected around {min_forecast_date.strftime('%B %Y')}.")
    
    return insights


def main():
    # Header
    st.title("📈 Sales Forecasting System")
    st.markdown("### Professional forecasting with multiple time-series models")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source
        st.subheader("1. Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload CSV File"]
        )
        
        uploaded_file = None
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload your sales data (CSV with 'date' and 'sales' columns)",
                type=['csv']
            )
            st.info("📝 Required columns: 'date', 'sales'")
        
        # Model selection
        st.subheader("2. Select Model")
        model_options = {
            "Linear Regression": LinearRegressionForecaster,
            "ARIMA": ARIMAForecaster,
            "Exponential Smoothing": ExponentialSmoothingForecaster,
        }
        
        # Check if Prophet is available
        try:
            from prophet import Prophet
            model_options["Prophet (Advanced)"] = ProphetForecaster
        except ImportError:
            pass
        
        selected_model = st.selectbox(
            "Choose forecasting model:",
            list(model_options.keys())
        )
        
        # Forecast parameters
        st.subheader("3. Forecast Settings")
        forecast_periods = st.slider(
            "Forecast horizon (periods):",
            min_value=3,
            max_value=24,
            value=12,
            step=1
        )
        
        test_size = st.slider(
            "Test set size (for validation):",
            min_value=3,
            max_value=12,
            value=6,
            step=1
        )
        
        # Run forecast button
        run_forecast = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
    
    # Main content
    try:
        # Load data
        if data_source == "Upload CSV File" and uploaded_file is None:
            st.info("👈 Please upload a CSV file or switch to sample data to begin.")
            return
        
        df = load_data(uploaded_file, use_sample=(data_source == "Use Sample Data"))
        
        # Data overview
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Avg Sales", f"${df['sales'].mean():,.0f}")
        with col3:
            st.metric("Max Sales", f"${df['sales'].max():,.0f}")
        with col4:
            st.metric("Min Sales", f"${df['sales'].min():,.0f}")
        
        # Historical data plot
        st.plotly_chart(plot_historical_data(df), use_container_width=True)
        
        # Data preview
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True)
        
        # Run forecasting
        if run_forecast:
            with st.spinner('Training model and generating forecast...'):
                # Split data
                train_df, test_df = split_train_test(df, test_size=test_size)
                
                # Initialize and train model
                ModelClass = model_options[selected_model]
                model = ModelClass()
                model.fit(train_df)
                
                # Generate forecast
                forecast_df = model.predict(steps=forecast_periods)
                
                # Evaluate on test set
                metrics = model.evaluate(test_df)
                
                st.success("✅ Forecast completed successfully!")
                
                # Display forecast
                st.header("🔮 Forecast Results")
                
                # Metrics
                st.subheader("📏 Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"${metrics['MAE']:,.0f}")
                with col2:
                    st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
                with col3:
                    st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                with col4:
                    st.metric("R² Score", f"{metrics['R²']:.3f}")
                
                # Forecast plot
                st.subheader("📈 Forecast Visualization")
                st.plotly_chart(
                    plot_forecast(train_df, test_df, forecast_df, selected_model),
                    use_container_width=True
                )
                
                # Business insights
                st.header("💡 Business Insights")
                insights = generate_business_insights(df, forecast_df, metrics)
                
                for insight in insights:
                    st.markdown(insight)
                
                # Forecast table
                st.subheader("📊 Detailed Forecast")
                forecast_display = forecast_df.copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                forecast_display['forecast'] = forecast_display['forecast'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(forecast_display, use_container_width=True)
                
                # Download forecast
                st.download_button(
                    label="📥 Download Forecast (CSV)",
                    data=forecast_df.to_csv(index=False),
                    file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("👈 Configure settings and click 'Run Forecast' to generate predictions.")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
