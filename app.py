import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

plt.style.use('seaborn-v0_8') 
sns.set_theme(style="whitegrid")


COLORS = {
    'stock': '#2ecc71',  
    'ma50': '#e74c3c',   
    'ma100': '#3498db',  
    'ma200': '#9b59b6',  
    'prediction': '#e67e22'
}

def load_stock_model():
    """Load the pre-trained stock prediction model"""
    try:
        return load_model('Stock Prediction Model.keras')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_stock_data(symbol, start_date, end_date):
    """Fetch stock data for given symbol and date range"""
    try:
        data = yf.download(symbol, start_date, end_date)
        if data.empty:
            st.warning("No data found for this stock symbol")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_moving_averages(data):
    """Calculate moving averages for different time periods"""
    return {
        'MA50': data.Close.rolling(window=50, min_periods=1).mean(),
        'MA100': data.Close.rolling(window=100, min_periods=1).mean(),
        'MA200': data.Close.rolling(window=200, min_periods=1).mean()
    }

def plot_stock_comparison(data, moving_avgs, title, ma_list):
    """Create comparison plots for stock prices and moving averages"""
    fig = plt.figure(figsize=(12, 6))
    
    # Plot actual stock price
    plt.plot(data.Close, color=COLORS['stock'], label='Stock Price', linewidth=2)
    
    # Plot selected moving averages
    for ma in ma_list:
        color_key = ma.lower()
        plt.plot(moving_avgs[ma], color=COLORS[color_key], 
                label=f"{ma} Moving Average", linewidth=1.5)
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add padding to prevent label cutoff
    plt.tight_layout()
    
    return fig

def prepare_prediction_data(data_train, data_test):
    """Prepare data for stock price prediction"""
    scaler = MinMaxScaler(feature_range=(0,1))
    
    # Combine last 100 days of training data with test data
    past_100_days = data_train.tail(100)
    full_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
    scaled_data = scaler.fit_transform(full_test_data)
    
    return scaled_data, scaler

def main():
    st.set_page_config(page_title="Stock Market Predictor", layout="wide")
    st.title('📈 Stock Market Predictor')
    
    # User inputs
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        stock = st.text_input('Enter Stock Symbol', 'GOOG')
    with col2:
        start_date = '2012-01-01'
    with col3:
        end_date = '2022-12-31'
    
    
    data = get_stock_data(stock, start_date, end_date)
    model = load_stock_model()
    
    
    st.subheader('📊 Historical Stock Data')
    st.dataframe(data.style.highlight_max(axis=0))
    
   
    moving_avgs = create_moving_averages(data)
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Price vs MA50')
        fig1 = plot_stock_comparison(data, moving_avgs, 
                                   '50-Day Moving Average Analysis', ['MA50'])
        st.pyplot(fig1)
        
        st.subheader('Price vs MA100 vs MA200')
        fig3 = plot_stock_comparison(data, moving_avgs, 
                                   'Long-term Trend Analysis', ['MA100', 'MA200'])
        st.pyplot(fig3)
    
    with col2:
        st.subheader('Price vs MA50 vs MA100')
        fig2 = plot_stock_comparison(data, moving_avgs, 
                                   'Medium-term Trend Analysis', ['MA50', 'MA100'])
        st.pyplot(fig2)
        
        # Prepare data for prediction
        data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
        
        scaled_data, scaler = prepare_prediction_data(data_train, data_test)
        
        # Create sequences for prediction
        x = []
        y = []
        for i in range(100, scaled_data.shape[0]):
            x.append(scaled_data[i-100:i])
            y.append(scaled_data[i,0])
        
        x, y = np.array(x), np.array(y)
        
        # Make predictions
        predictions = model.predict(x) * (1/scaler.scale_)
        actual_values = y * (1/scaler.scale_)
        
        # Plot predictions
        st.subheader(' Prediction Results')
        fig4 = plt.figure(figsize=(10, 6))
        plt.plot(predictions, 'r', label='Predicted Price', linewidth=2)
        plt.plot(actual_values, 'g', label='Actual Price', linewidth=2)
        plt.title('Price Prediction Analysis', fontsize=14, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig4)

if __name__ == "__main__":
    main()

