# Stock Market Predictor 📈

A powerful web application for stock price prediction using machine learning and technical analysis. The app provides interactive visualizations of historical data and future price predictions.

## Features ✨

- **Real-time Stock Data**: Fetch live stock data from Yahoo Finance
- **Technical Analysis**: Moving averages (50, 100, 200 days)
- **ML Predictions**: Neural network-based price forecasting
- **Interactive Charts**: Dynamic visualizations using Matplotlib
- **User-friendly Interface**: Built with Streamlit

## Tech Stack 🛠️

- Python 3.11
- TensorFlow/Keras
- Streamlit
- Pandas
- yfinance
- Matplotlib/Seaborn
- scikit-learn

## Installation 🚀

1. **Clone the repository**
   ```bash
   git clone https://github.com/Dolphin-Syndrom/stockmarket-prediction-app.git
   cd stockmarket-prediction-app
   ```

2. **Set up virtual environment (recommended)**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage 💡

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser
   - Navigate to `http://localhost:8501`

3. **Using the Application**
   - Enter a stock symbol (e.g., 'GOOG' for Google)
   - View historical data and moving averages
   - Check price predictions

## Project Structure 📁

```
stockmarket-prediction-app/
├── app.py                      # Main application file
├── Stock Prediction Model.keras # Pre-trained ML model
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Features Explained 📊

### Moving Averages
- **MA50**: 50-day moving average (short-term trend)
- **MA100**: 100-day moving average (medium-term trend)
- **MA200**: 200-day moving average (long-term trend)

### Visualizations
1. Price vs MA50
2. Price vs MA50 vs MA100
3. Price vs MA100 vs MA200
4. Prediction Results

## Development 👨‍💻

### Prerequisites
- Python 3.11
- pip package manager
- Git

## Contributing 🤝

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Yahoo Finance API for real-time stock data
- TensorFlow team for the machine learning framework
- Streamlit team for the awesome web framework
