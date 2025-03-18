from flask import Flask, request, jsonify, render_template
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load_model('stock_price_lstm_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Serve the frontend

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    prediction_date = request.form['prediction_date']
    prediction_time = request.form.get('prediction_time', '16:00')  # Default to 4 PM

    try:
        # Validate date and time
        now = datetime.now()
        selected_datetime = datetime.strptime(f"{prediction_date} {prediction_time}", "%Y-%m-%d %H:%M")
        
        if selected_datetime < now:
            return jsonify({'error': 'Please enter a future date and time for prediction.'}), 400
        
        # Check if the selected time is during market hours (9 AM - 4 PM)
        market_open = datetime.strptime("09:00", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        selected_time = selected_datetime.time()

        if selected_time < market_open or selected_time > market_close:
            return jsonify({'error': 'Stock market is closed. Predictions are only available between 9 AM and 4 PM.'}), 400

        # Fetch stock data
        stock = yf.Ticker(ticker)
        data_1y = stock.history(period="1y")  # 1-year data
        data_1m = stock.history(period="1mo")  # 1-month data
        data_1w = stock.history(period="1wk")  # 1-week data

        if data_1y.empty:
            return jsonify({'error': 'Invalid ticker or no data available'}), 400

        # Fetch stock information
        info = stock.info
        logo_url = info.get('logo_url', '')
        company_name = info.get('longName', 'N/A')
        sector = info.get('sector', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        current_price = float(info.get('currentPrice', 0))
        day_high = float(info.get('dayHigh', 0))
        day_low = float(info.get('dayLow', 0))
        fifty_two_week_high = float(info.get('fiftyTwoWeekHigh', 0))
        fifty_two_week_low = float(info.get('fiftyTwoWeekLow', 0))
        dividend_yield = float(info.get('dividendYield', 0)) * 100  # Convert to percentage
        pe_ratio = info.get('trailingPE', 'N/A')
        revenue = info.get('totalRevenue', 'N/A')
        earnings = info.get('earnings', 'N/A')

        latest_performance = data_1y['Close'][-5:].tolist()

        # Prepare data for prediction
        prices = data_1y['Close'].values.reshape(-1, 1)
        scaler.fit(prices)
        scaled_prices = scaler.transform(prices)

        seq_length = 60
        X_input = scaled_prices[-seq_length:].reshape(1, seq_length, 1)

        # Predict price
        predicted_price_scaled = model.predict(X_input)
        predicted_price = float(scaler.inverse_transform(predicted_price_scaled)[0][0])

        suggestion = "Hold"
        if predicted_price > current_price:
            suggestion = "Buy"
        elif predicted_price < current_price:
            suggestion = "Sell"

        # Generate graphs
        def generate_graph(data, title, filename):
            plt.figure(figsize=(10, 6))
            plt.plot(data.index, data['Close'], label='Close Price')
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            graph_path = f"static/{filename}.png"
            plt.savefig(graph_path)
            plt.close()
            return graph_path

        graph_1y = generate_graph(data_1y, f"{ticker} 1-Year Historical Prices", f"{ticker}_1y_graph")
        graph_1m = generate_graph(data_1m, f"{ticker} 1-Month Historical Prices", f"{ticker}_1m_graph")
        graph_1w = generate_graph(data_1w, f"{ticker} 1-Week Historical Prices", f"{ticker}_1w_graph")

        return jsonify({
            'ticker': ticker,
            'prediction_date': prediction_date,
            'prediction_time': prediction_time,
            'predicted_price': predicted_price,
            'logo_url': logo_url,
            'company_name': company_name,
            'sector': sector,
            'market_cap': market_cap,
            'current_price': current_price,
            'day_high': day_high,
            'day_low': day_low,
            'fifty_two_week_high': fifty_two_week_high,
            'fifty_two_week_low': fifty_two_week_low,
            'dividend_yield': dividend_yield,
            'pe_ratio': pe_ratio,
            'revenue': revenue,
            'earnings': earnings,
            'latest_performance': latest_performance,
            'graph_1y': graph_1y,
            'graph_1m': graph_1m,
            'graph_1w': graph_1w,
            'suggestion': suggestion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
