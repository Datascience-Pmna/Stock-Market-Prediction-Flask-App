from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
from keras._tf_keras.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

app.secret_key = 'Stock-market-prediction-24'


# Load your pre-trained model
model = load_model("stock_future_prediction_saved.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock_id", "").strip().upper()
        if not stock:
            error_message = "Please enter a valid stock ID."
            return render_template("index.html", error_message=error_message)
        session["stock_id"] = stock  # Store the stock ID in the session
        return redirect(url_for("results"))  # Redirect to the results page
    return render_template("index.html")


@app.route("/results")
def results():
    stock = session.get("stock_id")
    if not stock:
        return redirect(url_for("index"))

    try:
        # Fetch stock data
        from datetime import datetime
        end = datetime.now()
        start = datetime(end.year - 15, end.month, end.day)
        df = yf.download(stock, start, end)

        if df.empty:
            return render_template("results.html", error_message=f"No data found for stock ID '{stock}'.")

        Close_price = df['Close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(Close_price.values.reshape(-1, 1))

        # Prepare data for model prediction
        x_data = []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
        x_data = np.array(x_data)

        predicted_scaled = model.predict(x_data)
        predicted_prices = scaler.inverse_transform(predicted_scaled)

        # Historical vs. Predicted Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index[-len(predicted_prices):], Close_price[-len(predicted_prices):], label="Actual Prices", color="blue")
        ax.plot(df.index[-len(predicted_prices):], predicted_prices, label="Model Predictions", color="orange")
        ax.set_title(f"{stock} Actual vs Predicted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        actual_vs_predicted_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Next 10 Days Prediction
        last_100_days = scaled_data[-100:].reshape(1, 100, 1)
        prediction_10_days = []

        for _ in range(10):
            next_day_pred = model.predict(last_100_days)
            prediction_10_days.append(next_day_pred)
            last_100_days = np.append(
                last_100_days[:, 1:, :], next_day_pred.reshape(1, 1, 1), axis=1
            )

        prediction_10_days = np.array(prediction_10_days).reshape(-1, 1)
        prediction_10_days = scaler.inverse_transform(prediction_10_days)

        last_date = df.index[-1]
        next_10_days = pd.date_range(last_date + pd.DateOffset(days=1), periods=10)
        predictions = [
            {"Date": date.date(), "Predicted": float(pred)}
            for date, pred in zip(next_10_days, prediction_10_days)
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(next_10_days, prediction_10_days, label="Predicted Prices for Next 10 Days", color="purple")
        ax.set_title(f"{stock} Predicted Prices for Next 10 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        plt.xticks(rotation=30)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        next_10_days_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Closing Price Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, Close_price, label="Closing Prices", color="blue")
        ax.set_title(f"{stock} Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        closing_prices_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Moving Averages Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, Close_price.rolling(window=100).mean(), label="100-Day MA", color="orange")
        ax.plot(df.index, Close_price.rolling(window=200).mean(), label="200-Day MA", color="green")
        ax.set_title(f"{stock} Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        moving_averages_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        return render_template(
            "results.html",
            predictions=predictions,
            plots={
                "actual_vs_predicted": actual_vs_predicted_plot,
                "closing_prices": closing_prices_plot,
                "moving_averages": moving_averages_plot,
                "next_10_days": next_10_days_plot 
            },
        )

    except Exception as e:
        return render_template("results.html", error_message=f"An error occurred: {e}")
    
@app.route("/back_to_index")
def back_to_index():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)