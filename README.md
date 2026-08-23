# AAPL Stock Price Forecasting: Naive vs Prophet vs LSTM

A time series forecasting project comparing three approaches of increasing complexity to predict Apple (AAPL) stock closing prices — a naive baseline, Facebook's Prophet, and an LSTM neural network.

## 📌 Project Overview

**Goal:** Predict AAPL daily closing prices and evaluate whether more complex models actually outperform a simple baseline.

**Dataset:** 5 years of daily AAPL closing prices (2019–2024), pulled live via the `yfinance` API.

**Approach:**
1. Explore and decompose the series into trend, seasonality, and noise
2. Test for stationarity using the Augmented Dickey-Fuller (ADF) test
3. Build a naive baseline forecast
4. Fit a classical model (Prophet)
5. Fit a deep learning model (LSTM)
6. Compare all three on the same held-out 90-day test period

## 🛠️ Tech Stack

- **Data:** `yfinance`, `pandas`, `numpy`
- **Classical time series:** `statsmodels`, `Prophet`
- **Deep learning:** `TensorFlow` / `Keras` (LSTM)
- **Evaluation & visualization:** `scikit-learn`, `matplotlib`

## 📊 Results

Models were evaluated on the last 90 days of data (unseen during training), using MAE and RMSE (lower = better).

| Model | MAE | RMSE |
|---|---|---|
| **Naive Baseline** | **8.07** | **9.80** |
| Prophet | 10.03 | 11.46 |
| LSTM | 10.32 | 11.37 |

The naive baseline — simply predicting "tomorrow's price = today's price" — outperformed both Prophet and the LSTM on this test window.

## 💡 Conclusion

The naive baseline (MAE: 8.07, RMSE: 9.80) outperformed both Prophet (MAE: 10.03) and the LSTM (MAE: 10.32) on this 90-day test window. This is a well-documented phenomenon in stock forecasting: daily closing prices closely follow a random walk, so "tomorrow ≈ today" is a surprisingly strong benchmark. Prophet and the LSTM both had more room to overfit to noise in the training period, since they're trying to learn patterns in a series that has very little exploitable structure. This suggests that improving forecast accuracy further would likely require additional features (trading volume, technical indicators, macroeconomic signals) rather than a more complex model architecture on price data alone.

## 🚀 Next Steps

- Add exogenous features (trading volume, moving averages, RSI, macroeconomic indicators)
- Try SARIMA or `auto_arima` for a fuller classical model comparison
- Backtest across multiple rolling time windows instead of a single train/test split
- Deploy as an interactive Streamlit app where users can pick a ticker and forecast horizon

## 📁 Repository Structure

```
├── time_series_forecasting.ipynb   # Full notebook: data, EDA, models, evaluation
└── README.md                       # Project overview (this file)
```

## ▶️ How to Run

1. Open `time_series_forecasting.ipynb` in [Google Colab](https://colab.research.google.com) or Jupyter locally
2. Run the first cell to install dependencies:
   ```bash
   pip install pandas numpy matplotlib yfinance statsmodels pmdarima prophet scikit-learn tensorflow
   ```
3. Run all cells in order, top to bottom

## ⚠️ Disclaimer

This project is for educational purposes only and is not financial advice. Stock price forecasting is inherently difficult, and no model here should be used to make real trading or investment decisions.

## Project Workflow

1. Load historical stock price data
2. Explore and visualize the time series
3. Create a naive baseline forecast
4. Train a Prophet forecasting model
5. Train an LSTM deep learning model
6. Evaluate models using MAE and RMSE
7. Compare the forecasting results
8. Identify the best-performing approach

## Model Performance

| Model | MAE | RMSE |
|---|---:|---:|
| Naive Baseline | 8.07 | 9.80 |
| Prophet | 10.03 | 11.46 |
| LSTM | 10.32 | 11.37 |

The Naive Baseline achieved the lowest MAE and RMSE on the 90-day test window, outperforming both Prophet and LSTM.

## Conclusion

The Naive Baseline outperformed both Prophet and LSTM on the 90-day test window. It achieved the lowest MAE of 8.07 and RMSE of 9.80.

This result shows that daily stock closing prices can be difficult to forecast because they often behave similarly to a random walk. More complex models such as Prophet and LSTM may learn noise from the historical data rather than useful future patterns.

Further improvements could include additional features such as trading volume, technical indicators, and macroeconomic variables instead of relying only on historical closing prices.

## Project Structure

```text
time-series-forecasting/
│
├── README.md
├── requirements.txt
└── time_series_forecasting.ipynb