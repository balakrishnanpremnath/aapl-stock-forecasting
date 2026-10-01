# AAPL Stock Price Forecasting

Comparing a naive baseline, Prophet, and an LSTM neural network
for forecasting Apple (AAPL) closing prices.

## Project Overview

This project explores historical stock prices, builds three forecasting
models, and compares their prediction errors using MAE and RMSE.

The main learning objective is to understand how forecasting assumptions,
data preparation, and evaluation methods affect model comparisons.

## Dataset

- **Source:** Yahoo Finance through `yfinance`
- **Ticker:** AAPL
- **Requested date range:** January 1, 2019 to January 1, 2024
  (end date excluded)
- **Target:** Closing price
- **Preparation:** Reindex to weekdays and interpolate missing values
- **Test split:** Last 90 rows of the prepared series

The prepared series includes interpolated weekday values, so its rows
do not correspond exclusively to actual exchange trading days.

## Tools and Libraries

- Python
- Pandas and NumPy
- Matplotlib
- yfinance
- Statsmodels
- Prophet
- Scikit-learn
- TensorFlow / Keras

## Project Workflow

1. Download historical AAPL prices.
2. Prepare and inspect the time series.
3. Explore trend and seasonality.
4. Apply the Augmented Dickey-Fuller stationarity test.
5. Split the series chronologically.
6. Build naive, Prophet, and LSTM models.
7. Calculate MAE and RMSE.
8. Visualize predictions and review evaluation limitations.

## Models

| Model | Current implementation |
|---|---|
| Naive baseline | Repeats the final training price across the entire test period. |
| Prophet | Produces forecasts for the test period from the training data. |
| LSTM | Uses 30-observation input windows to predict the next value. Later test windows include earlier observed test values. |

## Preliminary Results

The following values were reported in the original experiment:

| Model | MAE | RMSE |
|---|---:|---:|
| Naive baseline | 8.07 | 9.80 |
| Prophet | 10.03 | 11.46 |
| LSTM | 10.32 | 11.37 |

Lower values indicate smaller prediction errors.

The naive baseline has the lowest reported errors in this experiment.
However, the evaluation issues below must be corrected before treating
these numbers as a fair comparison of model performance.

## Evaluation Limitations

- The LSTM scaler is fitted on the full series before splitting.
  It should be fitted using training data only.
- Naive and Prophet forecasts use a fixed forecast origin, while the
  LSTM uses rolling windows containing newly observed test-period prices.
  A consistent evaluation protocol is needed.
- Interpolation introduces values for weekdays without trading data.
  Its effect on the experiment should be assessed.
- Results cover one test window and may not generalize to other periods.
- Downloaded prices and model results may vary with data revisions,
  package versions, and random initialization.

These results are preliminary and do not demonstrate investment
profitability.

## Main Notebook

`time_series_forecasting.ipynb` contains the data preparation,
exploratory analysis, model training, evaluation, and visualizations.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/balakrishnanpremnath/aapl-stock-forecasting.git
cd aapl-stock-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy matplotlib yfinance statsmodels prophet scikit-learn tensorflow jupyter
```

### 4. Check the Prophet import

The notebook must include this import before creating the Prophet model:

```python
from prophet import Prophet
```

### 5. Open the notebook

```bash
jupyter notebook time_series_forecasting.ipynb
```

Run the cells from top to bottom. An internet connection is required
to download the price data.

## Planned Improvements

- Fit preprocessing using training data only.
- Use a consistent forecasting protocol across all models.
- Evaluate across multiple chronological test windows.
- Review interpolation and retain actual trading dates where appropriate.
- Record package versions and random seeds.
- Export reproducible metrics and comparison charts.
- Test additional features and measure whether they improve performance.

## Author

**Balakrishnan Premnath**  
BSc (Hons) in Data Science — Sri Lanka Technology Campus

[GitHub](https://github.com/balakrishnanpremnath) |
[LinkedIn](https://www.linkedin.com/in/balakrishnan-premnath)

## Disclaimer

This project is for educational purposes and is not financial advice.
