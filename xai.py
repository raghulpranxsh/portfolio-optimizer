import os
import warnings
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import time
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import yfinance as yf
from arch import arch_model
import pickle
from pypfopt.efficient_frontier import EfficientFrontier
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import psycopg2
from psycopg2 import sql

# ======================  DATABASE INITIALISATION  ======================
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_history (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255),
            timestamp TIMESTAMP,
            total_investment FLOAT,
            weights JSONB,
            expected_return FLOAT,
            volatility FLOAT,
            sharpe_ratio FLOAT,
            forecast_returns JSONB,
            forecast_vols JSONB
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'raghul2005',
    'host': 'db.uoddmwersflnrkurprtj.supabase.co',
    'port': '5432'
}
init_db()

# ======================  ENVIRONMENT & FLAGS  ======================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO)

FLAGS = {
    'QUICK_MODE': True,
    'TRAIN_LSTM': True,
    'TRAIN_GARCH': True,
    'LSTM_EPOCHS': 10,
    'LSTM_BATCH': 32,
    'GARCH_P': 3,
    'GARCH_Q': 5,
    'SEQ_LENGTH': 50,
    'SEED': 42,
    'LOOKBACK_YEARS': 10,
    'BACKTEST_YEARS': 3
}

SEED = FLAGS['SEED']
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SEQ_LENGTH = FLAGS['SEQ_LENGTH']
MODEL_DIR = 'models_cache'
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================  AVAILABLE TICKERS  ======================
AVAILABLE_TICKERS = [
    'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'GS', 'HD',
    'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
    'NKE', 'PG', 'TRV', 'UNH', 'VZ', 'WBA', 'WMT'
]

# ======================  STREAMLIT CONFIG  ======================
st.set_page_config(page_title="Real-time Portfolio Optimizer", layout="wide")

# ======================  UTILITIES  ======================
def get_dynamic_date_range(lookback_years=10):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def calculate_rsi(prices, window=14):
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    if len(xs) == 0:
        return np.empty((0, seq_length, data.shape[1])), np.empty((0,))
    return np.array(xs), np.array(ys)

def download_realtime_data(tickers, start, end, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
            if df.empty:
                raise RuntimeError("yfinance returned empty data")
            return df['Adj Close']
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}")

# ======================  MODEL HELPERS  ======================
def build_simple_lstm(seq_length, n_features=2):
    model = Sequential([
        Input(shape=(seq_length, n_features)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
    return model

def train_or_load_lstm(ticker, X_train, y_train, X_val, y_val, scaler, force_retrain=False):
    path = os.path.join(MODEL_DIR, f'lstm_{ticker}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{ticker}.pkl')
    if os.path.exists(path) and os.path.exists(scaler_path) and not force_retrain:
        model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        if model_age.days < 1:
            return load_model(path, compile=False), pickle.load(open(scaler_path, 'rb'))

    model = build_simple_lstm(SEQ_LENGTH)
    if len(X_train) == 0:
        raise ValueError("Empty training data")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=FLAGS['LSTM_EPOCHS'], batch_size=FLAGS['LSTM_BATCH'], verbose=0)
    model.save(path)
    pickle.dump(scaler, open(scaler_path, 'wb'))
    K.clear_session()
    return model, scaler

def fit_or_load_garch(ticker, returns, p=3, q=5, force_retrain=False):
    path = os.path.join(MODEL_DIR, f'garch_{ticker}.pkl')
    if os.path.exists(path) and not force_retrain:
        model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        if model_age.days < 1:
            with open(path, 'rb') as f:
                return pickle.load(f)

    model = arch_model(returns, vol='Garch', p=p, q=q, dist='normal')
    res = model.fit(disp='off')
    with open(path, 'wb') as f:
        pickle.dump(res, f)
    return res


# ======================  TRAINING / CACHING  ======================
@st.cache_resource(ttl=3600)
def train_models(selected_tickers, force_retrain=False):
    if not selected_tickers:
        return [], {}, {}

    TICKERS = selected_tickers
    START, END = get_dynamic_date_range(FLAGS['LOOKBACK_YEARS'])
    adj_close = download_realtime_data(TICKERS, START, END)
    log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
    rsis = adj_close.apply(calculate_rsi).dropna()

    common_idx = log_returns.index.intersection(rsis.index)
    log_returns = log_returns.loc[common_idx]
    rsis = rsis.loc[common_idx]

    scalers = {}
    rmse_results = {'Decision Tree': [], 'LSTM': [], 'ARIMA': []}
    lstm_predictions, arima_predictions, dt_predictions = {}, {}, {}

    for ticker in tqdm(TICKERS, desc='Training models'):
        if ticker not in log_returns.columns:
            continue
        df = pd.DataFrame({'log_return': log_returns[ticker], 'rsi': rsis[ticker]}).dropna()
        if df.empty:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        scalers[ticker] = scaler
        pickle.dump(scaler, open(os.path.join(MODEL_DIR, f'scaler_{ticker}.pkl'), 'wb'))

        train_size = int(len(scaled) * 0.8)
        train_data, test_data = scaled[:train_size], scaled[train_size:]
        X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
        X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        y_inv = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 1))]))[:, 0]

        if FLAGS['TRAIN_LSTM']:
            lstm, _ = train_or_load_lstm(ticker, X_train, y_train, X_test, y_test, scaler, force_retrain)
            pred_scaled = lstm.predict(X_test, verbose=0)
            inv = scaler.inverse_transform(np.hstack([pred_scaled, np.zeros((len(pred_scaled), 1))]))[:, 0]
            lstm_rmse = np.sqrt(mean_squared_error(y_inv, inv))
            rmse_results['LSTM'].append(lstm_rmse)
            lstm_predictions[ticker] = inv.tolist()

        arima_model = ARIMA(df['log_return'][:train_size], order=(5, 1, 0))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=len(test_data))
        arima_rmse = np.sqrt(mean_squared_error(df['log_return'][train_size:], arima_pred))
        rmse_results['ARIMA'].append(arima_rmse)
        arima_predictions[ticker] = arima_pred.tolist()

        dt = DecisionTreeRegressor()
        dt.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        dt_pred_scaled = dt.predict(X_test.reshape(X_test.shape[0], -1))
        dt_inv = scaler.inverse_transform(np.hstack([dt_pred_scaled.reshape(-1,1), np.zeros((len(dt_pred_scaled), 1))]))[:, 0]
        dt_rmse = np.sqrt(mean_squared_error(y_inv, dt_inv))
        rmse_results['Decision Tree'].append(dt_rmse)
        dt_predictions[ticker] = dt_inv.tolist()

    forecast_vols = {}
    if FLAGS['TRAIN_GARCH']:
        for ticker in TICKERS:
            ret = log_returns[ticker].dropna()
            if ret.empty:
                forecast_vols[ticker] = float(ret.std())
                continue
            res = fit_or_load_garch(ticker, ret, FLAGS['GARCH_P'], FLAGS['GARCH_Q'], force_retrain)
            fc = res.forecast(horizon=1)
            var = fc.variance.iloc[-1].iat[0]
            forecast_vols[ticker] = np.sqrt(var)

    report = {
        'timestamp': datetime.now().isoformat(),
        'latest_data_date': str(log_returns.index[-1]),
        'avg_rmse': {k: float(np.nanmean(v)) for k, v in rmse_results.items()},
        'forecast_volatilities': forecast_vols,
        'arima_predictions': arima_predictions,
        'dt_predictions': dt_predictions,
        'lstm_predictions': lstm_predictions
    }
    with open(os.path.join(MODEL_DIR, 'realtime_report.pkl'), 'wb') as f:
        pickle.dump(report, f)

    return TICKERS, scalers, rmse_results

# ======================  PORTFOLIO ALLOCATION  ======================
@st.cache_resource(ttl=3600)
def allocate_portfolio(total_investment, selected_tickers,
                       shock_factor=1.0, vol_multiplier=1.0,
                       username=None, retrain_models=False):
    if not selected_tickers:
        return pd.DataFrame(), 0, 0, 0, None, {}, {}, {}

    TICKERS, scalers, rmse_results = train_models(selected_tickers, force_retrain=retrain_models)
    END = datetime.now()
    START = END - timedelta(days=365 * FLAGS['LOOKBACK_YEARS'])
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=False, progress=False)
    adj_close = data['Adj Close']
    latest_date = adj_close.index[-1]
    adj_close = adj_close.dropna(how='all').ffill()
    log_returns = np.log(adj_close / adj_close.shift(1)).dropna()

    forecast_returns = {}
    for ticker in TICKERS:
        model_path = os.path.join(MODEL_DIR, f'lstm_{ticker}.keras')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{ticker}.pkl')
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            forecast_returns[ticker] = float(log_returns[ticker].mean()) * shock_factor
            continue
        scaler = pickle.load(open(scaler_path, 'rb'))
        rsi = calculate_rsi(adj_close[ticker])
        df = pd.DataFrame({"log_return": log_returns[ticker], "rsi": rsi}).dropna()
        if len(df) < SEQ_LENGTH:
            forecast_returns[ticker] = float(log_returns[ticker].mean()) * shock_factor
            continue
        scaled = scaler.transform(df)
        model = load_model(model_path, compile=False)
        last_seq = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 2)
        pred_scaled = model.predict(last_seq, verbose=0)
        pred = scaler.inverse_transform(np.hstack([pred_scaled, np.zeros((1, 1))]))[0, 0]
        forecast_returns[ticker] = max(float(pred), 0.0015) * shock_factor

    forecast_vols = {}
    for ticker in TICKERS:
        path = os.path.join(MODEL_DIR, f"garch_{ticker}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                res = pickle.load(f)
            fc = res.forecast(horizon=1)
            var = fc.variance.iloc[-1].iat[0]
            forecast_vols[ticker] = np.sqrt(var) * vol_multiplier
        else:
            forecast_vols[ticker] = float(log_returns[ticker].std()) * vol_multiplier

    recent_returns = log_returns.iloc[-180:]
    corr = recent_returns.corr().fillna(0)
    vols = pd.Series(forecast_vols)
    cov = (np.diag(vols.reindex(TICKERS).fillna(vols.mean()).values) @
           corr.values @
           np.diag(vols.reindex(TICKERS).fillna(vols.mean()).values))
    cov = pd.DataFrame(cov, index=TICKERS, columns=TICKERS)
    mu = pd.Series(forecast_returns).reindex(TICKERS).fillna(1e-6)

    try:
        ef = EfficientFrontier(mu, cov)
        raw_weights = ef.max_sharpe()
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False)
        expected_return, volatility, sharpe = perf
    except Exception as e:
        total_ret = sum(forecast_returns.values())
        weights = {t: max(forecast_returns.get(t, 0) / total_ret, 0) for t in TICKERS}
        expected_return = sum(weights[t] * forecast_returns.get(t, 0) for t in TICKERS)
        volatility = np.sqrt(sum(weights[t1] * weights[t2] * cov.loc[t1, t2] for t1 in TICKERS for t2 in TICKERS))
        sharpe = expected_return / volatility if volatility > 0 else 0
        st.warning(f"Optimisation failed – using proportional weights: {e}")

    alloc_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])
    alloc_df["Investment"] = alloc_df["Weight"] * total_investment
    alloc_df["Forecast_Return"] = [forecast_returns.get(t, 0) for t in alloc_df.index]
    alloc_df["Forecast_Vol"] = [forecast_vols.get(t, 0) for t in alloc_df.index]
    alloc_df = alloc_df[alloc_df["Weight"] > 1e-6]

    if username:
        clean = lambda d: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in d.items()}
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            sql.SQL("""
                INSERT INTO portfolio_history
                (username, timestamp, total_investment, weights, expected_return, volatility, sharpe_ratio, forecast_returns, forecast_vols)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """),
            (username, datetime.now(), float(total_investment),
             pd.Series(clean(weights)).to_json(),
             float(expected_return), float(volatility), float(sharpe),
             pd.Series(clean(forecast_returns)).to_json(),
             pd.Series(clean(forecast_vols)).to_json())
        )
        conn.commit()
        cur.close()
        conn.close()

    return (alloc_df, float(expected_return), float(volatility),
            float(sharpe), latest_date, forecast_returns, forecast_vols, rmse_results)

# ======================  PIE CHART  ======================
def create_allocation_pie(alloc_df: pd.DataFrame) -> go.Figure:
    labels = []
    for ticker in alloc_df.index:
        name = yf.Ticker(ticker).info.get('longName', ticker)
        pct = alloc_df.loc[ticker, 'Weight'] * 100
        labels.append(f"{ticker} ({name}) – {pct:.2f}%")

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=alloc_df['Weight'],
        hole=0.4,
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        marker=dict(colors=px.colors.sequential.Tealgrn)
    )])
    fig.update_layout(
        title="Maximum Sharpe Ratio – Asset Allocation",
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# ======================  BACKTEST & GROWTH CHART  ======================
def backtest_portfolio(alloc_df: pd.DataFrame, total_investment: float,
                       start_date: str, end_date: str) -> pd.DataFrame:
    tickers = alloc_df.index.tolist()
    weights = alloc_df['Weight'].to_dict()
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    data = data.ffill().dropna(how='all')
    norm_prices = data / data.iloc[0]
    portfolio = sum(norm_prices[t] * weights.get(t, 0) for t in tickers)
    portfolio_value = portfolio * total_investment
    return pd.DataFrame({'Date': portfolio_value.index, 'Portfolio_Value': portfolio_value.values})

def create_portfolio_growth_chart(portfolio_value_df: pd.DataFrame, total_investment: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_value_df['Date'],
        y=portfolio_value_df['Portfolio_Value'],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(100, 149, 237, 0.3)',
        line=dict(color='royalblue', width=2),
        name='Portfolio Value'
    ))

    max_row = portfolio_value_df.loc[portfolio_value_df['Portfolio_Value'].idxmax()]
    fig.add_annotation(
        x=max_row['Date'], y=max_row['Portfolio_Value'],
        text=f"Maximum Sharpe Ratio: ${max_row['Portfolio_Value']:,.0f}",
        showarrow=True, arrowhead=2, ax=40, ay=-40,
        bgcolor="white", bordercolor="black", borderwidth=1
    )

    fig.update_layout(
        title="Portfolio Growth",
        xaxis_title="Year",
        yaxis_title="Portfolio Balance ($)",
        hovermode='x unified',
        template="simple_white",
        height=550,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    fig.update_xaxes(tickformat='%b %Y')
    fig.update_yaxes(tickprefix='$', tickformat=',.0f')
    return fig

# ======================  ANNUAL RETURNS BAR CHART (PORTFOLIO)  ======================
def calculate_annual_returns(alloc_df: pd.DataFrame, total_investment: float) -> pd.DataFrame:
    tickers = alloc_df.index.tolist()
    weights = alloc_df['Weight'].to_dict()
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    data = data.ffill()

    yearly_returns = []
    for year in [2023, 2024, 2025]:
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        mask = (data.index >= year_start) & (data.index <= year_end)
        if not mask.any():
            yearly_returns.append(0.0)
            continue
        year_data = data.loc[mask]
        if year_data.empty:
            yearly_returns.append(0.0)
            continue
        start_price = year_data.iloc[0]
        end_price = year_data.iloc[-1]
        portfolio_start = sum(start_price[t] * weights.get(t, 0) for t in tickers)
        portfolio_end = sum(end_price[t] * weights.get(t, 0) for t in tickers)
        ret = (portfolio_end - portfolio_start) / portfolio_start if portfolio_start > 0 else 0
        yearly_returns.append(ret * 100)

    return pd.DataFrame({
        'Year': [2023, 2024, 2025],
        'Annual_Return': yearly_returns
    })

def create_annual_returns_chart(annual_df: pd.DataFrame) -> go.Figure:
    max_year = annual_df.loc[annual_df['Annual_Return'].idxmax(), 'Year']
    max_ret = annual_df['Annual_Return'].max()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=annual_df['Year'],
        y=annual_df['Annual_Return'],
        marker_color='royalblue',
        text=[f"{r:.2f}%" for r in annual_df['Annual_Return']],
        textposition='outside'
    ))

    fig.add_annotation(
        x=max_year, y=max_ret,
        text=f"{max_year}<br>Maximum Sharpe Ratio: {max_ret:.2f}%",
        showarrow=True, arrowhead=2, ax=0, ay=-40,
        bgcolor="white", bordercolor="black", borderwidth=1
    )

    fig.update_layout(
        title="Annual Returns",
        xaxis_title="Year",
        yaxis_title="Annual Return",
        yaxis=dict(tickformat=".0%", range=[0, 45]),
        template="simple_white",
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# ======================  NEW: ANNUAL ASSET RETURNS BAR CHART  ======================
def calculate_asset_annual_returns(alloc_df: pd.DataFrame) -> pd.DataFrame:
    tickers = alloc_df.index.tolist()
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date,
                       auto_adjust=True, progress=False)['Close'].ffill()

    rows = []
    for year in [2023, 2024, 2025]:
        y_start = f"{year}-01-01"
        y_end   = f"{year}-12-31"
        mask = (data.index >= y_start) & (data.index <= y_end)
        if not mask.any():
            rows.append({t: 0.0 for t in tickers})
            continue

        year_data = data.loc[mask]
        if year_data.empty:
            rows.append({t: 0.0 for t in tickers})
            continue

        start_price = year_data.iloc[0]
        end_price   = year_data.iloc[-1]

        ret = {}
        for t in tickers:
            if t not in start_price or t not in end_price:
                ret[t] = 0.0
                continue
            p0 = start_price[t]
            p1 = end_price[t]
            ret[t] = (p1 - p0) / p0 * 100 if p0 > 0 else 0.0
        rows.append(ret)

    return pd.DataFrame(rows, index=[2023, 2024, 2025])

def create_asset_annual_returns_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    colors = {
        'AAPL': '#1f77b4',  # blue
        'JPM':  '#2ca02c',  # green
        'NKE':  '#7f7f7f'   # gray
    }

    for ticker in df.columns:
        fig.add_trace(go.Bar(
            name=ticker,
            x=df.index,
            y=df[ticker],
            marker_color=colors.get(ticker, px.colors.sequential.Tealgrn[0]),
            text=[f"{v:.1f}%" for v in df[ticker]],
            textposition='outside'
        ))

    fig.update_layout(
        title="Annual Asset Returns",
        xaxis_title="Year",
        yaxis_title="Return",
        yaxis=dict(tickformat=".0%", range=[-40, 60]),
        barmode='group',
        template="simple_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

# ======================  MODEL COMPARISON  ======================
@st.cache_data(ttl=3600)
def get_model_comparisons(selected_tickers):
    _, _, rmse_results = train_models(selected_tickers)
    report_path = os.path.join(MODEL_DIR, 'realtime_report.pkl')
    if os.path.exists(report_path):
        with open(report_path, 'rb') as f:
            report = pickle.load(f)
        return (rmse_results,
                report.get('arima_predictions', {}),
                report.get('dt_predictions', {}),
                report.get('lstm_predictions', {}))
    return rmse_results, {}, {}, {}

# ======================  HISTORY  ======================
def get_portfolio_history(username):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT timestamp, expected_return, volatility, sharpe_ratio FROM portfolio_history WHERE username = %s ORDER BY timestamp", (username,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=['timestamp', 'expected_return', 'volatility', 'sharpe_ratio'])

# ======================  LIME EXPLAINABILITY (FIXED)  ======================
def get_lime_explanation(ticker: str, X_flat: np.ndarray, model: tf.keras.Model, scaler: MinMaxScaler, feature_names: list):
    """
    X_flat: 2D array of shape (n_samples, n_features) — already flattened
    """
    def predict_fn(data_2d):
        # data_2d: (n_samples, 100)
        data_3d = data_2d.reshape(-1, SEQ_LENGTH, 2)
        pred = model.predict(data_3d, verbose=0)
        return pred  # shape: (n_samples, 1)

    explainer = LimeTabularExplainer(
        training_data=X_flat,
        feature_names=feature_names,
        mode='regression',
        discretize_continuous=True,
        sample_around_instance=True
    )

    # Explain the last instance
    instance = X_flat[-1]
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=min(20, len(feature_names))  # show top 20
    )
    return exp.as_html()

def prepare_lime_data(ticker: str):
    """
    Returns: X_flat (2D), model, scaler, feature_names
    """
    model_path = os.path.join(MODEL_DIR, f'lstm_{ticker}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{ticker}.pkl')
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None, None, None

    model = load_model(model_path, compile=False)
    scaler = pickle.load(open(scaler_path, 'rb'))

    START, END = get_dynamic_date_range(FLAGS['LOOKBACK_YEARS'])
    adj_close = download_realtime_data([ticker], START, END)
    log_returns = np.log(adj_close[ticker] / adj_close[ticker].shift(1)).dropna()
    rsi = calculate_rsi(adj_close[ticker]).loc[log_returns.index]

    df = pd.DataFrame({'log_return': log_returns, 'rsi': rsi}).dropna()
    if len(df) < SEQ_LENGTH:
        return None, None, None, None

    scaled = scaler.transform(df)
    X_3d, _ = create_sequences(scaled, SEQ_LENGTH)
    if len(X_3d) == 0:
        return None, None, None, None

    # FLATTEN: (N, 50, 2) → (N, 100)
    X_flat = X_3d.reshape(X_3d.shape[0], -1)

    # Create feature names: log_return_t-49, ..., log_return_t-0, rsi_t-49, ..., rsi_t-0
    log_names = [f"log_return_t-{SEQ_LENGTH-1-i}" for i in range(SEQ_LENGTH)]
    rsi_names = [f"rsi_t-{SEQ_LENGTH-1-i}" for i in range(SEQ_LENGTH)]
    feature_names = log_names + rsi_names

    return X_flat, model, scaler, feature_names
# ======================  LIME – PORTFOLIO-WIDE  ======================
def lime_for_allocation(alloc_df: pd.DataFrame):
    """
    Returns a dict: {ticker: (X_flat, model, scaler, feature_names, pred_log_ret)}
    Only for tickers with weight > 0.
    """
    result = {}
    for ticker in alloc_df.index:
        if alloc_df.loc[ticker, "Weight"] <= 1e-6:
            continue

        model_path = os.path.join(MODEL_DIR, f'lstm_{ticker}.keras')
        scaler_path = os.path.join(MODEL_DIR, f'scaler_{ticker}.pkl')
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            continue

        model = load_model(model_path, compile=False)
        scaler = pickle.load(open(scaler_path, 'rb'))

        START, END = get_dynamic_date_range(FLAGS['LOOKBACK_YEARS'])
        adj_close = download_realtime_data([ticker], START, END)
        log_returns = np.log(adj_close[ticker] / adj_close[ticker].shift(1)).dropna()
        rsi = calculate_rsi(adj_close[ticker]).loc[log_returns.index]

        df = pd.DataFrame({'log_return': log_returns, 'rsi': rsi}).dropna()
        if len(df) < SEQ_LENGTH:
            continue

        scaled = scaler.transform(df)
        X_3d, _ = create_sequences(scaled, SEQ_LENGTH)
        if X_3d.shape[0] == 0:
            continue

        X_flat = X_3d.reshape(X_3d.shape[0], -1)                # (N, 100)

        # feature names
        log_names = [f"log_return_t-{SEQ_LENGTH-1-i}" for i in range(SEQ_LENGTH)]
        rsi_names = [f"rsi_t-{SEQ_LENGTH-1-i}" for i in range(SEQ_LENGTH)]
        feature_names = log_names + rsi_names

        # next-day prediction (for display)
        last_seq = X_3d[-1:].reshape(1, SEQ_LENGTH, 2)
        pred_scaled = model.predict(last_seq, verbose=0)
        pred = scaler.inverse_transform(
            np.hstack([pred_scaled, np.zeros((1, 1))])
        )[0, 0]

        result[ticker] = (X_flat, model, scaler, feature_names, pred)

    return result
# ======================  AUTHENTICATION  ======================
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    login_response = authenticator.login(location='main')
    if login_response is not None and len(login_response) == 3:
        name, authentication_status, username = login_response
    else:
        name = st.session_state.get('name')
        authentication_status = st.session_state.get('authentication_status')
        username = st.session_state.get('username')

    if authentication_status:
        st.sidebar.title(f"Welcome {name}")
        authenticator.logout('Logout', 'sidebar')

        # === ALL TABS CREATED HERE ===
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Portfolio Allocation",
            "History Tracking", 
            "Model Comparison", 
            "Scenario Testing",
            "Explainable AI (LIME)"
        ])
        # ====================== TAB 1: Portfolio Allocation ======================
        with tab1:
            st.header("Real-time Portfolio Allocation")
            selected_tickers = st.multiselect("Select Tickers", AVAILABLE_TICKERS,
                default=['AAPL', 'NKE', 'JPM'] if FLAGS['QUICK_MODE'] else AVAILABLE_TICKERS)
            total_investment = st.number_input("Total Investment ($)", min_value=1000.0, value=100000.0)
            retrain_models = st.checkbox("Retrain models if older than 1 day", value=False)

            if st.button("Optimize Portfolio"):
                if not selected_tickers:
                    st.error("Please select at least one ticker.")
                else:
                    with st.spinner("Optimizing..."):
                        (alloc_df, portfolio_return, volatility, sharpe, latest_date, _, _, _) = allocate_portfolio(
                            total_investment, selected_tickers, username=username, retrain_models=retrain_models)
                    st.session_state["last_allocation_df"] = alloc_df   # <-- NEW
                    st.session_state["last_investment"] = total_investment
                    st.session_state["last_sharpe"] = sharpe
                    st.session_state["last_return"] = portfolio_return
                    st.session_state["last_vol"] = volatility
                    st.success("Optimization Complete!")

                    st.subheader("Allocation")
                    st.dataframe(alloc_df)

                    st.subheader("Maximum Sharpe Ratio – Asset Allocation")
                    st.plotly_chart(create_allocation_pie(alloc_df), use_container_width=True)

                    st.subheader("Portfolio Growth")
                    START_BACKTEST = (datetime.now() - timedelta(days=365*FLAGS['BACKTEST_YEARS'])).strftime('%Y-%m-%d')
                    END_BACKTEST = datetime.now().strftime('%Y-%m-%d')
                    with st.spinner("Backtesting..."):
                        growth_df = backtest_portfolio(alloc_df, total_investment, START_BACKTEST, END_BACKTEST)
                    st.plotly_chart(create_portfolio_growth_chart(growth_df, total_investment), use_container_width=True)

                    st.subheader("Annual Returns")
                    with st.spinner("Calculating annual returns..."):
                        annual_df = calculate_annual_returns(alloc_df, total_investment)
                    st.plotly_chart(create_annual_returns_chart(annual_df), use_container_width=True)

                    st.subheader("Annual Asset Returns")
                    with st.spinner("Calculating asset-level annual returns..."):
                        asset_annual_df = calculate_asset_annual_returns(alloc_df)
                    st.plotly_chart(create_asset_annual_returns_chart(asset_annual_df), use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Expected Return", f"{portfolio_return*100:.2f}%")
                    with col2: st.metric("Volatility", f"{volatility*100:.2f}%")
                    with col3: st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    st.write(f"**Latest data:** {latest_date}")

        # ====================== TAB 2: History Tracking ======================
        with tab2:
            st.header("Portfolio History Tracking")
            history_df = get_portfolio_history(username)
            if not history_df.empty:
                st.dataframe(history_df.tail(10))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['expected_return']*100, mode='lines+markers', name='Return'))
                fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['volatility']*100, mode='lines+markers', name='Volatility'))
                fig.add_trace(go.Scatter(x=history_df['timestamp'], y=history_df['sharpe_ratio'], mode='lines+markers', name='Sharpe'))
                fig.update_layout(title='Portfolio Metrics Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No history yet.")

        # ====================== TAB 3: Model Comparison ======================
        with tab3:
            st.header("Model Comparison")
            comp_tickers = st.multiselect("Select Tickers", AVAILABLE_TICKERS, key='comp', default=['AAPL'])
            rmse_results, arima_preds, dt_preds, lstm_preds = get_model_comparisons(comp_tickers)
            st.dataframe(pd.DataFrame([rmse_results]).T)
            sel = st.selectbox("Select Ticker", comp_tickers)
            if sel in lstm_preds:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=lstm_preds[sel], name='LSTM', line=dict(color='green')))
                fig.add_trace(go.Scatter(y=arima_preds.get(sel, []), name='ARIMA', line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=dt_preds.get(sel, []), name='DT', line=dict(color='orange')))
                st.plotly_chart(fig, use_container_width=True)

        # ====================== TAB 4: Scenario Testing ======================
        with tab4:
            st.header("Scenario Testing")
            scen_tickers = st.multiselect("Select Tickers", AVAILABLE_TICKERS, key='scen', default=['AAPL', 'JPM'])
            col1, col2 = st.columns(2)
            with col1: shock_factor = st.slider("Shock Factor", 0.5, 1.5, 1.0, 0.1)
            with col2: vol_multiplier = st.slider("Volatility Multiplier", 0.5, 2.0, 1.0, 0.1)
            total_investment_scen = st.number_input("Investment", min_value=1000.0, value=100000.0, key='inv_scen')
            retrain_scen = st.checkbox("Retrain models", value=False)

            if st.button("Simulate"):
                with st.spinner("Simulating..."):
                    (alloc_df, ret, vol, shr, _, _, _, _) = allocate_portfolio(
                        total_investment_scen, scen_tickers, shock_factor, vol_multiplier,
                        username=username, retrain_models=retrain_scen)
                st.dataframe(alloc_df)
                st.plotly_chart(create_allocation_pie(alloc_df), use_container_width=True)

                START_BACKTEST = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
                with st.spinner("Backtesting..."):
                    growth_df = backtest_portfolio(alloc_df, total_investment_scen, START_BACKTEST, datetime.now().strftime('%Y-%m-%d'))
                st.plotly_chart(create_portfolio_growth_chart(growth_df, total_investment_scen), use_container_width=True)

                st.subheader("Annual Returns")
                with st.spinner("Calculating..."):
                    annual_df = calculate_annual_returns(alloc_df, total_investment_scen)
                st.plotly_chart(create_annual_returns_chart(annual_df), use_container_width=True)

                st.subheader("Annual Asset Returns")
                with st.spinner("Calculating..."):
                    asset_annual_df = calculate_asset_annual_returns(alloc_df)
                st.plotly_chart(create_asset_annual_returns_chart(asset_annual_df), use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Return", f"{ret*100:.2f}%")
                with col2: st.metric("Volatility", f"{vol*100:.2f}%")
                with col3: st.metric("Sharpe", f"{shr:.2f}")

        # ====================== TAB 5: Explainable AI – LIME (BEAUTIFUL UI) ======================
        with tab5:
            st.header("Explainable AI – Why Did the Model Pick These Weights?")

            # Custom CSS for LIME
            st.markdown("""
            <style>
            .lime-container {
                background: #ffffff;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
            .lime-title {
                font-size: 1.4em;
                font-weight: bold;
                color: #1e3799;
                margin-bottom: 10px;
            }
            .lime-pred {
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
                color: white;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.2em;
                font-weight: bold;
            }
            .lime-bar {
                font-size: 14px !important;
            }
            .lime-bar text {
                fill: white !important;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)

            alloc_df = st.session_state.get("last_allocation_df")
            if alloc_df is None or alloc_df.empty:
                st.warning("No allocation yet. Optimize in **Portfolio Allocation** first!")
                st.info("Tip: Run optimization → come back here → see **why** each stock was chosen!")
            else:
                st.success(f"Explaining **{len(alloc_df)} assets** in your **${st.session_state.get('last_investment', 100000):,}** portfolio")

                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Portfolio Sharpe", f"{st.session_state.get('last_sharpe', 0):.2f}")
                with col2:
                    st.metric("Expected Return", f"{st.session_state.get('last_return', 0)*100:.2f}%")
                with col3:
                    st.metric("Volatility", f"{st.session_state.get('last_vol', 0)*100:.2f}%")

                st.markdown("---")

                with st.spinner("Generating **beautiful** LIME explanations..."):
                    lime_data = lime_for_allocation(alloc_df)

                if not lime_data:
                    st.error("No trained models found. Try optimizing first!")
                else:
                    for ticker, (X_flat, model, scaler, feat_names, pred) in lime_data.items():
                        weight_pct = alloc_df.loc[ticker, 'Weight'] * 100
                        investment = alloc_df.loc[ticker, 'Investment']

                        # Get company name
                        try:
                            company_name = yf.Ticker(ticker).info.get('longName', ticker)
                        except:
                            company_name = ticker

                        with st.container():
                            st.markdown(f"""
                            <div class="lime-container">
                                <div class="lime-title">
                                    {ticker} – {company_name}
                                </div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666;">
                                    <span>Weight: <b>{weight_pct:.2f}%</b></span>
                                    <span>Investment: <b>${investment:,.0f}</b></span>
                                    <span>Forecast Return: <b>{pred*100:+.3f}%</b></span>
                                </div>
                                <div class="lime-pred">
                                    Model Predicts: Next-Day Log Return = {pred:.5f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # LIME Explanation
                            html = get_lime_explanation(ticker, X_flat, model, scaler, feat_names)
                            # Inject CSS class
                            html = html.replace('<div class="explanation"', '<div class="explanation lime-bar"')
                            st.components.v1.html(html, height=500, scrolling=True)

                            st.markdown("---")

    elif authentication_status is False:
        st.error('Username/password is incorrect')
    elif authentication_status is None:
        st.warning('Please enter your username and password')

except FileNotFoundError:
    st.error("config.yaml not found! Please create it with credentials.")
except Exception as e:
    st.error(f"Authentication error: {e}")