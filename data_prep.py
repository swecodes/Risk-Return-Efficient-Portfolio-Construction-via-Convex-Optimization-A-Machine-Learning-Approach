
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TICKERS = ['NVDA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSM', 'AVGO', 'META', 'TSLA', 'JPM']
BENCHMARK = '^GSPC'

MACRO_TICKERS  = ['^VIX', '^TNX', '^IRX', 'DX-Y.NYB']
SECTOR_TICKERS = ['XLK', 'XLF', 'XLY']

ALL_TICKERS = TICKERS + [BENCHMARK] + MACRO_TICKERS + SECTOR_TICKERS

SECTOR_MAP = {
    'NVDA': 'XLK', 'GOOGL': 'XLK', 'AAPL': 'XLK', 'MSFT': 'XLK',
    'AMZN': 'XLY', 'META': 'XLK', 'AVGO': 'XLK', 'TSM': 'XLK',
    'TSLA': 'XLY', 'JPM': 'XLF'
}

START = '2015-01-01'
END   = '2025-12-31'

# Lookback window used by all sequence-based models
WINDOW = 20


def download_prices(tickers=ALL_TICKERS, start=START, end=END):
    """
    Download adjusted close prices for all tickers and forward-fill gaps.

    Returns
    -------
    pd.DataFrame  (dates x tickers)
    """
    print("Downloading price data …")
    raw = yf.download(tickers, start=start, end=end)['Close']
    clean = raw.ffill().dropna()
    print(f"Shape: {clean.shape}  |  {clean.index[0].date()} → {clean.index[-1].date()}")
    return clean


def compute_returns(price_df):
   
    return np.log(price_df / price_df.shift(1)).dropna()


def engineer_features(returns_df, tickers, vix_level, tnx_level, irx_level,
                       dxy_returns, sector_rets, sector_map):
    
    feats = returns_df.copy()

    for ticker in tickers:
        feats[f'{ticker}_Lag1']  = returns_df[ticker].shift(1)
        feats[f'{ticker}_Lag5']  = returns_df[ticker].shift(5)
        feats[f'{ticker}_MA20']  = returns_df[ticker].rolling(20).mean()
        feats[f'{ticker}_Vol20'] = returns_df[ticker].rolling(20).std()
        sec = sector_map[ticker]
        feats[f'{ticker}_RelStr'] = returns_df[ticker] - sector_rets[sec]

    aligned = lambda s: s.reindex(returns_df.index).ffill()
    feats['VIX_Level']    = aligned(vix_level)
    feats['VIX_Change']   = aligned(vix_level).diff()
    feats['Yield_Spread'] = aligned(tnx_level) - aligned(irx_level)
    feats['DXY_Return']   = dxy_returns.reindex(returns_df.index).fillna(0)

    return feats.dropna()


def get_sentiment_scores(tickers, date_index):

    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = {}

    for ticker in tickers:
        print(f"  Fetching sentiment for {ticker} …", end=" ")
        try:
            tk = yf.Ticker(ticker)
            news = tk.news
            records = []
            for item in news:
                try:
                    title = item.get('content', {}).get('title', '')
                    pub   = item.get('content', {}).get('pubDate', None)
                    if title and pub:
                        score = analyzer.polarity_scores(title)['compound']
                        date  = pd.Timestamp(pub).tz_localize(None).normalize()
                        records.append({'date': date, 'score': score})
                except Exception:
                    continue

            if records:
                df = pd.DataFrame(records).groupby('date')['score'].mean()
                sentiment_dict[ticker] = df
                print(f"({len(records)} headlines)")
            else:
                print("(no headlines — using zeros)")
        except Exception as e:
            print(f"(error: {e} — using zeros)")

    sent_df = pd.DataFrame(sentiment_dict).reindex(date_index).fillna(0)
    return sent_df.rolling(5, min_periods=1).mean()


TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END   = '2023-12-31'
TEST_START = '2024-01-01'


def split_and_scale(X, y):

    X_train = X.loc[:TRAIN_END]
    y_train = y.loc[:TRAIN_END]
    X_val   = X.loc[VAL_START:VAL_END]
    y_val   = y.loc[VAL_START:VAL_END]
    X_test  = X.loc[TEST_START:]
    y_test  = y.loc[TEST_START:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'X_train_sc': X_train_sc, 'X_val_sc': X_val_sc, 'X_test_sc': X_test_sc,
        'scaler': scaler,
    }

def make_sequences(X, y_arr, window):

    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y_arr[i])
    return np.array(Xs), np.array(ys)


def make_rank_targets(y_df):
    """
    Convert daily return targets into cross-sectional percentile ranks
    centered at zero.

    Returns
    -------
    pd.DataFrame  same shape as y_df, values in [-0.5, +0.5]
    """
    return y_df.rank(axis=1, pct=True) - 0.5


def build_all_datasets():
    
   
    prices     = download_prices()
    all_rets   = compute_returns(prices)
    stock_rets = all_rets[TICKERS].copy()
    bench_rets = all_rets[BENCHMARK]

    vix_level   = prices['^VIX']
    tnx_level   = prices['^TNX']
    irx_level   = prices['^IRX']
    dxy_rets    = all_rets['DX-Y.NYB']
    sector_rets = {t: all_rets[t] for t in SECTOR_TICKERS}


    ext_data = engineer_features(
        stock_rets, TICKERS,
        vix_level, tnx_level, irx_level,
        dxy_rets, sector_rets, SECTOR_MAP
    )
    print(f"Base features: {ext_data.shape[1]} features × {ext_data.shape[0]} days")

    # ── Sentiment-enriched features ───────────────────────────────────────────
    print("Fetching sentiment scores (Yahoo Finance + VADER) …")
    sent_df = get_sentiment_scores(TICKERS, ext_data.index)
    ext_data_sent = ext_data.copy()
    for t in TICKERS:
        ext_data_sent[f'{t}_Sent'] = sent_df[t].reindex(ext_data.index).fillna(0)
    print(f"Enriched features: {ext_data_sent.shape[1]} cols")

    
    def align_xy(X_df, drop_na=True):
        y_raw = stock_rets.shift(-1)
        X, y  = X_df.align(y_raw, join='inner', axis=0)
        mask  = X.notna().all(axis=1) & y.notna().all(axis=1)
        return X.loc[mask], y.loc[mask]

    X_base, y_base = align_xy(ext_data)
    X_sent, y_sent = align_xy(ext_data_sent)

    
    base = split_and_scale(X_base, y_base)
    sent = split_and_scale(X_sent, y_sent)

    n_features      = base['X_train_sc'].shape[1]
    n_features_sent = sent['X_train_sc'].shape[1]
    n_outputs       = len(TICKERS)

    
    X_tr_seq, y_tr_seq = make_sequences(base['X_train_sc'], base['y_train'].values, WINDOW)
    X_vl_seq, y_vl_seq = make_sequences(base['X_val_sc'],   base['y_val'].values,   WINDOW)
    X_ts_seq, y_ts_seq = make_sequences(base['X_test_sc'],  base['y_test'].values,  WINDOW)

    val_idx_seq  = base['y_val'].index[WINDOW:]
    test_idx_seq = base['y_test'].index[WINDOW:]

    
    y_ranked       = make_rank_targets(y_sent)
    train_y_ranked = y_ranked.loc[:TRAIN_END]
    val_y_ranked   = y_ranked.loc[VAL_START:VAL_END]
    test_y_ranked  = y_ranked.loc[TEST_START:]

    X_tr_sacre, y_tr_sacre = make_sequences(sent['X_train_sc'], train_y_ranked.values, WINDOW)
    X_vl_sacre, y_vl_sacre = make_sequences(sent['X_val_sc'],   val_y_ranked.values,   WINDOW)
    X_ts_sacre, y_ts_sacre = make_sequences(sent['X_test_sc'],  test_y_ranked.values,  WINDOW)

    _, y_tr_raw_sacre = make_sequences(sent['X_train_sc'], sent['y_train'].values, WINDOW)
    _, y_vl_raw_sacre = make_sequences(sent['X_val_sc'],   sent['y_val'].values,   WINDOW)
    _, y_ts_raw_sacre = make_sequences(sent['X_test_sc'],  sent['y_test'].values,  WINDOW)

    sacre_idx_seq = sent['y_test'].index[WINDOW:]

    return {
        
        'stock_rets': stock_rets,
        'bench_rets': bench_rets,
        
        'base': base,
        'sent': sent,
        
        'n_features':      n_features,
        'n_features_sent': n_features_sent,
        'n_outputs':       n_outputs,
        
        'X_tr_seq': X_tr_seq, 'y_tr_seq': y_tr_seq,
        'X_vl_seq': X_vl_seq, 'y_vl_seq': y_vl_seq,
        'X_ts_seq': X_ts_seq, 'y_ts_seq': y_ts_seq,
        'val_idx_seq':  val_idx_seq,
        'test_idx_seq': test_idx_seq,
        
        'X_tr_sacre': X_tr_sacre, 'y_tr_sacre': y_tr_sacre,
        'X_vl_sacre': X_vl_sacre, 'y_vl_sacre': y_vl_sacre,
        'X_ts_sacre': X_ts_sacre, 'y_ts_sacre': y_ts_sacre,
        'y_tr_raw_sacre': y_tr_raw_sacre,
        'y_ts_raw_sacre': y_ts_raw_sacre,
        'sacre_idx_seq':  sacre_idx_seq,
    }
