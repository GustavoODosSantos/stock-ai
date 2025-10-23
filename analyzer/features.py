# analyzer/features.py
# Feature engineering for daily OHLCV data
# Scalable to decades of data (all vectorized; no Python loops)

from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- basic helpers ----------

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=1).mean()

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def _wilder_ema(s: pd.Series, n: int) -> pd.Series:
    # Wilder smoothing == EMA with alpha = 1/n
    return s.ewm(alpha=1.0/n, adjust=False, min_periods=n).mean()

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

def _rolling_percentile_current_value(s: pd.Series, n: int) -> pd.Series:
    # Percentile of current value within the last n observations
    # Uses rolling window + proportion <= last value
    def pct(w: np.ndarray) -> float:
        if len(w) == 0:
            return np.nan
        last = w[-1]
        return float((w <= last).mean())
    return s.rolling(n, min_periods=1).apply(pct, raw=True)


# ---------- indicators ----------

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder_ema(gain, n)
    avg_loss = _wilder_ema(loss, n)
    rs = _safe_div(avg_gain, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    atr = _wilder_ema(tr, n)
    return tr, atr  # return TR too for other uses

def _adx_di(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    # +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0.0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0.0)

    tr = _true_range(high, low, close)
    atr = _wilder_ema(tr, n)

    plus_di = 100 * _safe_div(_wilder_ema(plus_dm, n), atr)
    minus_di = 100 * _safe_div(_wilder_ema(minus_dm, n), atr)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di).replace(0, np.nan))
    adx = _wilder_ema(dx, n)
    return adx, plus_di, minus_di

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = _sma(close, n)
    stdev = close.rolling(n, min_periods=n).std()
    up = mid + k * stdev
    lo = mid - k * stdev
    width = (up - lo) / mid.replace(0, np.nan)  # relative width
    return mid, up, lo, width, stdev


# ---------- main: feature builder ----------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns: date, open, high, low, close, (volume optional)
    Output: Original columns + engineered features (no printing, ready for modeling/report)
    """
    # Ensure required columns exist
    needed = {"date", "open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for features: {sorted(missing)}")

    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    o = out["open"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)
    v = out["volume"].astype(float) if "volume" in out.columns else pd.Series(index=out.index, dtype=float)

    # Returns
    out["ret_1"]  = c.pct_change(1)
    out["ret_5"]  = c.pct_change(5)
    out["ret_10"] = c.pct_change(10)
    out["ret_21"] = c.pct_change(21)

    # Moving averages (trend)
    out["sma_5"]   = _sma(c, 5)
    out["sma_20"]  = _sma(c, 20)
    out["sma_50"]  = _sma(c, 50)
    out["sma_200"] = _sma(c, 200)

    out["ema_12"] = _ema(c, 12)
    out["ema_26"] = _ema(c, 26)
    out["ema_50"] = _ema(c, 50)

    # MACD
    macd_line, macd_signal, macd_hist = _macd(c, 12, 26, 9)
    out["macd_line"]   = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"]   = macd_hist

    # RSI
    out["rsi_14"] = _rsi(c, 14)

    # ATR / TR
    tr, atr_14 = _atr(h, l, c, 14)
    out["tr"] = tr
    out["atr_14"] = atr_14

    # ADX + DI
    adx_14, di_plus_14, di_minus_14 = _adx_di(h, l, c, 14)
    out["adx_14"] = adx_14
    out["di_plus_14"] = di_plus_14
    out["di_minus_14"] = di_minus_14

    # Bollinger Bands (20, 2)
    bb_mid_20, bb_up, bb_lo, bb_width, bb_stdev = _bollinger(c, 20, 2.0)
    out["bb_mid_20"]   = bb_mid_20
    out["bb_up_20_2"]  = bb_up
    out["bb_lo_20_2"]  = bb_lo
    out["bb_width_20"] = bb_width
    out["stdev_20"]    = bb_stdev  # keep raw stdev(20) too

    # Volatility (extra)
    out["stdev_10"] = c.pct_change(1).rolling(10, min_periods=10).std()

    # BB width percentile vs. ~1yr (252 trading days)
    out["bb_width_pct_252"] = _rolling_percentile_current_value(out["bb_width_20"], 252)

    # Volume stats (if available)
    if "volume" in df.columns:
        out["vol_ma20"] = _sma(v, 20)
        out["vol_ratio"] = _safe_div(v, out["vol_ma20"])
        out["vol_spike_flag"] = (out["vol_ratio"] >= 1.5).astype(int)
    else:
        out["vol_ma20"] = np.nan
        out["vol_ratio"] = np.nan
        out["vol_spike_flag"] = 0

    # Candle strength metrics
    range_ = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper_wick = (h - c).where(c >= o, h - o)  # if green candle: h-c ; if red: h-o
    lower_wick = (o - l).where(c >= o, c - l)  # if green: o-l ; if red: c-l

    out["body_pct"]       = _safe_div(body, range_)
    out["upper_wick_pct"] = _safe_div(upper_wick, range_)
    out["lower_wick_pct"] = _safe_div(lower_wick, range_)

    # Time context
    dt = pd.to_datetime(out["date"])
    out["day_of_week"] = dt.dt.dayofweek  # 0=Mon ... 6=Sun
    out["month"] = dt.dt.month

    # Clean up NaNs from rolling windows at the head
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[
        "ret_1","sma_20","sma_50","sma_200","ema_12","ema_26","macd_line",
        "rsi_14","atr_14","adx_14","bb_mid_20","bb_width_20","stdev_10"
    ]).reset_index(drop=True)

    return out
