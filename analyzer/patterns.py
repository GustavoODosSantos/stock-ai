import pandas as pd

class PatternDetector:
    def __init__(self, df: pd.DataFrame):
        # a clean internal copy
        self.df = df.copy().reset_index(drop=True)

        # Normalize column names (just in case)
        self.df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close'
        }, inplace=True)

        # Convert to float
        for col in ['open', 'high', 'low', 'close']:
            self.df[col] = self.df[col].astype(float)

        # Helpful columns for patterns
        self.df["bullish"] = self.df["close"] > self.df["open"]
        self.df["bearish"] = self.df["close"] < self.df["open"]


    def bullish_engulfing(self):
        df = self.df

        # Conditions for bullish engulfing
        cond1 = df["bearish"].shift(1)        # Previous candle is bearish
        cond2 = df["bullish"]                 # Current candle is bullish
        cond3 = df["open"] < df["close"].shift(1)   # Current open is below previous close
        cond4 = df["close"] > df["open"].shift(1)   # Current close is above previous open

        pattern = cond1 & cond2 & cond3 & cond4
        df["bullish_engulfing"] = pattern

        return df["bullish_engulfing"]

    def bearish_engulfing(self):
        df = self.df

        # Conditions for bearish engulfing
        cond1 = df["bullish"].shift(1)        # Previous candle is bullish
        cond2 = df["bearish"]                 # Current candle is bearish
        cond3 = df["open"] > df["close"].shift(1)   # Current open is above previous close
        cond4 = df["close"] < df["open"].shift(1)   # Current close is below previous open

        pattern = cond1 & cond2 & cond3 & cond4
        df["bearish_engulfing"] = pattern

        return df["bearish_engulfing"]

    def hammer(self):
        df = self.df

        # Calculate candle components
        body = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        # Hammer conditions
        cond1 = lower_wick >= 2 * body     # big lower shadow
        cond2 = upper_wick <= body         # small upper shadow
        cond3 = body > 0                   # avoid division / doji issues

        pattern = cond1 & cond2 & cond3
        df["hammer"] = pattern

        return df["hammer"]
    
    def shooting_star(self):
        df = self.df

        # Candle components
        body = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        # Shooting Star conditions
        cond1 = upper_wick >= 2 * body     # big upper shadow
        cond2 = lower_wick <= body         # small lower shadow
        cond3 = body > 0                   # avoid doji

        pattern = cond1 & cond2 & cond3
        df["shooting_star"] = pattern

        return df["shooting_star"]

    def doji(self, threshold: float = 0.1):
        """
        Detects Doji candles.
        threshold: body size percentage relative to full candle range.
                   default = 10% (0.1) of total range.
        """

        df = self.df

        # Candle components
        body = (df["close"] - df["open"]).abs()
        full_range = df["high"] - df["low"]

        # Avoid division by zero
        full_range = full_range.replace(0, 1)

        # Body must be very small
        cond1 = body / full_range <= threshold

        df["doji"] = cond1
        return df["doji"]
    
    def inside_bar(self):
        df = self.df

        # Previous candle's range
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)

        # Inside Bar conditions
        cond1 = df["high"] <= prev_high
        cond2 = df["low"] >= prev_low

        pattern = cond1 & cond2
        df["inside_bar"] = pattern

        return df["inside_bar"]

    def outside_bar(self):
        df = self.df

        # Previous candle's range
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)

        # Outside Bar conditions
        cond1 = df["high"] >= prev_high
        cond2 = df["low"] <= prev_low

        pattern = cond1 & cond2
        df["outside_bar"] = pattern

        return df["outside_bar"]

    def morning_star(self):
        df = self.df
        
        # Candle components
        body = (df["close"] - df["open"]).abs()
        
        # Shifted components for 3-candle pattern
        body_prev1 = body.shift(1)
        body_prev2 = body.shift(2)
        
        close_prev1 = df["close"].shift(1)
        close_prev2 = df["close"].shift(2)
        
        open_prev1 = df["open"].shift(1)
        open_prev2 = df["open"].shift(2)
        
        # Pattern conditions
        cond1 = df["bearish"].shift(2)                     # Candle 1 bearish
        cond2 = body_prev1 <= body_prev2 * 0.5             # Candle 2 small body (50% or less)
        cond3 = df["bullish"]                              # Candle 3 bullish
        cond4 = df["close"] > (open_prev2 + close_prev2) / 2   # Candle 3 closes into Candle 1's body
        
        pattern = cond1 & cond2 & cond3 & cond4
        df["morning_star"] = pattern
        
        return df["morning_star"]

    def evening_star(self):
        df = self.df
        
        # Candle components
        body = (df["close"] - df["open"]).abs()
        
        # Shifted components
        body_prev1 = body.shift(1)
        body_prev2 = body.shift(2)
        
        close_prev1 = df["close"].shift(1)
        close_prev2 = df["close"].shift(2)
        
        open_prev1 = df["open"].shift(1)
        open_prev2 = df["open"].shift(2)
        
        # Pattern conditions
        cond1 = df["bullish"].shift(2)                      # Candle 1 bullish
        cond2 = body_prev1 <= body_prev2 * 0.5              # Candle 2 small
        cond3 = df["bearish"]                               # Candle 3 bearish
        cond4 = df["close"] < (open_prev2 + close_prev2) / 2   # Candle 3 closes deep
        
        pattern = cond1 & cond2 & cond3 & cond4
        df["evening_star"] = pattern
        
        return df["evening_star"]
    
    def detect_all(self):
        """
        Runs all implemented patterns and returns the dataframe
        with all pattern columns included.
        """
        self.bullish_engulfing()
        self.bearish_engulfing()
        self.hammer()
        self.shooting_star()
        self.doji()
        self.inside_bar()
        self.outside_bar()
        self.morning_star()
        self.evening_star()

        return self.df
