import pandas as pd

class Model:
    def __init__(self, df: pd.DataFrame):
        """
        df must already contain:
        - features (RSI, MACD, ATR, etc.)
        - patterns (hammer, engulfing, etc.)
        - trend column
        """
        self.df = df.copy().reset_index(drop=True)
        self.hybrid_prob = None
    
    def _momentum_state(self, df):
        """Classifies momentum based on RSI and MACD histogram."""
        # RSI: above 55 = bullish momentum
        if df["rsi14"].iloc[-1] > 55 and df["macd_hist"].iloc[-1] > 0:
            return "bullish"

        # RSI below 45 = bearish momentum
        if df["rsi14"].iloc[-1] < 45 and df["macd_hist"].iloc[-1] < 0:
            return "bearish"

        return "neutral"
    
    def _volatility_state(self, df):
        """Classifies volatility using ATR."""
        atr_today = df["atr14"].iloc[-1]
        median_atr = df["atr14"].median()

        if atr_today > median_atr * 1.2:
            return "high"

        if atr_today < median_atr * 0.8:
            return "low"

        return "normal"
    
    def classify_states(self):
        df = self.df

        df["momentum_state"] = self._momentum_state(df)
        df["volatility_state"] = self._volatility_state(df)

        return df


    def hybrid_probability(self):
        """
        Hybrid probability model:
        - strict match for primary pattern + trend
        - flexible match for momentum + volatility
        - checks entire history
        - computes next-candle real bullish probability
        """

        df = self.df.copy()

        # 1. Identify today's conditions
        last = df.iloc[-1]

        # Primary pattern (strict)
        patterns = [
            "bullish_engulfing", "bearish_engulfing",
            "hammer", "shooting_star",
            "morning_star", "evening_star",
            "inside_bar", "outside_bar"
        ]

        primary_pattern = None
        for p in patterns:
            if last.get(p, False):
                primary_pattern = p
                break

        # If no pattern, fallback to 'none'
        if primary_pattern is None:
            primary_pattern = "none"

        # Trend (strict)
        current_trend = last["trend"]

        # Momentum (flexible)
        current_momentum = self._momentum_state(df)

        # Volatility (flexible)
        current_volatility = self._volatility_state(df)

        # 2. Find similar historical cases
        matches = df.copy()

        # strict match: same pattern
        if primary_pattern != "none":
            matches = matches[matches[primary_pattern] == True]

        # strict match: same trend
        matches = matches[matches["trend"] == current_trend]

        # flexible match: momentum
        matches = matches[matches["momentum_state"] == current_momentum]

        # flexible match: volatility
        matches = matches[matches["volatility_state"] == current_volatility]

        # shift matches forward by 1 to see what happened next
        next_moves = matches["bullish"].shift(-1)

        # remove NaN (last row)
        next_moves = next_moves.dropna()

        if len(next_moves) == 0:
            return 50.0  # no information

        # probability
        prob = next_moves.mean() * 100
        self.hybrid_prob = round(prob, 2)

        return self.hybrid_prob

    def summary(self):
        """Convenience function for use in reports."""
        self.classify_states()
        self.hybrid_probability()

        return {
            "probability_next_bullish": self.hybrid_prob,
            "last_trend": self.df["trend"].iloc[-1],
            "last_momentum": self._momentum_state(self.df),
            "last_volatility": self._volatility_state(self.df)
        }
    
