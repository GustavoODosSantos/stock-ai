import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_csv(self):
        # Load the CSV file
        df = pd.read_csv(self.filepath)

        # Auto-detect and rename any known column formats
        df = self._auto_rename_columns(df)

        # Normalize all column names (lowercase, no spaces)
        df.columns = [c.lower().strip() for c in df.columns]

        # Required columns for the project
        required = ["date", "open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert date column
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Convert price columns to numeric
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle volume column (optional)
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        else:
            df["volume"] = 0.0

        # Sort chronologically
        df = df.sort_values("date").reset_index(drop=True)

        # Detect timeframe
        self.timeframe = self.detect_timeframe(df)

        return df


        return df

    def _auto_rename_columns(self, df):
        rename_map = {
            # Common variations for date
            "date": "date",
            "time": "date",
            "datetime": "date",
            "timestamp": "date",
            "open time": "date",

            # Prices
            "open": "open",
            "o": "open",
            "high": "high",
            "h": "high",
            "low": "low",
            "l": "low",
            "close": "close",
            "c": "close",
            "adj close": "close",

            # Volume
            "volume": "volume",
            "v": "volume",
            "vol": "volume",
            "tickvol": "volume",
        }

        new_cols = {}
        for col in df.columns:
            key = col.lower().strip()
            new_cols[col] = rename_map.get(key, key)

        df = df.rename(columns=new_cols)
        return df
   
   
    def detect_timeframe(self, df):
        """
        Detects the timeframe by calculating the median difference between timestamps.
        Returns: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo'
        """

        # Need at least 2 rows
        if len(df) < 2:
            return "unknown"

        # Time difference between each row
        diffs = df["date"].diff().dropna()

        # Median difference (most consistent)
        delta = diffs.median()

        # Convert to seconds
        seconds = delta.total_seconds()

        # Match thresholds
        if seconds < 60 * 2:
            return "1m"
        elif seconds < 60 * 10:
            return "5m"
        elif seconds < 60 * 20:
            return "15m"
        elif seconds < 60 * 40:
            return "30m"
        elif seconds < 60 * 90:
            return "1h"
        elif seconds < 60 * 60 * 3:
            return "2h"
        elif seconds < 60 * 60 * 6:
            return "4h"
        elif seconds < 60 * 60 * 12:
            return "12h"
        elif seconds < 60 * 60 * 24 * 2:
            return "1d"
        elif seconds < 60 * 60 * 24 * 10:
            return "1w"
        else:
            return "1mo"
