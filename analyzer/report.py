import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Avoid GUI issues when generating charts
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


class Report:
    def __init__(self, df, model_output):
        """
        df : pandas DataFrame with features & patterns
        model_output : dictionary from the model.summary()
        """
        self.df = df
        self.model_output = model_output

    # =========================================================
    # 1. SUMMARY SECTION
    # =========================================================
    def generate_text_summary(self):
        last = self.df.iloc[-1]

        prob = self.model_output["probability_next_bullish"]
        trend = self.model_output["last_trend"]
        momentum = self.model_output["last_momentum"]
        volatility = self.model_output["last_volatility"]

        pattern_cols = [
            "bullish_engulfing", "bearish_engulfing",
            "hammer", "shooting_star",
            "morning_star", "evening_star",
            "inside_bar", "outside_bar"
        ]

        active_patterns = [
            p.replace("_", " ").title()
            for p in pattern_cols
            if p in self.df.columns and last.get(p, False)
        ]

        patterns_text = ", ".join(active_patterns) if active_patterns else "None"

        summary = f"""
        <b>Summary of Last Candle</b><br/><br/>
        <b>Trend:</b> {trend}<br/>
        <b>Momentum:</b> {momentum}<br/>
        <b>Volatility:</b> {volatility}<br/>
        <b>Detected Patterns:</b> {patterns_text}<br/>
        <b>Probability of Next Bullish Candle:</b> {prob}%<br/>
        """

        return summary

    # =========================================================
    # 2. DETAILED ANALYSIS
    # =========================================================
    def generate_detailed_analysis(self):
        last = self.df.iloc[-1]

        prob = self.model_output["probability_next_bullish"]
        trend = self.model_output["last_trend"]
        momentum = self.model_output["last_momentum"]
        volatility = self.model_output["last_volatility"]

        pattern_cols = [
            "bullish_engulfing", "bearish_engulfing",
            "hammer", "shooting_star",
            "morning_star", "evening_star",
            "inside_bar", "outside_bar"
        ]

        active_patterns = [
            p.replace("_", " ").title()
            for p in pattern_cols
            if p in self.df.columns and last.get(p, False)
        ]

        analysis = []

        # TREND ANALYSIS
        if trend == "uptrend":
            analysis.append(
                "The stock is in an <b>uptrend</b>, with price trading above key moving averages. "
                "This environment typically favors bullish continuation."
            )
        elif trend == "downtrend":
            analysis.append(
                "The stock is in a <b>downtrend</b>, reflecting consistent selling pressure. "
                "Bullish signals tend to have weaker follow-through here."
            )
        else:
            analysis.append(
                "The stock is currently <b>sideways</b>, showing consolidation and indecision."
            )

        # MOMENTUM ANALYSIS
        if momentum == "bullish":
            analysis.append(
                "Momentum indicators reflect <b>bullish strength</b>, suggesting buyers are in control."
            )
        elif momentum == "bearish":
            analysis.append(
                "Momentum indicators show <b>bearish pressure</b>, indicating sellers dominate."
            )
        else:
            analysis.append("Momentum is <b>neutral</b>, showing no strong directional force.")

        # VOLATILITY ANALYSIS
        if volatility == "high":
            analysis.append(
                "Volatility is <b>high</b>, producing larger price swings. This increases opportunity but also risk."
            )
        elif volatility == "low":
            analysis.append(
                "Volatility is <b>low</b>, often preceding a breakout or expansion in price range."
            )
        else:
            analysis.append("Volatility is at a <b>normal</b> level, indicating stable price movement.")

        # PATTERN ANALYSIS
        if active_patterns:
            analysis.append(
                f"The following candle pattern(s) were detected: <b>{', '.join(active_patterns)}</b>. "
                "These patterns, combined with trend and momentum, provide meaningful signals."
            )
        else:
            analysis.append("No major candle patterns were detected in the most recent candle.")

        # PROBABILITY INTERPRETATION
        if prob >= 60:
            analysis.append(
                f"The model assigns a <b>{prob}% probability</b> that the next candle will be bullish. "
                "This indicates favorable bullish conditions."
            )
        elif prob <= 40:
            analysis.append(
                f"The model estimates only <b>{prob}% probability</b> of a bullish candle, "
                "signaling stronger bearish conditions."
            )
        else:
            analysis.append(
                f"The model shows a <b>{prob}% probability</b>, indicating neutral or mixed conditions."
            )

        analysis.append(
            "This analysis incorporates trend structure, momentum strength, volatility levels, and "
            "candle patterns to estimate likely future price behavior."
        )

        return "<br/><br/>".join(analysis)

    # =========================================================
    # 3. CANDLE CHART WITH PATTERN MARKERS
    # =========================================================
    def plot_candles(self, save_path="chart.png", last=100):
        df = self.df.tail(last).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, row in df.iterrows():
            o = row["open"]
            c = row["close"]
            h = row["high"]
            l = row["low"]

            color = "green" if c > o else "red"

            # Wick
            ax.plot([i, i], [l, h], color=color, linewidth=1)

            # Body
            rect = Rectangle(
                (i - 0.3, min(o, c)),
                0.6,
                abs(c - o),
                color=color
            )
            ax.add_patch(rect)

            # === PATTERN MARKERS ===
            if "bullish_engulfing" in df.columns and row["bullish_engulfing"]:
                ax.scatter(i, l, color="green", marker="^", s=60)

            if "bearish_engulfing" in df.columns and row["bearish_engulfing"]:
                ax.scatter(i, h, color="red", marker="v", s=60)

            if "hammer" in df.columns and row["hammer"]:
                ax.scatter(i, l, color="blue", marker="o", s=50)

            if "shooting_star" in df.columns and row["shooting_star"]:
                ax.scatter(i, h, color="purple", marker="o", s=50)

            if "morning_star" in df.columns and row["morning_star"]:
                ax.scatter(i, l, color="gold", marker="*", s=120)

            if "evening_star" in df.columns and row["evening_star"]:
                ax.scatter(i, h, color="black", marker="*", s=120)

        # EMA LINES
        if "ema_short" in df.columns:
            ax.plot(df.index, df["ema_short"], label="EMA Short", color="blue", linewidth=1.5)

        if "ema_long" in df.columns:
            ax.plot(df.index, df["ema_long"], label="EMA Long", color="orange", linewidth=1.5)

        ax.set_title("Candlestick Chart (with Patterns)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Price")
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        return save_path

    # =========================================================
    def export_pdf(self, filename="report.pdf", chart_path="chart.png"):
        pdf = SimpleDocTemplate(
            filename,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        Story = []

        # Title
        title_style = styles["Title"]
        title_style.alignment = TA_CENTER
        Story.append(Paragraph("Stock Technical Analysis Report", title_style))
        Story.append(Spacer(1, 20))

        # Summary
        summary = self.generate_text_summary()
        Story.append(Paragraph(summary, styles["BodyText"]))
        Story.append(Spacer(1, 20))

        # Detailed Analysis
        detailed = self.generate_detailed_analysis()
        Story.append(Paragraph(detailed, styles["BodyText"]))
        Story.append(Spacer(1, 20))

        # Chart
        try:
            img = Image(chart_path, width=6*inch, height=3*inch)
            Story.append(img)
        except:
            Story.append(Paragraph("Chart failed to load.", styles["BodyText"]))

        pdf.build(Story)
        return filename
