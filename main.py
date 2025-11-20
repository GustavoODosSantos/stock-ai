from analyzer.data_loader import DataLoader
from analyzer.features import build_features
from analyzer.patterns import PatternDetector
from analyzer.model import Model
from analyzer.report import Report


def run_analysis(file_path):
    print("\n=== Starting Stock Analysis ===\n")

    # 1. LOAD DATA
    print("1. Loading data...")
    loader = DataLoader(file_path)
    df = loader.load_csv()                     # ✔ Load DataFrame

    # 2. BUILD FEATURES
    print("2. Computing features...")
    df = build_features(df)                    # ✔ Adds RSI, ATR, MACD, trend, etc.

    # 3. DETECT PATTERNS
    print("3. Detecting patterns...")
    df = PatternDetector(df).detect_all()      # ✔ Correct method name

    # 4. RUN MODEL
    print("4. Running model...")
    model = Model(df)                          # ✔ Model requires df
    model_output = model.summary()             # ✔ Uses hybrid_probability()

    prob = model_output["probability_next_bullish"]

    # 5. GENERATE CHART + REPORT
    print("5. Generating chart & PDF report...")
    report = Report(df, model_output)
    chart_path = report.plot_candles("chart.png")
    pdf_path = report.export_pdf("report.pdf", chart_path)

    print("\n === ANALYSIS COMPLETE ===")
    print(f"Chart saved as: {chart_path}")
    print(f"Report saved as: {pdf_path}")
    print(f"Next Bullish Candle Probability: {prob}%")
    print("============================\n")


if __name__ == "__main__":
    print("=== STOCK ANALYSIS TOOL ===")
    file_path = input("Enter the path or name of the CSV file: ").strip()

    if not file_path.lower().endswith(".csv"):
        file_path = f"data/{file_path}.csv"

    run_analysis(file_path)
    