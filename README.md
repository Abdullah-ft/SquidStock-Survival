#  SquidStock Survival: Neon-Lit Market Forecasting 📈

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/SciKit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![YFinance](https://img.shields.io/badge/Yahoo_Finance-5F00D1?style=for-the-badge&logo=yahoofinance&logoColor=white)

**Dive into SquidStock Survival: your neon-lit gateway to smarter stock decisions!** 🚀

This interactive Streamlit app blends sharp financial analysis with predictive machine learning. Fetch live Yahoo Finance data or upload your own CSV, visualize market dynamics with stunning Plotly charts, and gain a 5-day edge with price & trend forecasts powered by Linear & Logistic Regression. Navigate the market's high-stakes game with AI-driven insights wrapped in a unique, engaging "Squid Game" inspired aesthetic. **Can your strategy survive?** 👾

---

## 📚 Course Context

-   **Course:** Programming for Finance
-   **Instructor:** Dr. Usama Arshad

---

## 🔗 Live Access & Showcase

🔗 **Experience it Live:**  (https://squidstock-survival-e5mmm46pdubakwutnjfzrc.streamlit.app/)

🎥 **Watch the Walkthrough:** https://shorturl.at/D8G9R


---

## ✨ Key Features

*   **Dynamic Data Input:** Fetch real-time stock data using Yahoo Finance (`yfinance`) by ticker symbol or upload your historical data via CSV.
*   **Comprehensive Analysis:**
    *   Display raw and processed data tables.
    *   Visualize historical closing prices, volume, and Moving Averages.
    *   Plot candlestick charts for technical analysis patterns.
    *   Calculate and display key financial statistics.
*   **ML-Powered Forecasting:**
    *   **Price Prediction:** Uses Linear Regression to forecast the closing price for the next 5 trading days.
    *   **Trend Prediction:** Employs Logistic Regression to predict the market trend (Up/Down) for the next 5 trading days.
*   **Interactive Visualization:** Leverages Plotly for interactive, zoomable, and informative charts.
*   **Engaging UI:** Unique "Squid Game" inspired theme with neon colors and custom components for a memorable user experience.
*   **Data Handling:** Robust data loading, cleaning, and feature engineering using Pandas and Scikit-learn.

---

## 🛠️ Tech Stack

*   **Frontend/UI:** Streamlit
*   **Data Handling:** Pandas
*   **Machine Learning:** Scikit-learn
*   **Data Fetching:** yfinance (Yahoo Finance API)
*   **Plotting:** Plotly Express
*   **Core Language:** Python (3.8+)

---

## 🚀 Getting Started Locally

Follow these steps to set up and run SquidStock Survival on your own machine.

### ✅ Prerequisites

*   **Python:** Version 3.8 or higher installed. [Download Python](https://www.python.org/downloads/)
*   **pip:** Python package installer (usually comes with Python).
*   **Git:** Version control system for cloning the repository. [Download Git](https://git-scm.com/downloads/)
*   **(Optional but Recommended) Virtual Environment Tool:** Like `venv` (built-in) or `conda`.

### ⚙️ Installation Steps

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```
    *(Replace `<your-username>` and `<your-repo-name>` with your actual GitHub details)*

2.  **Create & Activate a Virtual Environment (Recommended):**
    This isolates project dependencies.

    *Using `venv` (standard Python):*
    ```bash
    # Create environment (replace '.venv' with your preferred name)
    python -m venv .venv

    # Activate environment
    # Windows (Command Prompt)
    .\.venv\Scripts\activate
    # Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1
    # macOS / Linux (Bash/Zsh)
    source .venv/bin/activate
    ```
    You should see `(.venv)` prefixing your terminal prompt.

3.  **Install Dependencies:**
    Ensure you have a `requirements.txt` file in the repository root containing all necessary libraries (e.g., streamlit, pandas, scikit-learn, plotly, yfinance).
    ```bash
    pip install -r requirements.txt
    ```
    This command reads the file and installs the specified versions of the packages.

4.  **Prepare Your Data (If Using CSV):**
    *   Ensure your CSV file has at least columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
    *   The `Date` column should be in a recognizable date format (e.g., `YYYY-MM-DD`).
    *   Place your CSV file within the project directory or note its path.

### ▶️ Running the Application

1.  **Navigate to the Project Directory:**
    Make sure your terminal is still in the `<your-repo-name>` directory where your main Streamlit script (e.g., `app.py`, `main.py`, `squidstock.py`) is located.

2.  **Launch the Streamlit App:**
    Run the following command, replacing `<your_script_name>.py` with the actual name of your main Python file:
    ```bash
    streamlit run <your_script_name>.py
    ```
    *Example:* If your main file is `app.py`, use `streamlit run app.py`

3.  **Access the App:**
    Streamlit will automatically open the application in your default web browser. It will also display the local URL (usually `http://localhost:8501`) in your terminal.

---

## 📝 How It Works (Simplified Flow)

1.  **Data Input:** User selects a stock ticker (data fetched via `yfinance`) or uploads a CSV file.
2.  **Data Processing:** Pandas is used to load, clean, and prepare the data (e.g., handling missing values, setting date index).
3.  **Feature Engineering:** Relevant features for the ML models are created (e.g., lagged prices, moving averages, target variables for price/trend).
4.  **Model Training:**
    *   **Linear Regression:** Trained on historical data to predict the next day's 'Close' price.
