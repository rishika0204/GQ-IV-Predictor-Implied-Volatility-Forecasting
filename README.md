# 📈 GQ-IV Predictor – Implied Volatility Forecasting

## Overview

This notebook is a **starter framework for predicting short-horizon implied volatility (IV)** in cryptocurrency markets. It combines **limit order book (LOB) microstructure features** with advanced **machine learning models** to forecast volatility over a 10-second horizon, helping evaluate market microdynamics and potential trading signals.

---

## ✨ Features

* **Data Engineering**

  * Constructs LOB-based features: spreads, queue imbalance, microprice dynamics, and realized volatility estimates.
  * Supports peer-asset pretraining using correlated crypto assets (e.g., BTC, SOL, DOGE).

* **Volatility Modeling**

  * Targets **10-second-ahead implied volatility** prediction.
  * Includes feature pipelines designed to avoid leakage and ensure time-consistent evaluation.

* **Evaluation Framework**

  * Implements **purged walk-forward cross-validation** to prevent lookahead bias in time series.
  * Custom **Pearson correlation-based loss function** for aligning forecasts with realized volatility patterns.

* **Machine Learning Models**

  * Baseline linear/GBM models for rapid iteration.
  * Optimized **LightGBM** training for higher predictive accuracy and robustness.

---

## ⚙️ Requirements

* Python 3.8+
* pandas, numpy
* scikit-learn
* lightgbm
* matplotlib, seaborn

Install dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

---

## 🚀 Usage

1. Open the notebook in Jupyter or VS Code:

   ```bash
   jupyter notebook gqimpliedvol-starter-notebook.ipynb
   ```

2. Run the data preprocessing cells to generate microstructure features.

3. Train models using the provided **LightGBM + custom loss objective**.

4. Evaluate results with **purged walk-forward validation** and correlation metrics.

---

## 📊 Example Workflow

* Generate LOB-derived features.
* Train LightGBM with correlation-based objective.
* Validate on rolling windows with purged CV.
* Visualize predicted vs realized volatility.

---

## 📝 Notes

* The notebook is structured for experimentation; extend it with deep learning (e.g., LSTM, Transformer-based time-series models) or more complex volatility surfaces.
* Suitable for **quant research, strategy prototyping, and model benchmarking** in crypto microstructure studies.

