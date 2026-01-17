# 📈 Market Trend Analysis System
### Intelligent Stock Scoring, Forecasting & Risk-Aware Decision Framewor
---

## 🧠 Project Overview

This project presents a **comprehensive market trend analysis system**
designed to evaluate stocks using a **multi-layered, risk-aware approach**.

Unlike traditional tools that rely on isolated indicators, this system:
- Combines **fundamental, technical, and probabilistic models**
- Adapts to **changing market regimes**
- Converts forecasts into **actionable investment decisions**

The goal is not just prediction — but **better decision-making under uncertainty**.

---

## 🏗️ System Architecture

Market Data (Price + Fundamentals)
│
▼
📊 Scoring Engine
│
▼
🔮 Forecasting Engine
(ARIMA + Regime + Monte Carlo)
│
▼
🧠 Meta-Ensemble Layer
│
▼
🧪 Decision & Risk Lab

yaml
Copy code

---

## 📂 Project Structure

Market_trend_analysis/

│

├── app.py # Streamlit application
├── Final_Market_Trend_Analysis.ipynb # 📘 Main notebook (START HERE)
│
├── scoring.py # Fundamental & momentum scoring logic
├── forecast.py # Multi-model forecasting system
├── decision_risk_lab.py # Decision & risk evaluation
│
├── insights/ # Modular analytical engines
│ ├── price.py
│ ├── momentum.py
│ ├── fundamentals.py
│ ├── scoring.py
│ ├── forecast.py
│ └── engines.py
│
├── *.png # Dashboards, forecasting & decision visuals
├── requirement.txt # Project dependencies
└── .gitignore

yaml
Copy code

---

## 🧭 Application Flow (Page-wise)

---

### 🏠 1️⃣ Dashboard
**Purpose:**
- High-level overview of selected stock
- Quick snapshot of market context

**Includes:**
- Current price & trend
- Market regime indicator
- Navigation to analysis modules

---

### 📊 2️⃣ Scoring Module
**Objective:** Quantify overall stock quality.

**Scoring Pillars:**
- Valuation (P/E, P/B)
- Profitability (ROE, margins)
- Growth (revenue & earnings)
- Financial Health (debt, liquidity)
- Momentum (RSI, volatility, trend)

**Outputs:**
- Total score (0–100)
- Pillar-wise breakdown
- Recommendation & confidence level

---

### 🔮 3️⃣ Forecasting Module
This is the **core intelligence layer**.

**Models Used:**
- **ARIMA** – Short-term trend modeling
- **Hidden Markov Model (HMM)** – Market regime detection
- **Monte Carlo Simulation** – Risk & uncertainty modeling
- **Meta-Ensemble** – Intelligent blending of all models

**Forecast Horizons:**
- 20 Days
- 60 Days
- 120 Days
- 1 Year

**Outputs:**
- Expected return & expected price
- Regime-adjusted forecasts
- Confidence & stability indicators

---

### 🧪 4️⃣ Decision & Risk Lab
**Purpose:** Convert forecasts into real investment decisions.

**Evaluates:**
- Upside vs downside balance
- Value-at-Risk (VaR)
- Drawdown probabilities
- Market stability

**Focus:**
> Even strong return forecasts can be downgraded
> if associated risk is unacceptably high.

This ensures decisions are **practical, explainable, and risk-aware**.

---

## 🧩 Key Design Principles

- ✅ Multi-model (no single point of failure)
- ✅ Regime-aware (markets are non-stationary)
- ✅ Risk-first (not blindly optimistic)
- ✅ Modular & extensible architecture

---

## 🖥️ Streamlit Application

Run the interactive web app locally:

```bash
pip install -r requirement.txt
streamlit run app.py
