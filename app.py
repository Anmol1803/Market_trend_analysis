# 2.py - UPDATED VERSION WITH VISUAL INTELLIGENCE + ENHANCED INTERPRETATION
import streamlit as st
st.set_option("client.showErrorDetails", False)
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from pandas.io.formats.style import Styler
from datetime import date
from scoring import calculate_stock_score
from insights.engines import generate_insight
from scoring import calculate_stock_score, get_debt_ratio
from decision_risk_lab import DecisionRiskLab
from forecast import (
    CompatibleForecastSystem,      # Main interface for forecasting
                
)


# ==================================================
# PREVENT AUTO-REFRESH
# ==================================================
st.set_page_config(page_title="Investor NSE Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Add this line RIGHT HERE:
if 'prevent_refresh' not in st.session_state:
    st.session_state.prevent_refresh = True

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Investor NSE Dashboard", layout="wide")


# ==================================================
# SESSION STATE INITIALIZATION (IMPORTANT)
# ==================================================
if "df" not in st.session_state:
    st.session_state["df"] = None

if "info" not in st.session_state:
    st.session_state["info"] = None

if "symbol" not in st.session_state:
    st.session_state["symbol"] = None

if "stock_name" not in st.session_state:
    st.session_state["stock_name"] = None

if "full_df" not in st.session_state:
    st.session_state["full_df"] = None

if "full_info" not in st.session_state:
    st.session_state["full_info"] = None

if "scoring_result" not in st.session_state:
    st.session_state.scoring_result = None

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

if "last_scored_symbol" not in st.session_state:
    st.session_state.last_scored_symbol = None

if "last_forecast_symbol" not in st.session_state:
    st.session_state.last_forecast_symbol = None

if "current_stock_key" not in st.session_state:
    st.session_state.current_stock_key = None


# ==================================================
# YAHOO SEARCH HEADERS
# ==================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

# ==================================================
# SMART SEARCH
# ==================================================
st.sidebar.header("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["📊 Main App", "🧠 Scoring", "🔮 Forecasting", "🧪 Decision & Risk Lab"]
)
st.session_state.current_page = page

# ==================================================
# RESET BUTTON
# ==================================================
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Current Stock Analysis", type="secondary"):
    # Clear only analysis data, keep price data
    st.session_state.scoring_result = None
    st.session_state.forecast_result = None
    st.session_state.last_scored_symbol = None
    st.session_state.last_forecast_symbol = None
    
    # Clear forecast engine if exists
    if 'forecast_engine' in st.session_state:
        del st.session_state.forecast_engine
    
    # Clear forecast_results (plural) if exists for compatibility
    if 'forecast_results' in st.session_state:
        del st.session_state.forecast_results
    
    st.sidebar.success("Analysis reset! Run scoring/forecasting again.")
    st.rerun()


# ==================================================
# ADDED: NARRATIVE FLOW HEADERS
# ==================================================
if page == "📊 Main App":
    st.markdown("*Get the current market picture and technical signals*")
elif page == "🧠 Scoring":
    st.markdown("*Evaluate fundamental health and investment quality*")
elif page == "🔮 Forecasting":
    st.markdown("*Explore probabilistic forecasts based on current regime*")
elif page == "🧪 Decision & Risk Lab":
    st.markdown("## 🧪 **Given all evidence, what should I do?**")

# ==================================================
# NEW: VISUALIZATION HELPER FUNCTIONS
# ==================================================
def style_dataframe(df):
    styled = df.style
    
    # Colour regime column
    def color_regime(val):
        if val == 'Bull':
            return 'background-color: #1e7f43; color: white'
        elif val == 'Bear':
            return 'background-color: #8b1e1e; color: white'
        elif val == 'Sideways':
            return 'background-color: #8a6d1d; color: white'
        return ''
    
    # Colour confidence column
    def color_confidence(val):
        if val == 'High':
            return 'background-color: #145a32; color: white'
        elif val == 'Medium':
            return 'background-color: #7d6608; color: white'
        elif val == 'Low':
            return 'background-color: #78281f; color: white'
        return ''

      # Apply styling
    styled = styled.applymap(color_regime, subset=['Regime'])
    styled = styled.applymap(color_confidence, subset=['Confidence'])
    
    # Colour positive returns green, negative red
    def color_return(val):
        try:
            if '%' in str(val):
                num = float(val.replace('%', '').replace('+', ''))
                if num > 0:
                    return 'color: #1e7f43; font-weight: bold'
                elif num < 0:
                    return 'color: #8b1e1e; font-weight: bold'
        except:
            pass
        return '' 
    styled = styled.applymap(color_return, subset=['Expected Return'])
    
    return styled


# Add this right after your imports
def safe_float(value, default=0):
    """Safely convert to float"""
    if value is None:
        return default
    try:
        return float(value)
    except:
        return default


def create_regime_confidence_gauge(uncertainty_level):
    """Create a gauge chart for regime confidence"""
    confidence = 1 - uncertainty_level
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Market Regime Confidence", 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2E86AB"},
            'steps': [
                {'range': [0, 30], 'color': "rgba(255, 50, 50, 0.3)"},
                {'range': [30, 70], 'color': "rgba(255, 200, 50, 0.3)"},
                {'range': [70, 100], 'color': "rgba(50, 200, 50, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    
    return fig, confidence

def create_regime_contribution_chart(regime_details):
    """Create donut chart for regime contribution"""
    labels = [r['label'] for r in regime_details]
    probabilities = [r['probability'] * 100 for r in regime_details]
    colors = [r['color'] for r in regime_details]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=probabilities,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>Probability: %{percent}<br>Return: %{customdata[0]:.1f}%<br>Volatility: %{customdata[1]:.1f}%<extra></extra>"
    )])
    
    # Add custom data for hover
    fig.data[0].customdata = [
        [r['avg_return_20d']*100, r['avg_volatility_20d']*100] 
        for r in regime_details
    ]
    
    fig.update_layout(
        title="Regime Contribution to Forecast",
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_downside_risk_chart(regime_paths):
    """Create bar chart for downside risk by regime"""
    regimes = []
    weighted_drawdowns = []
    colors = []
    
    for label, data in regime_paths.items():
        if 'paths' in data and len(data['paths']) > 0:
            # Calculate drawdowns for this regime
            drawdowns = []
            for path in data['paths']:
                running_max = np.maximum.accumulate(path)
                drawdown = np.min((path / running_max) - 1)
                drawdowns.append(drawdown)
            
            if drawdowns:
                median_drawdown = np.median(drawdowns)
                # Weight by regime probability
                weighted_dd = median_drawdown * data.get('probability', 1.0)
                regimes.append(label)
                weighted_drawdowns.append(abs(weighted_dd) * 100)  # Convert to percentage
                colors.append(data.get('color', '#666666'))
    
    if not regimes:
        return None
    
    # Sort by risk (highest first)
    sorted_indices = np.argsort(weighted_drawdowns)[::-1]
    regimes = [regimes[i] for i in sorted_indices]
    weighted_drawdowns = [weighted_drawdowns[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    fig = go.Figure(data=[go.Bar(
        x=regimes,
        y=weighted_drawdowns,
        marker_color=colors,
        text=[f"{d:.1f}%" for d in weighted_drawdowns],
        textposition='auto',
        hovertemplate="<b>%{x}</b><br>Weighted Drawdown: %{y:.1f}%<extra></extra>"
    )])
    
    fig.update_layout(
        title="Downside Risk Attribution by Regime",
        xaxis_title="Regime",
        yaxis_title="Weighted Median Drawdown (%)",
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_regime_timeline_chart(historical_df, regime_detector):
    """Create price chart with regime background shading"""
    if regime_detector.gmm is None or len(historical_df) < 100:
        return None
    
    # Get historical regime probabilities
    features = regime_detector.extract_features(historical_df)
    if len(features) == 0:
        return None
    
    features_scaled = regime_detector.scaler.transform(features)
    historical_probs = regime_detector.gmm.predict_proba(features_scaled)
    
    # Find dominant regime for each day
    dominant_regimes = np.argmax(historical_probs, axis=1)
    
    # Align dates
    price_data = historical_df.copy()
    if 'Date' in price_data.columns:
        price_data = price_data.set_index('Date')
    
    # Ensure alignment
    aligned_dates = features.index
    price_data = price_data.loc[aligned_dates]
    
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=aligned_dates,
        y=price_data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='black', width=2)
    ))
    
    # Add regime background shading
    if hasattr(regime_detector, 'regime_stats'):
        # Create shaded regions for each regime
        for regime_id, regime_info in enumerate(regime_detector.regime_stats):
            regime_mask = dominant_regimes == regime_id
            if np.sum(regime_mask) > 10:  # Only show if meaningful duration
                # Find consecutive periods
                regime_changes = np.diff(np.concatenate(([0], regime_mask, [0])))
                regime_starts = np.where(regime_changes == 1)[0]
                regime_ends = np.where(regime_changes == -1)[0]
                
                for start_idx, end_idx in zip(regime_starts, regime_ends):
                    if end_idx > start_idx:
                        fig.add_vrect(
                            x0=aligned_dates[start_idx],
                            x1=aligned_dates[end_idx-1],
                            fillcolor=regime_info['color'],
                            opacity=0.15,
                            line_width=0,
                            annotation_text=regime_info['label'] if (end_idx - start_idx) > 20 else "",
                            annotation_position="top left"
                        )
    
    fig.update_layout(
        title="Price History with Regime Background",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=400,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True
    )
    
    return fig

def create_stability_indicator(quantiles, current_price):
    """Create stability badge based on forecast spread"""
    if '50%' not in quantiles or len(quantiles['50%']) == 0:
        return "Unknown", "gray", ""
    
    median_price = quantiles['50%'][-1]  # Final median forecast
    q25 = quantiles['25%'][-1] if '25%' in quantiles else median_price
    q75 = quantiles['75%'][-1] if '75%' in quantiles else median_price
    
    # Calculate relative spread
    spread = (q75 - q25) / median_price if median_price > 0 else 0
    
    if spread < 0.1:  # Less than 10% spread
        stability = "Stable"
        color = "green"
        explanation = "Narrow forecast range suggests higher confidence"
    elif spread < 0.25:  # 10-25% spread
        stability = "Moderate"
        color = "orange"
        explanation = "Moderate forecast uncertainty"
    else:
        stability = "Fragile"
        color = "red"
        explanation = "Wide forecast range indicates low confidence"
    
    return stability, color, explanation

def create_pillar_barchart(pillars):
    """Create horizontal bar chart for pillar scores"""
    pillar_names = []
    scores = []
    colors = []
    
    for pillar_name, data in pillars.items():
        pillar_names.append(pillar_name)
        scores.append(data['score'])
        
        # Color based on score
        if data['score'] >= 15:
            colors.append('rgba(50, 200, 50, 0.7)')
        elif data['score'] >= 10:
            colors.append('rgba(255, 200, 50, 0.7)')
        else:
            colors.append('rgba(255, 50, 50, 0.7)')
    
    # Sort by score
    sorted_indices = np.argsort(scores)
    pillar_names = [pillar_names[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    fig = go.Figure(data=[go.Bar(
        y=pillar_names,
        x=scores,
        orientation='h',
        marker_color=colors,
        text=[f"{s}/20" for s in scores],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>Score: %{x}/20<extra></extra>"
    )])
    
    fig.update_layout(
        title="Pillar Score Breakdown",
        xaxis_title="Score (out of 20)",
        yaxis_title="Pillar",
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(range=[0, 20])
    )
    
    return fig

def create_regime_compatibility_matrix(pillars, current_regime):
    """Create compatibility matrix between pillars and current regime"""
    # Simplified compatibility logic
    compatibility_matrix = []
    
    for pillar_name, data in pillars.items():
        score = data['score']
        
        # Basic compatibility rules based on regime type
        if current_regime and 'label' in current_regime:
            regime_label = current_regime['label'].lower()
            
            if 'growth' in regime_label:
                if pillar_name in ['Growth', 'Momentum']:
                    compatibility = "🟢 High"
                elif pillar_name in ['Valuation', 'Profitability']:
                    compatibility = "🟡 Moderate"
                else:
                    compatibility = "🔴 Low"
            
            elif 'risk' in regime_label or 'volatile' in regime_label:
                if pillar_name in ['Financial Health', 'Valuation']:
                    compatibility = "🟢 High"
                elif pillar_name == 'Profitability':
                    compatibility = "🟡 Moderate"
                else:
                    compatibility = "🔴 Low"
            
            elif 'calm' in regime_label or 'sideways' in regime_label:
                if pillar_name in ['Valuation', 'Financial Health']:
                    compatibility = "🟢 High"
                else:
                    compatibility = "🟡 Moderate"
            
            else:
                # Default compatibility based on score
                if score >= 15:
                    compatibility = "🟢 High"
                elif score >= 10:
                    compatibility = "🟡 Moderate"
                else:
                    compatibility = "🔴 Low"
        else:
            # No regime info, use score only
            if score >= 15:
                compatibility = "🟢 High"
            elif score >= 10:
                compatibility = "🟡 Moderate"
            else:
                compatibility = "🔴 Low"
        
        compatibility_matrix.append({
            'Pillar': pillar_name,
            'Score': f"{score}/20",
            'Current Regime Compatibility': compatibility
        })
    
    return pd.DataFrame(compatibility_matrix)

def safe_get_nested(dictionary, keys, default=None):
    """Safely get value from nested dictionary"""
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

# ==================================================
# UTILITY FUNCTIONS FOR SAFE NUMERIC CONVERSIONS
# ==================================================
def safe_convert_to_float(value, default=0.0):
    """
    Robustly convert any value to float.
    Handles: float, int, string with %, string with +/-, None, np.nan
    """
    if value is None:
        return default
    
    # If already numeric type
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    
    # Handle numpy arrays/scalars
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except:
            return default
    
    # Handle string values
    if isinstance(value, str):
        # Remove common non-numeric characters
        cleaned = value.strip()
        cleaned = cleaned.replace('%', '').replace('+', '').replace(',', '')
        cleaned = cleaned.replace('₹', '').replace('$', '').replace('€', '').replace('£', '')
        
        # If empty after cleaning
        if cleaned == '' or cleaned == '—' or cleaned == 'N/A':
            return default
        
        try:
            return float(cleaned)
        except ValueError:
            # Try to extract first number from string
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', cleaned)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    return default
            return default
    
    # Last resort try direct conversion
    try:
        return float(value)
    except:
        return default

def safe_percentage_to_float(percentage_str, default=0.0):
    """
    Convert percentage string to float (e.g., '+1.7%' -> 0.017)
    """
    if percentage_str is None:
        return default
    
    # If already a number, assume it's already in decimal form
    if isinstance(percentage_str, (int, float, np.integer, np.floating)):
        return float(percentage_str)
    
    # Convert to string and clean
    if not isinstance(percentage_str, str):
        percentage_str = str(percentage_str)
    
    cleaned = percentage_str.strip()
    
    # Handle special cases
    if cleaned in ['', '—', 'N/A', 'None', 'null', 'NaN']:
        return default
    
    # Remove percentage sign and plus sign
    cleaned = cleaned.replace('%', '').replace('+', '')
    
    # Remove any other non-numeric except decimal and minus
    import re
    cleaned = re.sub(r'[^\d.-]', '', cleaned)
    
    try:
        value = float(cleaned)
        # Convert from percentage to decimal (1.7 -> 0.017)
        return value / 100.0
    except:
        return default

def sanitize_dataframe_for_display(df):
    """
    Ensure dataframe is 100% Arrow + Styler safe
    """
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: safe_convert_to_float(x, np.nan))
    return df


# ==================================================
# MAIN APP PAGE
# ==================================================
if page == "📊 Main App":
    st.title("📈 Investor NSE Stock Dashboard")

    query = st.sidebar.text_input(
        "Type stock name or symbol (Reliance, TCS, Infosys)",
        value="Reliance"
    )
    
    @st.cache_data(ttl=6000)
    def yahoo_search(q):
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": q, "quotesCount": 10, "newsCount": 0}
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=6)
            data = r.json()
            return [
                {
                    "symbol": i["symbol"],
                    "name": i.get("shortname", i["symbol"])
                }
                for i in data.get("quotes", [])
                if i.get("symbol", "").endswith(".NS")
            ]
        except:
            return []
    
    matches = yahoo_search(query)
    
    # ================= FIX: ADD STOCK SELECTION LOGIC =================
    if matches:
        selected_stock = st.sidebar.selectbox(
            "Select Stock",
            matches,
            format_func=lambda x: f"{x['name']} ({x['symbol']})"
        )
        yahoo_symbol = selected_stock["symbol"]
    else:
        st.sidebar.warning("No NSE stocks found. Using default: RELIANCE.NS")
        yahoo_symbol = "RELIANCE.NS"
    
    # Check if stock changed
    if st.session_state.current_stock_key != yahoo_symbol:
        # Clear old analysis when stock changes
        st.session_state.scoring_result = None
        st.session_state.forecast_result = None
        st.session_state.last_scored_symbol = None
        st.session_state.last_forecast_symbol = None
    
    # Set new stock key
    st.session_state.current_stock_key = yahoo_symbol
    
    # ==================================================
    # QUICK TIME RANGE (TradingView style)
    # ==================================================
    st.sidebar.subheader("⏱ Quick Time Range")
    
    time_range = st.sidebar.radio(
        "Select range",
        ["1D", "1W", "1M", "1Y", "5Y", "Max"],
        horizontal=True
    )
    
    
    # ==================================================
    # DATE RANGE
    # ==================================================
    start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", date.today())
    
    
    from datetime import timedelta
    
    end_date = date.today()
    
    if time_range == "1D":
        # For 1D, get data for the last 5 trading days to show last day's movement
        start_date = end_date - timedelta(days=7)  # Get 7 days to ensure we have data
    elif time_range == "1W":
        start_date = end_date - timedelta(weeks=1)
    elif time_range == "1M":
        start_date = end_date - timedelta(days=30)
    elif time_range == "1Y":
        start_date = end_date - timedelta(days=365)
    elif time_range == "5Y":
        start_date = end_date - timedelta(days=365 * 5)
    else:  # Max
        start_date = date(2000, 1, 1)
    
    
    # ==================================================
    # LOAD PRICE DATA
    # ==================================================
    @st.cache_data(ttl=30000)
    def load_full_price(symbol, start, end):
        """
        Enhanced full price loading with retry mechanism
        """
        for attempt in range(3):
            try:
                df = yf.download(symbol, start=start, end=end, progress=False, timeout=10)
                if df.empty:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    return None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.reset_index(inplace=True)
                return df
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return None
    
    # Load full historical data for scoring page
    full_df = load_full_price(yahoo_symbol, date(2000, 1, 1), date.today())
    if full_df is None:
        st.error("No price data available")
        st.stop()
    
    # Load filtered data for main app display
    df = load_full_price(yahoo_symbol, start_date, end_date)
    if df is None:
        st.error("No price data available")
        st.stop()
    
    # ==================================================
    # LOAD FUNDAMENTALS
    # ==================================================
    @st.cache_data(ttl=36000)
    def load_info(symbol):
        try:
            return yf.Ticker(symbol).info
        except:
            return {}
    
    info = load_info(yahoo_symbol)
    st.session_state["df"] = df
    st.session_state["info"] = info
    st.session_state["symbol"] = yahoo_symbol
    st.session_state["full_df"] = full_df  # Store full data for scoring page
    st.session_state["full_info"] = info   # Store full info for scoring page
    stock_name = info.get("longName", yahoo_symbol.replace(".NS", ""))
    st.session_state["stock_name"] = stock_name
    st.session_state.current_stock_key = yahoo_symbol
    st.markdown(f"## 📌 {stock_name}")

    # ==================================================
    # HELPERS
    # ==================================================
    def fmt_cr(val):
        return "—" if val in [None, ""] else f"₹ {val/1e7:,.0f} Cr"
    
    def fmt_pct(val):
        return "—" if val in [None, ""] else f"{val*100:.2f}%"
    
    def fmt(val):
        return "—" if val in [None, ""] else round(val, 2)
    
    # ==================================================
    # PRICE METRICS (GREEN / RED)
    # ==================================================
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    chg = latest["Close"] - prev["Close"]
    chg_pct = (chg / prev["Close"]) * 100
    color = "normal" if chg >= 0 else "inverse"
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Price (₹)", f"{latest['Close']:.2f}", f"{chg:.2f}", delta_color=color)
    c2.metric("Change %", f"{chg_pct:.2f}%", delta_color=color)
    c3.metric("Day High", f"{latest['High']:.2f}")
    c4.metric("Day Low", f"{latest['Low']:.2f}")
    
    # ==================================================
    # TECHNICAL INDICATORS
    # ==================================================
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    # ==================================================
    # PRICE CHART - SIMPLE WORKING VERSION
    # ==================================================
    st.subheader("📊 Price Chart")
    
    # Decide chart dataframe ONCE
    if time_range == "1D":
        chart_df = df.tail(5).copy()
        st.info("📊 1D View: Last trading day (with 5-day context)")
    else:
        chart_df = df.copy()
    
    # Safety checks
    chart_df = chart_df.drop_duplicates(subset=["Date"])
    chart_df = chart_df.sort_values("Date")
    
    # Moving averages on CHART DATA ONLY
    chart_df["MA20"] = chart_df["Close"].rolling(
        window=min(20, len(chart_df)),
        min_periods=1
    ).mean()
    
    chart_df["MA50"] = chart_df["Close"].rolling(
        window=min(50, len(chart_df)),
        min_periods=1
    ).mean()
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=chart_df["Date"],
        y=chart_df["Close"],
        mode="lines",
        name="Close",
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=chart_df["Date"],
        y=chart_df["MA20"],
        mode="lines",
        name="MA 20",
        line=dict(width=1.5, dash="dash")
    ))
    
    fig.add_trace(go.Scatter(
        x=chart_df["Date"],
        y=chart_df["MA50"],
        mode="lines",
        name="MA 50",
        line=dict(width=1.5)
    ))
    
    fig.update_layout(
        height=480,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    latest_close = chart_df["Close"].iloc[-1]
    ma20 = chart_df["MA20"].iloc[-1]
    ma50 = chart_df["MA50"].iloc[-1]
    
    st.markdown(
        generate_insight(
            "price_trend",
            {
                "price": latest_close,
                "ma20": ma20,
                "ma50": ma50
            }
        )
    )

    

    # ==================================================
    # SPECIAL 1D DETAILED VIEW
    # ==================================================
    if time_range == "1D":
        with st.expander("📋 Last Trading Day Details", expanded=False):
          
            if len(df) >= 2:
                last_day = df.iloc[-1]
                prev_day = df.iloc[-2]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    open_price = last_day.get('Open', last_day['Close'])
                    st.metric("Open", f"₹{open_price:.2f}")
                
                with col2:
                    high_price = last_day.get('High', last_day['Close'])
                    st.metric("High", f"₹{high_price:.2f}")
                
                with col3:
                    low_price = last_day.get('Low', last_day['Close'])
                    st.metric("Low", f"₹{low_price:.2f}")
                
                with col4:
                    close_price = last_day['Close']
                    st.metric("Close", f"₹{close_price:.2f}")
                
                # Calculate day's change
                day_change = close_price - open_price
                day_change_pct = (day_change / open_price) * 100 if open_price != 0 else 0
                
                st.metric(
                    "Day's Change", 
                    f"₹{day_change:.2f}", 
                    f"{day_change_pct:+.2f}%",
                    delta_color="normal" if day_change >= 0 else "inverse"
                )
                
                # Show OHLC data table
                with st.expander("📊 View OHLC Data"):
                    ohlc_data = pd.DataFrame({
                        "Metric": ["Open", "High", "Low", "Close", "Volume"],
                        "Value": [
                            f"₹{open_price:.2f}",
                            f"₹{high_price:.2f}",
                            f"₹{low_price:.2f}",
                            f"₹{close_price:.2f}",
                            f"{last_day.get('Volume', 0):,}"
                        ]
                    })
                    st.table(ohlc_data)


    # Add chart statistics
    if time_range != "1D":
        st.subheader("📈 Chart Statistics")
       
        col1, col2, col3, col4 = st.columns(4)
         
        with col1:
            price_change = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
            st.metric("Period Return", f"{price_change:.2f}%")
        
        with col2:
            days_count = len(df)
            st.metric("Trading Days", days_count)
        
        with col3:
            highest_price = df["Close"].max()
            st.metric("Highest Price", f"₹{highest_price:.2f}")
        
        with col4:
            lowest_price = df["Close"].min()
            st.metric("Lowest Price", f"₹{lowest_price:.2f}")
        
    # ==================================================
    # VOLUME
    # ==================================================
    st.subheader("📉 Volume")
    
    vol_fig = go.Figure()
    vol_fig.add_bar(x=df["Date"], y=df["Volume"])
    vol_fig.update_layout(height=300)
    st.plotly_chart(vol_fig, width="stretch")
    
    # ==================================================
    # RSI & MACD
    # ==================================================
    st.subheader("📐 Technical Indicators")

    t1, t2 = st.columns(2)
    
    # ================= RSI =================
    with t1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="orange", width=2)
        ))
    
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    
        fig_rsi.update_layout(
            title="RSI (14)",
            height=320,
            margin=dict(t=40, b=30),
            yaxis=dict(range=[0, 100]),
            template="plotly_white"
        )
    
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    # ================= MACD =================
    with t2:
        fig_macd = go.Figure()
    
        fig_macd.add_trace(go.Scatter(
            x=df["Date"],
            y=df["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="blue", width=2)
        ))
    
        fig_macd.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Signal"],
            mode="lines",
            name="Signal",
            line=dict(color="red", width=1.5)
        ))
    
        fig_macd.update_layout(
            title="MACD",
            height=320,
            margin=dict(t=40, b=30),
            template="plotly_white"
        )
    
        st.plotly_chart(fig_macd, use_container_width=True)

    
    # ==================================================
    # FUNDAMENTALS TABLE
    # ==================================================
    st.subheader("📌 Fundamentals")
    
    fund_df = pd.DataFrame([
        ("Market Cap", fmt_cr(info.get("marketCap"))),
        ("P/E Ratio", fmt(info.get("trailingPE"))),
        ("Price to Book", fmt(info.get("priceToBook"))),
        ("EPS (TTM)", fmt(info.get("trailingEps"))),
        ("ROE", fmt_pct(info.get("returnOnEquity"))),
        ("ROA", fmt_pct(info.get("returnOnAssets"))),
        ("Debt to Equity", fmt(info.get("debtToEquity"))),
        ("Dividend Yield", fmt_pct(info.get("dividendYield"))),
    ], columns=["Metric", "Value"])
    
    st.table(fund_df)
    
    # ==================================================
    # PEER COMPARISON & SECTOR AVERAGE
    # ==================================================
    st.subheader("🏭 Sector Comparison")
    
    sector = info.get("sector")
    st.write("**Sector:**", sector)
    
    @st.cache_data(ttl=36000)
    def get_sector_peers(symbol):
        try:
            peers = yf.Ticker(symbol).peers
            return peers[:5]
        except:
            return []
    
    peers = get_sector_peers(yahoo_symbol)
    
    peer_data = []
    for p in peers:
        i = load_info(p)
        peer_data.append({
            "Stock": p.replace(".NS", ""),
            "Market Cap (Cr)": i.get("marketCap", 0) / 1e7 if i.get("marketCap") else None,
            "P/E": i.get("trailingPE"),
            "ROE %": i.get("returnOnEquity") * 100 if i.get("returnOnEquity") else None
        })
    
    peer_df = pd.DataFrame(peer_data)
    
    if not peer_df.empty:
        st.dataframe(peer_df)
        st.write("**Sector Avg P/E:**", round(peer_df["P/E"].mean(), 2))
        st.write("**Sector Avg ROE:**", round(peer_df["ROE %"].mean(), 2))
    
    # ==================================================
    # COMPANY OVERVIEW
    # ==================================================
    st.subheader("🏢 Company Overview")
    
    st.write("**Company:**", info.get("longName"))
    st.write("**Industry:**", info.get("industry"))
    
    with st.expander("📄 Business Summary"):
        st.write(info.get("longBusinessSummary", "—"))
    
    # ==================================================
    # RAW HISTORICAL DATA (Yahoo Finance)
    # ==================================================
    st.subheader("📄 Raw Historical Data")
    
    with st.expander("Click to view full OHLCV data fetched via yfinance"):
        st.dataframe(df)


# ==================================================
# SCORING PAGE WITH VISUAL ENHANCEMENTS
# ==================================================
# ==================================================
# SCORING PAGE - REDESIGNED (PERFECT HIERARCHY)
# ==================================================
elif page == "🧠 Scoring":
    if "full_df" not in st.session_state or "full_info" not in st.session_state:
        st.warning("Please go to 📊 Main App first and select a stock.")
        st.stop()

    # Use full data for scoring page independent filtering
    full_df = st.session_state["full_df"]
    full_info = st.session_state["full_info"]
    stock_name = st.session_state.get("stock_name", "Selected Stock")
    
    # Add custom CSS for consistent spacing
    st.markdown("""

    """, unsafe_allow_html=True)

    # FIXED HEADER
    st.markdown(f"## 🧠 Stock Health Score — {stock_name}")
    
    # ===================== MAIN LAYOUT =====================
    # Two column layout: Left = Calm Zone, Right = Interactive
    left_col, right_col = st.columns([2, 1], gap="large")
    
    with right_col:
        # ========== RIGHT COLUMN - ALL FILTERS & CONTROLS ==========
        st.markdown("### ⚙️ Configuration")
        
        with st.expander("📅 Date Range", expanded=False):
            min_date = full_df["Date"].min().date()
            max_date = full_df["Date"].max().date()
            
            scoring_start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
            
            scoring_end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with st.expander("🎚️ Metric Configuration", expanded=False):
            # Default metric table
            metric_df = pd.DataFrame({
                "Metric": [
                    "P/E Ratio", "Price to Book", "ROE", "Operating Margin",
                    "Revenue Growth", "Earnings Growth",
                    "Debt to Equity", "Current Ratio",
                    "Period Return", "Avg RSI", "Volatility", "Trend Strength"
                ],
                "Include": [True] * 12,
                "Weight (%)": [8.33] * 12
            })
            
            edited_metrics = st.data_editor(
                metric_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include", default=True),
                    "Weight (%)": st.column_config.NumberColumn(
                        "Weight",
                        min_value=0,
                        max_value=100,
                        step=0.5
                    )
                }
            )
        
        with st.expander("🧪 Advanced Filters", expanded=False):
            rsi_threshold = st.slider("RSI Threshold", 30, 70, 50, 5)
            volatility_filter = st.checkbox("Filter High Volatility Days", value=False)
            volume_filter = st.checkbox("Above Average Volume Days Only", value=False)
            price_filter = st.checkbox("Exclude Extreme Price Movements", value=False)
        
        # Apply Changes button
        if st.button("🔄 Apply Configuration", type="primary", use_container_width=True):
            st.session_state.scoring_result = None  # Force recalculation
            st.rerun()
    
    # ========== LEFT COLUMN - SCORE SUMMARY ==========
    with left_col:
        # ================= ROW 1: SCORE SUMMARY =================
        # First check if we have scoring results or need to calculate
        current_symbol = st.session_state.current_stock_key
        need_scoring_calculation = (
            st.session_state.last_scored_symbol != current_symbol
            or st.session_state.scoring_result is None
        )
        
        # Filter data based on selected range
        filtered_df = full_df[
            (full_df["Date"] >= pd.to_datetime(scoring_start_date)) & 
            (full_df["Date"] <= pd.to_datetime(scoring_end_date))
        ].copy()
        
        if filtered_df.empty:
            st.error("No data available for the selected date range.")
            st.stop()
        
        # Calculate or load scoring results
        if need_scoring_calculation:
            with st.spinner("🔄 Calculating stock health score..."):
                scoring_result = calculate_stock_score(
                    full_info,
                    filtered_df,
                    start_date=scoring_start_date,
                    end_date=scoring_end_date,
                    metric_table=edited_metrics
                )
                st.session_state.scoring_result = scoring_result
                st.session_state.last_scored_symbol = current_symbol
        else:
            scoring_result = st.session_state.scoring_result
        
        # Extract data from scoring result
        total_score = scoring_result.get('total_score', 0)
        regime_adjusted_score = scoring_result.get('regime_adjusted_score', total_score)
        regime_delta = scoring_result.get('regime_delta', 0)
        market_regime = scoring_result.get('market_regime', 'Unknown')
        confidence_level = scoring_result.get('confidence_level', 'Low')
        upside_probability = scoring_result.get('upside_probability', 0.5)
        
        # Hero Score Box
        hero_col1, hero_col2 = st.columns([1, 2])
        with hero_col1:
            # Main score with delta
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; height: 240px; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">Regime Adjusted Score</div>
                <div style="font-size: 48px; font-weight: bold;">{regime_adjusted_score:.0f}</div>
                <div style="font-size: 18px; color: {'#4ade80' if regime_delta >= 0 else '#f87171'}">
                    {regime_delta:+.0f}
                </div>
                <div style="font-size: 16px; margin-top: 10px;">{market_regime}</div>
                <div style="font-size: 14px; margin-top: 5px; opacity: 0.8;">
                    Score Stability: {scoring_result.get('stability_level', 'Medium')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with hero_col2:
            # ========== ROW 1: 3 SYMMETRIC BOXES ==========
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="border: 2px solid #3b82f6; 
                            border-radius: 12px; 
                            padding: 20px 15px; 
                            text-align: center; 
                            background: rgba(59, 130, 246, 0.08);
                            height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;">
                    <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">BASE SCORE</div>
                    <div style="font-size: 36px; font-weight: bold; color: #1e40af; line-height: 1;">{total_score:.0f}</div>
                    <div style="font-size: 12px; color: #9ca3af; margin-top: 10px;">Out of 100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Color coding for regime
                regime_color = {
                    'Bull': '#10b981', 
                    'Bear': '#ef4444', 
                    'Sideways': '#f59e0b'
                }.get(market_regime.split()[0] if ' ' in market_regime else market_regime, '#6b7280')
                
                st.markdown(f"""
                <div style="border: 2px solid {regime_color}; 
                            border-radius: 12px; 
                            padding: 20px 15px; 
                            text-align: center; 
                            background: rgba({'16, 185, 129' if 'Bull' in market_regime else '239, 68, 68' if 'Bear' in market_regime else '245, 158, 11'}, 0.08);
                            height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;">
                    <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">MARKET REGIME</div>
                    <div style="font-size: 24px; font-weight: bold; color: {regime_color}; line-height: 1.2;">{market_regime.split()[0] if ' ' in market_regime else market_regime}</div>
                    <div style="font-size: 14px; color: {regime_color}; margin-top: 8px;">{market_regime.split()[1] if ' ' in market_regime else ''}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Color code based on confidence
                conf_color = "#ef4444" if confidence_level == "Low" else "#f59e0b" if confidence_level == "Medium" else "#10b981"
                conf_bg = "239, 68, 68" if confidence_level == "Low" else "245, 158, 11" if confidence_level == "Medium" else "16, 185, 129"
                
                st.markdown(f"""
                <div style="border: 2px solid {conf_color}; 
                            border-radius: 12px; 
                            padding: 20px 15px; 
                            text-align: center; 
                            background: rgba({conf_bg}, 0.08);
                            height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;">
                    <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">AI CONFIDENCE</div>
                    <div style="font-size: 24px; font-weight: bold; color: {conf_color}; line-height: 1.2;">{confidence_level}</div>
                    <div style="font-size: 12px; color: #9ca3af; margin-top: 10px;">Model Certainty</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ========== ROW 2: 3 SYMMETRIC BOXES ==========
            col4, col5, col6 = st.columns(3)
            
            with col4:
                market_score = scoring_result.get('market_fit_score', total_score)
                st.markdown(f"""
                <div style="border: 2px solid #8b5cf6; 
                            border-radius: 12px; 
                            padding: 20px 15px; 
                            text-align: center; 
                            background: rgba(139, 92, 246, 0.08);
                            height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;">
                    <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">MARKET SCORE</div>
                    <div style="font-size: 36px; font-weight: bold; color: #5b21b6; line-height: 1;">{market_score:.0f}</div>
                    <div style="font-size: 12px; color: #9ca3af; margin-top: 10px;">Regime Adjusted</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                structure_score = scoring_result.get('structural_score', total_score)
                st.markdown(f"""
                <div style="border: 2px solid #ec4899; 
                            border-radius: 12px; 
                            padding: 20px 15px; 
                            text-align: center; 
                            background: rgba(236, 72, 153, 0.08);
                            height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;">
                    <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">STRUCTURE SCORE</div>
                    <div style="font-size: 36px; font-weight: bold; color: #9d174d; line-height: 1;">{structure_score:.0f}</div>
                    <div style="font-size: 12px; color: #9ca3af; margin-top: 10px;">Fundamental Health</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                sector = scoring_result.get('adjusted_sector', 'Unknown')
                st.markdown(f"""
                <div style="border: 2px solid #f59e0b; 
                            border-radius: 12px; 
                            padding: 20px 15px; 
                            text-align: center; 
                            background: rgba(245, 158, 11, 0.08);
                            height: 120px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;">
                    <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px; font-weight: 500;">SECTOR</div>
                    <div style="font-size: 24px; font-weight: bold; color: #92400e; line-height: 1.2;">{sector}</div>
                    <div style="font-size: 14px; color: #f59e0b; margin-top: 8px;">Low Volatility</div>
                </div>
                """, unsafe_allow_html=True)
                       
            st.markdown("---")
        
        # ================= ROW 2: BUSINESS MODEL + STRENGTH BAR =================
        row2_col1, row2_col2 = st.columns([1, 1])
        
        with row2_col1:
        # Business Model Description - RED CARD
            business_desc = full_info.get('longBusinessSummary', '')
            if business_desc:
                # Extract first 250 chars for better readability
                short_desc = business_desc[:250] + "..." if len(business_desc) > 250 else business_desc
                st.markdown("### 🏢 Business Model")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); 
                            padding: 20px; 
                            border-radius: 12px; 
                            border: 1px solid #991b1b;
                            color: white;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">
                        {short_desc}
                    </div>
                    <div style="font-size: 12px; opacity: 0.7; margin-top: 10px;">
                        Source: Company Business Summary
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### 🏢 Business Model")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); 
                            padding: 20px; 
                            border-radius: 12px; 
                            border: 1px solid #991b1b;
                            color: white;
                            text-align: center;">
                    No business description available
                </div>
                """, unsafe_allow_html=True)
        
        with row2_col2:
            # Business Strength Bar
            strength_score = scoring_result.get('structural_score', total_score)
            strength_percent = strength_score
            
            st.markdown("### 📊 Business Strength")
            st.markdown(f"""
            <div style="background: #1e293b; padding: 15px; border-radius: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: white;">Overall Strength</span>
                    <span style="color: white; font-weight: bold;">{strength_percent:.0f}%</span>
                </div>
                <div style="background: #334155; height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #3b82f6, #60a5fa); 
                                width: {strength_percent}%; height: 100%;">
                    </div>
                </div>
                <div style="margin-top: 10px; color: #94a3b8; font-size: 12px;">
                    Based on fundamental analysis
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        
        # ================= CREATE ANALYSIS POINTS =================
        def safe_float(value, default=None):
            if value is None:
                return default
            try:
                return float(value)
            except:
                return default
        
        # Get values
        pe = safe_float(full_info.get('trailingPE'))
        rev_growth = safe_float(full_info.get('revenueGrowth'))
        roe = safe_float(full_info.get('returnOnEquity'))
        debt_ratio = scoring_result.get('business_structure', {}).get('debt_ratio')
        if debt_ratio is None:
            try:
                debt_ratio = get_debt_ratio(full_info)
            except:
                debt_ratio = None
        
        trend_analysis = scoring_result.get('trend_analysis', {})
        direction = trend_analysis.get('direction', 'unknown')
        
        # Create analysis_points list
        analysis_points = []
        
        # Overall quality
        if total_score >= 75:
            analysis_points.append("• **Overall Quality:** Strong")
        elif total_score >= 60:
            analysis_points.append("• **Overall Quality:** Good")
        else:
            analysis_points.append("• **Overall Quality:** Needs Improvement")
        
        # Valuation (with safety check)
        if pe is not None:
            if pe < 15:
                analysis_points.append(f"• **Valuation:** Attractive (P/E: {pe:.1f}x)")
            elif pe < 25:
                analysis_points.append(f"• **Valuation:** Reasonable (P/E: {pe:.1f}x)")
            else:
                analysis_points.append(f"• **Valuation:** Expensive (P/E: {pe:.1f}x)")
        else:
            analysis_points.append("• **Valuation:** Data not available")
        
        # Growth (with safety check)
        if rev_growth is not None:
            if rev_growth > 0.15:
                analysis_points.append(f"• **Growth:** Strong ({rev_growth:.1%})")
            elif rev_growth > 0.05:
                analysis_points.append(f"• **Growth:** Moderate ({rev_growth:.1%})")
            else:
                analysis_points.append(f"• **Growth:** Weak ({rev_growth:.1%})")
        else:
            analysis_points.append("• **Growth:** Data not available")
        
        # Profitability (with safety check)
        if roe is not None:
            if roe > 0.15:
                analysis_points.append(f"• **Profitability:** High (ROE: {roe:.1%})")
            elif roe > 0.08:
                analysis_points.append(f"• **Profitability:** Moderate (ROE: {roe:.1%})")
            else:
                analysis_points.append(f"• **Profitability:** Low (ROE: {roe:.1%})")
        else:
            analysis_points.append("• **Profitability:** Data not available")
        
        # Financial Health
        if debt_ratio is not None:
            if debt_ratio < 1.0:
                analysis_points.append(f"• **Financial Health:** Strong (D/E: {debt_ratio:.1f}x)")
            elif debt_ratio < 2.0:
                analysis_points.append(f"• **Financial Health:** Moderate (D/E: {debt_ratio:.1f}x)")
            else:
                analysis_points.append(f"• **Financial Health:** Weak (D/E: {debt_ratio:.1f}x)")
        else:
            analysis_points.append("• **Financial Health:** Data not available")
        
        # Momentum
        if direction == 'uptrend':
            analysis_points.append("• **Momentum:** Positive")
        else:
            analysis_points.append("• **Momentum:** Neutral/Negative")
        
        # Regime Impact
        if regime_delta > 0:
            analysis_points.append(f"• **Regime Impact:** Positive (+{regime_delta:.0f})")
        elif regime_delta < 0:
            analysis_points.append(f"• **Regime Impact:** Negative ({regime_delta:.0f})")
        else:
            analysis_points.append("• **Regime Impact:** Neutral")
        
        # ================= ROW 4: DETAILED ANALYSIS + INTELLIGENCE CARDS SIDE-BY-SIDE =================
        st.markdown("---")
        analysis_col, intel_col = st.columns([1, 1], gap="large")
        # ================= ROW 4: DETAILED ANALYSIS + INTELLIGENCE CARDS SIDE-BY-SIDE =================
        
        
        with analysis_col:
            st.markdown("### 📘 Detailed Analysis")
            
            # Your existing analysis points - just show them
            for point in analysis_points:
                st.markdown(point)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with intel_col:
            st.markdown("### 🧠 Intelligence Cards")
            
            # RED FLAGS Card (Expanded with details)
            red_flags = scoring_result.get('red_flags', [])
            flag_count = len(red_flags)
            
            if flag_count > 0:
                flag_details = "\n".join([f"• {flag}" for flag in red_flags[:3]])  # Show first 3 flags
                if flag_count > 3:
                    flag_details += f"\n• ...and {flag_count - 3} more"
            else:
                flag_details = "• No critical issues detected"
            
            st.markdown(f"""
            <div style="background: #fee2e2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626; margin-bottom: 15px;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 20px; margin-right: 10px;">⚠️</div>
                    <div style="font-weight: bold; color: #991b1b;">Red Flags ({flag_count})</div>
                </div>
                <div style="font-size: 14px; color: #7f1d1d;">
                    {flag_details}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # TREND ANALYSIS Card (Expanded)
            trend = scoring_result.get('trend_analysis', {})
            direction = trend.get('direction', 'unknown')
            strength = trend.get('strength_label', 'unknown')
            duration = trend.get('duration_days', 'N/A')
            
            trend_icon = "📈" if direction == 'uptrend' else "📉" if direction == 'downtrend' else "➡️"
            
            st.markdown(f"""
            <div style="background: #dbeafe; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 15px;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 20px; margin-right: 10px;">{trend_icon}</div>
                    <div style="font-weight: bold; color: #1e40af;">Trend Analysis</div>
                </div>
                <div style="font-size: 14px; color: #1e3a8a;">
                    • Direction: {direction.title()}<br>
                    • Strength: {strength}<br>
                    • Duration: {duration} days
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ML HISTORICAL INSIGHT Card (Expanded)
            upside_pct = upside_probability * 100
            confidence = confidence_level
            
            st.markdown(f"""
            <div style="background: #ede9fe; padding: 15px; border-radius: 8px; border-left: 4px solid #8b5cf6; margin-bottom: 15px;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 20px; margin-right: 10px;">🤖</div>
                    <div style="font-weight: bold; color: #5b21b6;">ML Historical Insight</div>
                </div>
                <div style="font-size: 14px; color: #4c1d95;">
                    • Upside Probability: {upside_pct:.0f}%<br>
                    • Confidence: {confidence}<br>
                    • Historical Context: Based on {len(full_df) if full_df is not None else 0} trading days
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # THREE-LAYER ASSESSMENT Card (Expanded)
            structural = scoring_result.get('structural_score', 0)
            market_fit = scoring_result.get('market_fit_score', 0)
            opportunity = scoring_result.get('opportunity_score', 0)
            
            st.markdown(f"""
            <div style="background: #dcfce7; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 20px; margin-right: 10px;">🎯</div>
                    <div style="font-weight: bold; color: #047857;">Three-Layer Assessment</div>
                </div>
                <div style="font-size: 14px; color: #065f46;">
                    • Structural: {structural:.0f}/100<br>
                    • Market Fit: {market_fit:.0f}/100<br>
                    • Opportunity: {opportunity:.0f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # ================= ROW 5: PILLAR CONFIGURATION =================
        with st.expander("⚖️ Pillar Weight Configuration", expanded=False):
            pillars = scoring_result.get('pillars', {})
            pillar_names = list(pillars.keys())
            
            # Create editable weight table
            weight_data = []
            for pillar in pillar_names:
                current_score = pillars[pillar].get('score', 10)
                weight_data.append({
                    'Pillar': pillar,
                    'Current Score': f"{current_score:.1f}/20",
                    'Weight (%)': 20.0  # Equal weight by default
                })
            
            weight_df = pd.DataFrame(weight_data)
            edited_weights = st.data_editor(
                weight_df,
                hide_index=True,
                column_config={
                    'Weight (%)': st.column_config.NumberColumn(
                        min_value=0,
                        max_value=100,
                        step=1.0
                    )
                }
            )
            
            # Visualize weights
            fig_weights = go.Figure(data=[go.Pie(
                labels=edited_weights['Pillar'],
                values=edited_weights['Weight (%)'],
                hole=0.4
            )])
            fig_weights.update_layout(height=300)
            st.plotly_chart(fig_weights, use_container_width=True)
        
        # ================= ROW 6: TABLES & CHARTS =================
        with st.expander("📊 Pillar Performance Chart", expanded=False):
            # Pillar bar chart
            pillar_names = list(pillars.keys())
            pillar_scores = [pillars[p].get('score', 0) for p in pillar_names]
            
            fig_pillars = go.Figure(data=[
                go.Bar(
                    x=pillar_names,
                    y=pillar_scores,
                    marker_color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
                )
            ])
            fig_pillars.update_layout(
                title="Pillar Scores",
                yaxis_title="Score (out of 20)",
                yaxis_range=[0, 20],
                height=300
            )
            st.plotly_chart(fig_pillars, use_container_width=True)
        
        with st.expander("📋 Detailed Pillar Table", expanded=False):
            # Detailed pillar data
            detailed_data = []
            for pillar_name, pillar_data in pillars.items():
                score = pillar_data.get('score', 0)
                reasons = pillar_data.get('reasons', [])
                key_reason = reasons[0] if reasons else "No analysis"
                
                detailed_data.append({
                    'Pillar': pillar_name,
                    'Score': f"{score:.1f}/20",
                    'Key Insight': key_reason[:100] + "..." if len(key_reason) > 100 else key_reason
                })
            
            st.dataframe(pd.DataFrame(detailed_data), use_container_width=True)
        
        with st.expander("🧾 Metric-Level Breakdown", expanded=False):
            # All metrics breakdown
            all_metrics = []
            for pillar_name, pillar_data in pillars.items():
                metrics = pillar_data.get('metrics', [])
                for metric in metrics:
                    all_metrics.append({
                        'Metric': metric.get('name', 'Unknown'),
                        'Pillar': pillar_name,
                        'Value': metric.get('formatted', 'N/A'),
                        'Impact': f"{metric.get('score_impact', 0):+.1f}"
                    })
            
            if all_metrics:
                st.dataframe(pd.DataFrame(all_metrics), use_container_width=True)
            else:
                st.info("No metric data available")
        
        # ================= ROW 7: EXPORT & ACTIONS =================
        st.markdown("---")
        st.markdown("### 📤 Export & Actions")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📥 Download Summary", use_container_width=True):
                # Create summary CSV
                summary_data = {
                    'Stock': stock_name,
                    'Total Score': total_score,
                    'Regime Adjusted Score': regime_adjusted_score,
                    'Market Regime': market_regime,
                    'AI Confidence': confidence_level,
                    'Upside Probability': f"{upside_probability:.1%}",
                    'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d')
                }
                summary_df = pd.DataFrame([summary_data])
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{stock_name}_score_summary.csv",
                    mime="text/csv"
                )
        
        with col_export2:
            if st.button("📊 Download Full CSV", use_container_width=True):
                # Create comprehensive CSV
                all_data = []
                for pillar_name, pillar_data in pillars.items():
                    all_data.append({
                        'Type': 'Pillar',
                        'Name': pillar_name,
                        'Score': pillar_data.get('score', 0),
                        'Details': ' | '.join(pillar_data.get('reasons', [])[:2])
                    })
                
                results_df = pd.DataFrame(all_data)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{stock_name}_full_analysis.csv",
                    mime="text/csv"
                )
        
        with col_export3:
            if st.button("🐛 Debug Info", use_container_width=True):
                with st.expander("Debug Data", expanded=True):
                    st.json(scoring_result)
        
        # ================= FOOTER NOTES =================
        st.markdown("---")
        st.caption("""
        *Note: Scores are based on fundamental analysis using the selected metrics and weights. 
        Past performance is not indicative of future results. Always conduct your own research.*
        """)

# ==================================================
# FORECASTING PAGE - COMPLETE WITH ALL ENHANCEMENTS
# ==================================================
elif page == "🔮 Forecasting":
    
    if "full_df" not in st.session_state or "full_info" not in st.session_state:
        st.warning("Please go to 📊 Main App first and select a stock.")
        st.stop()
    
    full_df = st.session_state["full_df"]
    full_info = st.session_state["full_info"]
    stock_name = st.session_state.get("stock_name", "Selected Stock")
    symbol = st.session_state.get("symbol", "")
    
    # ================= MAIN LAYOUT =================
    left_col, right_col = st.columns([2, 1], gap="large")
    
    with right_col:
        # ========== RIGHT COLUMN - CONFIGURATION ==========
        st.markdown("### ⚙️ Configuration")
        
        model_system = st.selectbox(
            "🔧 Model System",
            [
                "🧠 Intelligent 4-Model Ensemble", 
                "📊 Regime & Distribution Analysis",
                "📈 ARIMA Short-Term Focus",
                "🎲 Monte Carlo Risk Analysis",
                "🔍 Compare All Models"
            ],
            help="Choose the forecasting approach"
        )
        
        forecast_horizon = st.selectbox(
            "📅 Forecast Horizon",
            ["20D (1 Month)", "60D (3 Months)", "120D (6 Months)", "252D (1 Year)", "Custom"],
            index=3
        )
        
        if forecast_horizon == "Custom":
            forecast_days = st.slider("Custom Days", 10, 500, 252)
        else:
            forecast_days = int(forecast_horizon.split('D')[0])
        
        horizon_key = f"{forecast_days}D"
        
        distribution_display = st.selectbox(
            "📈 Distribution Focus",
            ["Fat Tail (Most Realistic)", "Compare All", "Normal", "Student T"],
            help="Select which distribution to focus on"
        )
        
        # ========== RUN FORECASTING BUTTON ==========
        current_symbol = st.session_state.current_stock_key
        
        if st.button("🚀 Run Forecasting", type="primary", use_container_width=True):
            with st.spinner("Running 4-model intelligent analysis..."):
                try:
                    engine = CompatibleForecastSystem(full_df)
                    forecast_results = engine.get_intelligent_forecast()
                    
                    st.session_state.forecast_result = forecast_results
                    st.session_state.forecast_engine = engine
                    st.session_state.last_forecast_symbol = current_symbol
                    st.session_state.last_horizon_key = horizon_key
                    
                    st.success("✅ Forecasting completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error running forecast: {str(e)}")
                    import traceback
                    with st.expander("Technical Details"):
                        st.code(traceback.format_exc())
        
        # Show status if already calculated
        if st.session_state.last_forecast_symbol == current_symbol and st.session_state.forecast_result is not None:
            st.info("📊 Forecast already calculated. Click button to refresh.")
    
    with left_col:
        # ========== LEFT COLUMN - FORECAST RESULTS ==========
        st.markdown(f"## Forecasting For: **{stock_name}**")
        
        # ================= CHECK IF WE HAVE FORECAST RESULTS =================
        if st.session_state.forecast_result is None:
            st.info("""
            **📋 How to use the Intelligent Forecasting System:**
            
            1. **Select Forecasting System**: Choose between different model approaches
            2. **Set Horizon**: Define how far ahead to forecast (20D to 252D)
            3. **Configure Distribution**: Select return distribution model focus
            4. **Adjust Advanced Options**: Tune display preferences as needed
            5. **Click 'Run Intelligent Forecasting'**: Generate comprehensive forecast
            """)
            
            current_price = float(full_df['Close'].iloc[-1])
            st.markdown(f"""
            ### 📊 Current Market Data
            **Current Price:** ₹{current_price:.2f}
            
            **Available Data:** {len(full_df)} trading days
            """)
            st.stop()
        
        # ================= GET FORECAST DATA =================
        forecast_results = st.session_state.forecast_result
        current_price = forecast_results['current_price']
        regime_timeline = forecast_results['regime_timeline']
        raw_models = forecast_results['raw_models']
        
        # Handle horizon selection
        available_horizons = ['20D', '60D', '120D', '252D']
        if horizon_key not in forecast_results['ensemble']:
            requested_days = forecast_days
            closest_horizon = min(available_horizons, 
                                 key=lambda x: abs(int(x.replace('D', '')) - requested_days))
            horizon_key = closest_horizon
            forecast_days = int(horizon_key.replace('D', ''))
            st.info(f"ℹ️ Forecasts available for 20D, 60D, 120D, 252D only. Showing closest match: {horizon_key}")
        
        # Define ensemble_forecast
        ensemble_forecast = forecast_results['ensemble'][horizon_key]
        
        # ================= 1. CURRENT MARKET CONTEXT =================
        st.markdown("### 🧭 Current Market Context")
        
        # Calculate values
        daily_change = 0
        if len(full_df) > 1:
            prev_close = float(full_df['Close'].iloc[-2])
            daily_change = ((current_price - prev_close) / prev_close) * 100
        
        # Get values safely (handle missing keys)
        current_regime = forecast_results.get('current_regime', 'Unknown')
        if not current_regime:
            current_regime = regime_timeline.get('current_regime', 'Unknown')
            
        current_confidence = regime_timeline.get('current_confidence', 0.5)
        signal_strength = regime_timeline.get('current_signal_strength', 0.5)
        
        # Create a SIMPLE display WITHOUT HTML - use Streamlit components
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Current Price**")
            st.markdown(f"### ₹{current_price:.2f}")
            st.markdown(f"""
            <div style="color: {'#10b981' if daily_change >= 0 else '#ef4444'}; font-size: 14px;">
                {'▲' if daily_change >= 0 else '▼'} {abs(daily_change):.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Market Regime**")
            regime_color = '#10b981' if 'Bull' in current_regime else '#f59e0b' if 'Sideways' in current_regime else '#ef4444'
            st.markdown(f"""
            <div style="color: {regime_color}; font-size: 18px; font-weight: bold;">
                {current_regime}
            </div>
            <div style="color: #9ca3af; font-size: 12px;">
                {current_confidence:.1%} confidence
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("**Signal Strength**")
            signal_color = '#10b981' if signal_strength > 0.7 else '#f59e0b' if signal_strength > 0.4 else '#ef4444'
            signal_label = 'Strong' if signal_strength > 0.7 else 'Moderate' if signal_strength > 0.4 else 'Weak'
            st.markdown(f"""
            <div style="color: {signal_color}; font-size: 18px; font-weight: bold;">
                {signal_label}
            </div>
            <div style="color: #9ca3af; font-size: 12px;">
                Regime clarity
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("**Forecast Horizon**")
            st.markdown(f"""
            <div style="color: #3b82f6; font-size: 18px; font-weight: bold;">
                {forecast_days}D
            </div>
            <div style="color: #9ca3af; font-size: 12px;">
                {horizon_key}
            </div>
            """, unsafe_allow_html=True)
        
        # Add a separator
        st.markdown("---")
        
        # ================= 2. PRIMARY FORECAST RESULTS =================
        st.markdown("### 🎯 Primary Forecast Results")
        
        # Get forecast data
        expected_price = ensemble_forecast['expected_price']
        expected_return = ensemble_forecast['expected_return'] * 100
        confidence_level = ensemble_forecast['confidence_level']
        confidence_low = ensemble_forecast['confidence_low']
        confidence_high = ensemble_forecast['confidence_high']
        
        # Positive Probability Calculation Fallback
        try:
            if (horizon_key in raw_models['monte_carlo'] and 
                'fat_tail' in raw_models['monte_carlo'][horizon_key]):
                positive_prob = (
                    raw_models['monte_carlo'][horizon_key]['fat_tail']['positive_probability']
                    * 100
                )
            else:
                if horizon_key in raw_models['arima']:
                    positive_prob = (
                        raw_models['arima'][horizon_key]['positive_probability']
                        * 100
                    )
                else:
                    positive_prob = 50.0
        except (KeyError, TypeError):
            positive_prob = 50.0
        
        # Display primary forecast in 2 columns
        primary_col1, primary_col2 = st.columns(2)
        
        with primary_col1:
            # Primary forecast box
            confidence_color = {
                'High': '#10b981',
                'Medium': '#f59e0b',
                'Low': '#ef4444'
            }.get(confidence_level, '#6b7280')
            
            html_primary = f"""
            <div style="border: 2px solid {confidence_color}; 
                        border-radius: 12px; 
                        padding: 20px; 
                        text-align: center; 
                        background: linear-gradient(135deg, {confidence_color} 0%, {confidence_color}80 100%);
                        color: white;
                        margin-bottom: 15px;
                        height: 320px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">
                    PRIMARY FORECAST ({forecast_days}D)
                </div>
                <div style="font-size: 36px; font-weight: bold; margin-bottom: 5px;">
                    {expected_return:+.1f}%
                </div>
                <div style="font-size: 20px; margin-bottom: 15px;">
                    Target: ₹{expected_price:.2f}
                </div>
                <div style="font-size: 14px; opacity: 0.9;">
                    Confidence: {confidence_level}
                </div>
                <div style="font-size: 12px; opacity: 0.8; margin-top: 10px;">
                    Upside: {positive_prob:.1f}%
                </div>
            </div>
            """
            st.markdown(html_primary, unsafe_allow_html=True)
        
        with primary_col2:
            # Other horizons preview
            st.markdown("""
            <div style="font-size: 13px; color: #6b7280; margin-bottom: 10px;">
            </div>
            """, unsafe_allow_html=True)
            
            # Get other horizon data
            display_horizons = ['20D', '120D', '252D']
            for h_key in display_horizons:
                if h_key in forecast_results['ensemble']:
                    horizon_data = forecast_results['ensemble'][h_key]
                    horizon_return = horizon_data['expected_return'] * 100
                    horizon_price = horizon_data['expected_price']
                    horizon_conf = horizon_data['confidence_level']
                    
                    conf_color = '#10b981' if horizon_conf == 'High' else '#f59e0b' if horizon_conf == 'Medium' else '#ef4444'
                    
                    html_horizon = f"""
                    <div style="border: 1px solid {conf_color}; 
                                border-radius: 8px; 
                                padding: 10px; 
                                margin-bottom: 8px;
                                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-size: 12px; color: #6b7280;">{h_key}</div>
                            <div style="font-size: 16px; font-weight: bold; color: {'#10b981' if horizon_return >= 0 else '#ef4444'};">
                                {horizon_return:+.1f}%
                            </div>
                        </div>
                        <div style="font-size: 12px; color: #6b7280; margin-top: 5px;">₹{horizon_price:.2f}</div>
                        <div style="font-size: 10px; color: {conf_color}; margin-top: 2px;">
                            {horizon_conf} confidence
                        </div>
                    </div>
                    """
                    st.markdown(html_horizon, unsafe_allow_html=True)
        st.markdown("---")
        
        # ================= 3. ACTIONABLE INSIGHTS =================
        st.markdown("### 💡 Actionable Insights")
        
        # Get insights
        insights = ensemble_forecast.get('insights', [])
        if insights:
            # Display insights in cards
            for insight in insights[:4]:  # Show top 4 insights
                if "⚠️" in insight or "High risk" in insight or "Severe" in insight or "Extreme" in insight:
                    html_insight = f"""
                    <div style="border-left: 4px solid #ef4444; 
                                background-color: #fef2f2; 
                                padding: 12px 15px; 
                                border-radius: 8px;
                                margin-bottom: 10px;">
                        <div style="font-size: 14px; color: #7f1d1d;">
                            {insight}
                        </div>
                    </div>
                    """
                    st.markdown(html_insight, unsafe_allow_html=True)
                elif "📈" in insight or "Strong" in insight or "Favorable" in insight:
                    html_insight = f"""
                    <div style="border-left: 4px solid #10b981; 
                                background-color: #f0fdf4; 
                                padding: 12px 15px; 
                                border-radius: 8px;
                                margin-bottom: 10px;">
                        <div style="font-size: 14px; color: #14532d;">
                            {insight}
                        </div>
                    </div>
                    """
                    st.markdown(html_insight, unsafe_allow_html=True)
                elif "🛡️" in insight or "Defensive" in insight or "Caution" in insight:
                    html_insight = f"""
                    <div style="border-left: 4px solid #f59e0b; 
                                background-color: #fffbeb; 
                                padding: 12px 15px; 
                                border-radius: 8px;
                                margin-bottom: 10px;">
                        <div style="font-size: 14px; color: #92400e;">
                            {insight}
                        </div>
                    </div>
                    """
                    st.markdown(html_insight, unsafe_allow_html=True)
                else:
                    html_insight = f"""
                    <div style="border-left: 4px solid #3b82f6; 
                                background-color: #eff6ff; 
                                padding: 12px 15px; 
                                border-radius: 8px;
                                margin-bottom: 10px;">
                        <div style="font-size: 14px; color: #1e40af;">
                            {insight}
                        </div>
                    </div>
                    """
                    st.markdown(html_insight, unsafe_allow_html=True)
        else:
            # Generate basic insights based on data
            if expected_return > 10:
                st.markdown("""
                <div style="border-left: 4px solid #10b981; 
                            background-color: #f0fdf4; 
                            padding: 12px 15px; 
                            border-radius: 8px;">
                    <div style="font-size: 14px; color: #14532d;">
                        🚀 <strong>Strong Upside Potential</strong>: Forecast indicates significant growth potential
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif expected_return < -5:
                st.markdown("""
                <div style="border-left: 4px solid #ef4444; 
                            background-color: #fef2f2; 
                            padding: 12px 15px; 
                            border-radius: 8px;">
                    <div style="font-size: 14px; color: #7f1d1d;">
                        ⚠️ <strong>Downside Risk</strong>: Forecast indicates potential decline
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if 'Bull' in current_regime and expected_return > 0:
                st.markdown("""
                <div style="border-left: 4px solid #3b82f6; 
                            background-color: #eff6ff; 
                            padding: 12px 15px; 
                            border-radius: 8px;">
                    <div style="font-size: 14px; color: #1e40af;">
                        📈 <strong>Bullish Alignment</strong>: Positive forecast aligns with current bull market
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif 'Bear' in current_regime and expected_return < 0:
                st.markdown("""
                <div style="border-left: 4px solid #f59e0b; 
                            background-color: #fffbeb; 
                            padding: 12px 15px; 
                            border-radius: 8px;">
                    <div style="font-size: 14px; color: #92400e;">
                        📉 <strong>Bearish Confirmation</strong>: Negative forecast aligns with current bear market
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ================= 4. RISK ASSESSMENT DASHBOARD =================
        st.markdown("### ⚠️ Risk Assessment")
        
        # Get risk metrics
        risk_metrics = ensemble_forecast.get('risk_metrics', {})
        var_95 = risk_metrics.get('var_95', 0) * 100 if risk_metrics.get('var_95') else 0
        max_dd = risk_metrics.get('max_drawdown', 0) * 100 if risk_metrics.get('max_drawdown') else 0
        sharpe = risk_metrics.get('sharpe', 0)
        sortino = risk_metrics.get('sortino', 0)
        risk_reward = risk_metrics.get('risk_reward', 0)
        expected_shortfall = risk_metrics.get('expected_shortfall', 0) * 100
        
        # Display risk metrics in 2x2 grid
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            # VaR (95%)
            var_color = '#ef4444' if var_95 < -15 else '#f59e0b' if var_95 < -5 else '#10b981'
            html_var = f"""
            <div style="border: 1px solid {var_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px;
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">VaR (95%)</div>
                <div style="font-size: 20px; font-weight: bold; color: {var_color};">
                    {var_95:.1f}%
                </div>
                <div style="font-size: 11px; color: #9ca3af;">Worst-case 5% outcome</div>
            </div>
            """
            st.markdown(html_var, unsafe_allow_html=True)
            
            # Max Drawdown
            dd_color = '#ef4444' if max_dd > 30 else '#f59e0b' if max_dd > 20 else '#10b981'
            html_dd = f"""
            <div style="border: 1px solid {dd_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px;
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Max Drawdown</div>
                <div style="font-size: 20px; font-weight: bold; color: {dd_color};">
                    {max_dd:.1f}%
                </div>
                <div style="font-size: 11px; color: #9ca3af;">Median expected decline</div>
            </div>
            """
            st.markdown(html_dd, unsafe_allow_html=True)
        
        with risk_col2:
            # Sharpe Ratio
            sharpe_color = '#10b981' if sharpe > 1.0 else '#f59e0b' if sharpe > 0.5 else '#ef4444'
            html_sharpe = f"""
            <div style="border: 1px solid {sharpe_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px;
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Sharpe Ratio</div>
                <div style="font-size: 20px; font-weight: bold; color: {sharpe_color};">
                    {sharpe:.2f}
                </div>
                <div style="font-size: 11px; color: #9ca3af;">Return per unit of risk</div>
            </div>
            """
            st.markdown(html_sharpe, unsafe_allow_html=True)
            
            # Expected Shortfall
            es_color = '#ef4444' if expected_shortfall < -20 else '#f59e0b' if expected_shortfall < -10 else '#10b981'
            html_es = f"""
            <div style="border: 1px solid {es_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px;
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Expected Shortfall</div>
                <div style="font-size: 20px; font-weight: bold; color: {es_color};">
                    {expected_shortfall:.1f}%
                </div>
                <div style="font-size: 11px; color: #9ca3af;">Avg loss in worst 5% scenarios</div>
            </div>
            """
            st.markdown(html_es, unsafe_allow_html=True)
        
        # Additional risk metrics
        risk_col3, risk_col4 = st.columns(2)
        
        with risk_col3:
            # Sortino Ratio
            sortino_color = '#10b981' if sortino > 1.0 else '#f59e0b' if sortino > 0.5 else '#ef4444'
            html_sortino = f"""
            <div style="border: 1px solid {sortino_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px;
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Sortino Ratio</div>
                <div style="font-size: 20px; font-weight: bold; color: {sortino_color};">
                    {sortino:.2f}
                </div>
                <div style="font-size: 11px; color: #9ca3af;">Downside risk-adjusted returns</div>
            </div>
            """
            st.markdown(html_sortino, unsafe_allow_html=True)
        
        with risk_col4:
            # Risk/Reward
            rr_color = '#10b981' if risk_reward > 2.0 else '#f59e0b' if risk_reward > 1.0 else '#ef4444'
            html_rr = f"""
            <div style="border: 1px solid {rr_color}; 
                        border-radius: 10px; 
                        padding: 15px; 
                        margin-bottom: 10px;
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);">
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 5px;">Risk/Reward</div>
                <div style="font-size: 20px; font-weight: bold; color: {rr_color};">
                    {risk_reward:.2f}
                </div>
                <div style="font-size: 11px; color: #9ca3af;">Reward per unit of risk</div>
            </div>
            """
            st.markdown(html_rr, unsafe_allow_html=True)
        
        # ================= 5. MODEL WEIGHTS VISUALIZATION =================
        st.markdown("### ⚖️ Model Contribution")
        
        weights = ensemble_forecast.get('model_weights', {})
        if weights:
            # Create pie chart
            fig_weights = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']),
                textfont=dict(size=12)
            )])
            
            fig_weights.update_layout(
                title=f"Model Weights for {horizon_key} Forecast",
                height=300,
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                template="plotly_white"
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)
        else:
            st.info("Model weights not available")
        
        # ================= 6. REGIME TIMELINE WITH PROBABILITIES =================
        st.markdown("### 📊 Regime Evolution")
        
        # Create timeline chart
        regime_data = []
        for h_key in available_horizons:
            if h_key in regime_timeline['timeline']:
                info = regime_timeline['timeline'][h_key]
                dist = info['full_distribution']
                signal_strength = info.get('signal_strength', 0.5)
                
                regime_data.append({
                    'Horizon': h_key,
                    'Days': int(h_key.replace('D', '')),
                    'Bull': dist['Bull'] * 100,
                    'Sideways': dist['Sideways'] * 100,
                    'Bear': dist['Bear'] * 100,
                    'Dominant': info['regime'],
                    'Confidence': info['probability'] * 100,
                    'Signal': signal_strength * 100
                })
        
        if regime_data:
            regime_df = pd.DataFrame(regime_data).sort_values('Days')
            
            # Stacked bar chart for regime probabilities
            fig_regime = go.Figure()
            
            fig_regime.add_trace(go.Bar(
                x=regime_df['Horizon'],
                y=regime_df['Bull'],
                name='Bull',
                marker_color='#10b981',
                hovertemplate="<b>Bull</b><br>Horizon: %{x}<br>Probability: %{y:.1f}%<extra></extra>"
            ))
            
            fig_regime.add_trace(go.Bar(
                x=regime_df['Horizon'],
                y=regime_df['Sideways'],
                name='Sideways',
                marker_color='#f59e0b',
                hovertemplate="<b>Sideways</b><br>Horizon: %{x}<br>Probability: %{y:.1f}%<extra></extra>"
            ))
            
            fig_regime.add_trace(go.Bar(
                x=regime_df['Horizon'],
                y=regime_df['Bear'],
                name='Bear',
                marker_color='#ef4444',
                hovertemplate="<b>Bear</b><br>Horizon: %{x}<br>Probability: %{y:.1f}%<extra></extra>"
            ))
            
            fig_regime.update_layout(
                title="Regime Probability by Horizon",
                xaxis_title="Forecast Horizon",
                yaxis_title="Probability (%)",
                barmode='stack',
                height=350,
                template="plotly_white",
                hovermode="x unified",
                showlegend=True
            )
            
            st.plotly_chart(fig_regime, use_container_width=True)
        
        # ================= NOW ADD THE RIGHT COLUMN EXPANDERS =================
        with right_col:
            # Export & Actions
            with st.expander("📤 Export & Actions", expanded=False):
                if st.button("📥 Download Forecast Summary", use_container_width=True):
                    try:
                        forecast_summary = {
                            'Stock': stock_name,
                            'Symbol': symbol,
                            'Current_Price': current_price,
                            'Forecast_Horizon_Days': forecast_days,
                            'Target_Price': ensemble_forecast['expected_price'],
                            'Expected_Return_Pct': ensemble_forecast['expected_return'] * 100,
                            'Confidence_Level': ensemble_forecast['confidence_level'],
                            'VaR_95_Pct': ensemble_forecast.get('risk_metrics', {}).get('var_95', 0) * 100,
                            'Max_Drawdown_Pct': ensemble_forecast.get('risk_metrics', {}).get('max_drawdown', 0) * 100,
                            'Generated_At': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        summary_df = pd.DataFrame([forecast_summary])
                        csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{symbol}_forecast_{forecast_days}D.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.warning(f"Cannot download: {str(e)}")
                
                if st.button("🔄 Refresh Forecast", type="secondary", use_container_width=True):
                    st.session_state.forecast_result = None
                    if 'forecast_engine' in st.session_state:
                        del st.session_state.forecast_engine
                    st.rerun()
            
            # Advanced Options
            with st.expander("⚙️ Advanced Options", expanded=False):
                show_raw_models = st.checkbox("Show Raw Model Outputs", value=False)
                show_regime_analysis = st.checkbox("Show Regime Analysis", value=True)
                show_confidence_intervals = st.checkbox("Show Confidence Intervals", value=True)
                show_drawdown_analysis = st.checkbox("Show Drawdown Analysis", value=True)
                show_distribution_comparison = st.checkbox("Compare Distributions", 
                                                         value=(distribution_display == "Compare All"))
                validate_models = st.checkbox("Run Model Validation", value=False)
                show_historical_accuracy = st.checkbox("Historical Backtest", value=False)
        
        # ================= 7. MULTI-HORIZON OUTLOOK TABLE =================
        st.markdown("### 📈 Multi-Horizon Outlook")
        
        # Create table using all horizons
        horizon_data = []
        for h_key in available_horizons:
            if h_key in forecast_results['ensemble']:
                forecast = forecast_results['ensemble'][h_key]
                
                # Positive Probability Calculation Fallback
                try:
                    if (h_key in raw_models['monte_carlo'] and 
                        'fat_tail' in raw_models['monte_carlo'][h_key]):
                        horizon_positive_prob = (
                            raw_models['monte_carlo'][h_key]['fat_tail']['positive_probability']
                            * 100
                        )
                    else:
                        horizon_positive_prob = 50.0
                except (KeyError, TypeError):
                    horizon_positive_prob = 50.0
                
                risk_metrics = forecast.get('risk_metrics', {})
                
                horizon_data.append({
                    'Horizon': h_key,
                    'Days': int(h_key.replace('D', '')),
                    'Target Price': f"₹{forecast['expected_price']:.2f}",
                    'Expected Return': f"{forecast['expected_return']*100:+.1f}%",
                    'Upside Prob': f"{horizon_positive_prob:.1f}%",
                    'VaR (95%)': f"{risk_metrics.get('var_95', 0)*100:+.1f}%" if risk_metrics.get('var_95') else "—",
                    'Max DD': f"{risk_metrics.get('max_drawdown', 0)*100:.1f}%" if risk_metrics.get('max_drawdown') else "—",
                    'Confidence': forecast['confidence_level'],
                    'Regime': forecast['regime']
                })
        
        # Sort by days
        horizon_df = pd.DataFrame(horizon_data)
        if not horizon_df.empty:
            horizon_df = horizon_df.sort_values('Days')
            
            # Styling functions
            def color_return(val):
                if '%' in str(val):
                    num = float(val.replace('%', '').replace('+', ''))
                    if num > 0:
                        return 'color: #1e7f43; font-weight: bold'
                    elif num < 0:
                        return 'color: #8b1e1e; font-weight: bold'
                return ''
            
            def color_confidence(val):
                if val == 'High':
                    return 'background-color: #145a32; color: white'
                elif val == 'Medium':
                    return 'background-color: #7d6608; color: white'
                elif val == 'Low':
                    return 'background-color: #78281f; color: white'
                return ''
            
            # Apply styling
            styled_df = horizon_df.style.map(color_return, subset=['Expected Return'])
            styled_df = styled_df.map(color_confidence, subset=['Confidence'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # ================= 8. DISCLAIMER =================
        st.markdown("---")
        st.warning("""
            **⚠️ Important Disclaimer:**
            
            All forecasts are based on statistical models and historical data. Past performance is not indicative of future results.
            Stock prices are influenced by numerous unpredictable factors including market sentiment, economic conditions, 
            geopolitical events, and company-specific developments.
            
            These forecasts should be used for informational purposes only and not as investment advice. 
            Always conduct your own research and consult with a financial advisor before making investment decisions.
            
            **Model Limitations:**
            - Assumes market regimes persist
            - Based on historical patterns
            - Cannot predict black swan events
            - Sensitive to distribution assumptions
            """)

# ==================================================
# Decision & Risk Lab Page (unchanged)
# ==================================================
elif page == "🧪 Decision & Risk Lab":
    st.title("🧪 Decision & Risk Lab")
    
    if "full_df" not in st.session_state or "full_info" not in st.session_state:
        st.warning("Please go to 📊 Main App first and select a stock.")
        st.stop()
    
    stock_name = st.session_state.get("stock_name", "Selected Stock")
    symbol = st.session_state.get("symbol", "")
    
    st.markdown(f"### Step 4: Decision Intelligence for **{stock_name}** ({symbol.replace('.NS', '')})")
    
    # Check if scoring and forecasting have been run
    has_scoring_data = st.session_state.scoring_result is not None
    has_forecast_data = st.session_state.forecast_result is not None
    
    # If data not available, show instructions
    if not has_scoring_data or not has_forecast_data:
        st.warning("⚠️ Please run Scoring and Forecasting steps first")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Scoring Data", "✅ Available" if has_scoring_data else "❌ Missing")
            if not has_scoring_data:
                st.info("Go to **🧠 Scoring** tab and run the scoring analysis")
        
        with col2:
            st.metric("Forecast Data", "✅ Available" if has_forecast_data else "❌ Missing")
            if not has_forecast_data:
                st.info("Go to **🔮 Forecasting** tab and run a forecast simulation")
        
        st.stop()
    
    # Get data from session state
    scoring_data = st.session_state.scoring_result
    forecast_data = st.session_state.forecast_result
    
    # Create and render the DecisionRiskLab
    lab = DecisionRiskLab(scoring_data, forecast_data)
    lab.render()

# ==================================================
# ADD AT THE VERY BOTTOM OF THE FILE (NO INDENTATION)
# ==================================================
# This prevents Streamlit from re-running calculations
# when you switch between tabs
if st.session_state.get('prevent_refresh', False):
    st.stop()