def forecast_range_insight(ctx):
    band_width = ctx["band_width"]

    if band_width > 0.6:
        return "Wide forecast range indicates high uncertainty and elevated volatility expectations."
    else:
        return "Forecast range remains relatively contained, suggesting stable expectations."


def regime_insight(ctx):
    dominant = ctx["dominant_regime"]
    prob = ctx["probability"]

    if prob > 0.6:
        return f"Market conditions are currently dominated by the {dominant} regime."
    else:
        return "No single regime dominates, indicating transitional or uncertain conditions."


def risk_insight(ctx):
    drawdown = ctx["drawdown_95"]

    if drawdown < -0.35:
        return "Severe downside scenarios are possible under adverse market conditions."
    else:
        return "Downside risk remains within historically typical ranges."
