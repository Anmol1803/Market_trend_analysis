def rsi_insight(ctx):
    rsi = ctx["rsi"]

    if rsi > 70:
        return "Momentum appears overheated, increasing short-term pullback risk."
    elif rsi < 30:
        return "Momentum is oversold, often seen near short-term exhaustion zones."
    else:
        return "Momentum remains balanced with no extreme conditions."


def macd_insight(ctx):
    macd = ctx["macd"]
    signal = ctx["signal"]

    if macd > signal:
        return "Momentum trend is positive with MACD above the signal line."
    else:
        return "Momentum appears to be weakening as MACD trades below the signal line."
