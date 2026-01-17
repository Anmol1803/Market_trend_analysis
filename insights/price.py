def price_trend_insight(ctx):
    price = ctx["price"]
    ma20 = ctx["ma20"]
    ma50 = ctx["ma50"]

    if price > ma20 > ma50:
        return "Price is trading above short- and medium-term averages, indicating positive trend alignment."
    elif price < ma20 < ma50:
        return "Price remains below key averages, suggesting weak momentum or consolidation."
    else:
        return "Price is moving near key averages, indicating indecision or range-bound behavior."


def volume_insight(ctx):
    vol = ctx["volume"]
    avg_vol = ctx["avg_volume"]

    if vol > 1.5 * avg_vol:
        return "Elevated volume suggests strong market participation and higher conviction."
    elif vol < 0.7 * avg_vol:
        return "Subdued volume indicates limited participation and reduced momentum."
    else:
        return "Volume levels appear in line with recent averages."
