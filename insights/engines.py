from insights.base import insight

from insights.price import price_trend_insight, volume_insight
from insights.momentum import rsi_insight, macd_insight
from insights.fundamentals import fundamentals_table_insight, peer_comparison_insight
from insights.scoring import score_insight, pillar_insight
from insights.forecast import forecast_range_insight, regime_insight, risk_insight

INSIGHT_REGISTRY = {
    "price_trend": price_trend_insight,
    "volume": volume_insight,
    "rsi": rsi_insight,
    "macd": macd_insight,
    "fundamentals": fundamentals_table_insight,
    "peers": peer_comparison_insight,
    "score": score_insight,
    "pillar": pillar_insight,
    "forecast_range": forecast_range_insight,
    "regime": regime_insight,
    "risk": risk_insight,
}

def generate_insight(insight_type: str, context: dict) -> str:
    func = INSIGHT_REGISTRY.get(insight_type)
    if not func:
        return insight("Interpretation not available.")

    return insight(func(context))
