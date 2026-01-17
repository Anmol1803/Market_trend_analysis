def insight(text: str) -> str:
    """
    Standard wrapper for all insights.
    Keeps tone consistent across app.
    """
    return f"📌 **Insight:** {text}"


def choose(*conditions):
    """
    Utility to return first valid (condition, text)
    """
    for cond, text in conditions:
        if cond:
            return text
    return None


def safe(val, default=None):
    """
    Safe value getter
    """
    return default if val is None else val
