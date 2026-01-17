def score_insight(ctx):
    score = ctx["score"]

    if score >= 80:
        return "Strong alignment across fundamentals and momentum based on historical patterns."
    elif score >= 60:
        return "Mixed signals present, indicating selective strengths with some risks."
    else:
        return "Weak alignment across key metrics suggests elevated risk under current conditions."


def pillar_insight(ctx):
    strongest = ctx["strongest"]
    weakest = ctx["weakest"]

    return (
        f"{strongest} is the strongest contributor, while {weakest} remains a relative weakness."
    )
