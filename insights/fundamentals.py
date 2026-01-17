from insights.base import choose

# -------- FUNDAMENTALS TABLE --------

def fundamentals_table_insight(ctx):
    pe = ctx.get("pe")
    pb = ctx.get("pb")
    roe = ctx.get("roe")
    sector_pe = ctx.get("sector_pe")

    return choose(
        (pe and sector_pe and pe < sector_pe * 0.8,
         "Valuation appears attractive relative to sector averages."),
        
        (pe and sector_pe and pe > sector_pe * 1.2,
         "Stock trades at a premium compared to sector peers."),
        
        (roe and roe > 0.15,
         "Profitability metrics indicate efficient capital usage."),
        
        (True,
         "Fundamental metrics remain broadly in line with sector norms.")
    )


# -------- PEER COMPARISON --------

def peer_comparison_insight(ctx):
    rank = ctx.get("rank")          # e.g. ROE rank among peers
    peer_count = ctx.get("count")

    if rank and peer_count:
        if rank <= peer_count * 0.3:
            return "Company ranks among top peers on key efficiency metrics."
        elif rank >= peer_count * 0.7:
            return "Company lags several peers on comparative metrics."

    return "Peer comparison shows mixed relative positioning."
