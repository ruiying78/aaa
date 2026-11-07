#!/usr/bin/env python3
"""50:50 (AAA + 듀얼모멘텀) PV 방식 견고성 점검: 모멘텀 기간 x 종목 수 x AAA 절대모멘텀"""
import json
import math
import sys

import numpy as np
import pandas as pd

U10 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "DBC", "IEF", "TLT", "BIL"]
END = pd.Period("2026-10")
SPLIT = "2016-12"
ANN = math.sqrt(252)


def stats(r):
    r = pd.Series(r)
    if len(r) == 0:
        return {}
    eq = (1 + r).cumprod()
    return {"cagr": float(eq.iloc[-1] ** (12 / len(r)) - 1),
            "mdd": float((eq / eq.cummax().clip(lower=1) - 1).min()),
            "sd": float(r.std(ddof=1) * math.sqrt(12)), "n": len(r)}


def main():
    import yfinance as yf
    adj = yf.download(U10, start="2004-01-01", auto_adjust=True, progress=False)["Close"][U10].sort_index()
    first = max(adj[t].first_valid_index() for t in U10).to_period("M")
    start_sig = first + 13                       # 12개월 모멘텀까지 가능한 공통 시작점
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    drets = adj.pct_change()
    months = [p for p in mpx.index if start_sig <= p <= END]

    def score(p, L):
        i = mpx.index.get_loc(p)
        cur = mpx.iloc[i]
        if L == "blend":
            return sum(cur / mpx.iloc[i - k] - 1 for k in (3, 6, 12)) / 3
        return cur / mpx.iloc[i - L] - 1

    def rp_weights(p, top):
        r = drets.loc[(p - 2).start_time:p.end_time, top].dropna()
        inv = 1 / r.std(ddof=1)
        return inv / inv.sum()

    def strat(L, N, absf):
        aaa, dm = [], []
        for j in range(len(months) - 1):
            p, p1 = months[j], months[j + 1]
            s = score(p, L).dropna()
            top = sorted(s.index, key=lambda x: (-s[x], U10.index(x)))[:N]
            nxt = mpx.loc[p1] / mpx.loc[p] - 1
            w = rp_weights(p, top)
            wa = {}
            for x in top:
                k = "BIL" if (absf and s[x] < 0) else x
                wa[k] = wa.get(k, 0) + w[x]
            aaa.append(sum(v * nxt[k] for k, v in wa.items()))
            wd = {}
            for x in top:
                k = x if s[x] >= 0 else "BIL"
                wd[k] = wd.get(k, 0) + 1 / N
            dm.append(sum(v * nxt[k] for k, v in wd.items()))
        mix = [(a + b) / 2 for a, b in zip(aaa, dm)]
        hold = [str(m) for m in months[1:]]
        h1 = [r for r, m in zip(mix, hold) if m <= SPLIT]
        h2 = [r for r, m in zip(mix, hold) if m > SPLIT]
        return {"full": stats(mix), "h1": stats(h1), "h2": stats(h2), "last5y": stats(mix[-60:])}

    res = {"period": [str(months[1]), str(months[-1])], "grid": []}
    for L in [3, 6, 9, 12, "blend"]:
        for N in [3, 4, 5]:
            for absf in [False, True]:
                r = strat(L, N, absf)
                res["grid"].append({"L": L, "N": N, "abs": absf, **r})
                f = r["full"]
                print(f"L={L!s:5} N={N} abs={absf!s:5} CAGR {f['cagr']:.2%} MDD {f['mdd']:.2%} | "
                      f"H1 {r['h1']['cagr']:.2%}/{r['h1']['mdd']:.2%} H2 {r['h2']['cagr']:.2%}/{r['h2']['mdd']:.2%}")
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/grid.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
