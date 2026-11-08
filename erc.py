#!/usr/bin/env python3
"""리스크패리티 비교: 역변동성(1/σ) vs 정밀(동일위험기여, 공분산 반영)"""
import json
import math
import sys

import numpy as np
import pandas as pd

PV10 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "SGOV", "DBMF", "BITO"]
U10 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "DBC", "IEF", "TLT", "BIL"]
ALL = sorted(set(PV10) | set(U10))
SPLIT = "2016-12"


def erc_weights(cov, iters=500, tol=1e-12):
    """동일위험기여: 순환 좌표하강법 (각 종목의 w_i*(Σw)_i 가 같아지도록)"""
    n = cov.shape[0]
    x = 1 / np.sqrt(np.diag(cov))
    x = x / x.sum()
    b = np.full(n, 1.0 / n)
    for _ in range(iters):
        x_old = x.copy()
        for i in range(n):
            s = cov[i] @ x - cov[i, i] * x[i]
            x[i] = (-s + math.sqrt(s * s + 4 * cov[i, i] * b[i])) / (2 * cov[i, i])
        if np.max(np.abs(x - x_old)) < tol:
            break
    return x / x.sum()


def aaa_weights(dret, top, p, method):
    r = dret.loc[(p - 2).start_time:p.end_time, top].dropna()
    if method == "inv":
        inv = 1 / r.std(ddof=1)
        return (inv / inv.sum()).to_dict()
    w = erc_weights(r.cov().values)
    return dict(zip(top, w))


def stats(r):
    r = pd.Series(r)
    eq = (1 + r).cumprod()
    return {"cagr": float(eq.iloc[-1] ** (12 / len(r)) - 1),
            "mdd": float((eq / eq.cummax().clip(lower=1) - 1).min()),
            "sd": float(r.std(ddof=1) * math.sqrt(12)), "n": len(r)}


def run(mpx, dret, tickers, cash, months, method):
    aaa, mix, hold, wlog = [], [], [], []
    for j in range(len(months) - 1):
        p, p1 = months[j], months[j + 1]
        i = mpx.index.get_loc(p)
        s = (mpx.iloc[i][tickers] / mpx.iloc[i - 6][tickers] - 1).dropna()
        top = sorted(s.index, key=lambda x: (-s[x], tickers.index(x)))[:5]
        nxt = mpx.loc[p1] / mpx.loc[p] - 1
        wa = aaa_weights(dret, top, p, method)
        ra = sum(w * nxt[t] for t, w in wa.items())
        wd = {}
        for t in top:
            k = t if s[t] >= 0 else cash
            wd[k] = wd.get(k, 0) + 0.2
        rd = sum(w * nxt[t] for t, w in wd.items())
        aaa.append(ra); mix.append((ra + rd) / 2); hold.append(str(p1))
        wlog.append({"hold": str(p1), "w": {t: round(float(w), 4) for t, w in wa.items()}})
    return aaa, mix, hold, wlog


def split_stats(r, hold):
    h1 = [x for x, m in zip(r, hold) if m <= SPLIT]
    h2 = [x for x, m in zip(r, hold) if m > SPLIT]
    return {"full": stats(r), "h1": stats(h1), "h2": stats(h2), "last5y": stats(r[-60:])}


def main():
    import yfinance as yf
    adj = yf.download(ALL, start="2004-01-01", auto_adjust=True, progress=False)["Close"][ALL].sort_index()
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    res = {}
    # 1) PV 재현 (2022-05 ~ 2026-10 보유분)
    pm = [p for p in mpx.index if pd.Period("2022-04") <= p <= pd.Period("2026-10")]
    for mth in ["inv", "erc"]:
        a, m, h, wl = run(mpx, dret, PV10, "SGOV", pm, mth)
        res[f"pv_{mth}"] = {"weights": wl, "aaa": stats(a)}
    # 2) 장기 (2008~)
    first = max(adj[t].first_valid_index() for t in U10).to_period("M") + 7
    lm = [p for p in mpx.index if first <= p <= pd.Period("2026-10")]
    for mth in ["inv", "erc"]:
        a, m, h, _ = run(mpx, dret, U10, "BIL", lm, mth)
        res[f"long_{mth}"] = {"aaa": split_stats(a, h), "mix": split_stats(m, h), "period": [h[0], h[-1]]}
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/erc.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    for k in ["long_inv", "long_erc"]:
        for s in ["aaa", "mix"]:
            v = res[k][s]["full"]
            print(k, s, f"CAGR {v['cagr']:.2%} MDD {v['mdd']:.2%}")


if __name__ == "__main__":
    main()
