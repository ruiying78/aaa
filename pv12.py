#!/usr/bin/env python3
"""현행 12종목 · PV 방식(6개월 총수익률 Top5) 4.5년 백테스트: AAA(ERC/역변동성) · DM · 50:50"""
import json
import math
import sys

import pandas as pd

from erc import aaa_weights, stats

T12 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "IEF", "TLT", "SGOV", "DBMF", "BITO"]
CASH = "SGOV"
FIRST, LAST = pd.Period("2022-04"), pd.Period("2026-10")   # 첫 신호 2022-04말 → 성과 2022-05 ~ 2026-10


def main():
    import yfinance as yf
    adj = yf.download(T12, start="2021-09-01", auto_adjust=True, progress=False)["Close"][T12].sort_index()
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    months = [p for p in mpx.index if FIRST <= p <= LAST]
    R = {k: [] for k in ("aaa_erc", "aaa_inv", "dm", "mix_erc", "mix_inv")}
    hold, rows = [], []
    for j in range(len(months) - 1):
        p, p1 = months[j], months[j + 1]
        i = mpx.index.get_loc(p)
        s = (mpx.iloc[i][T12] / mpx.iloc[i - 6][T12] - 1).dropna()
        top = sorted(s.index, key=lambda x: (-s[x], T12.index(x)))[:5]
        nxt = mpx.loc[p1] / mpx.loc[p] - 1
        we = aaa_weights(dret, top, p, "erc")
        wi = aaa_weights(dret, top, p, "inv")
        re_ = sum(w * nxt[t] for t, w in we.items())
        ri = sum(w * nxt[t] for t, w in wi.items())
        wd = {}
        for t in top:
            k = t if s[t] >= 0 else CASH
            wd[k] = wd.get(k, 0) + 0.2
        rd = sum(w * nxt[t] for t, w in wd.items())
        for k, v in (("aaa_erc", re_), ("aaa_inv", ri), ("dm", rd), ("mix_erc", (re_ + rd) / 2), ("mix_inv", (ri + rd) / 2)):
            R[k].append(float(v))
        hold.append(str(p1))
        rows.append({"hold": str(p1), "top": top, "dm": {t: round(w, 2) for t, w in wd.items()},
                     "aaa_erc": {t: round(float(w), 4) for t, w in we.items()},
                     "r": {k: round(R[k][-1], 5) for k in R}})
    res = {"period": [hold[0], hold[-1]], "rows": rows, "stats": {}, "yearly": {}}
    spy = [float(mpx.loc[months[j + 1], "SPY"] / mpx.loc[months[j], "SPY"] - 1) for j in range(len(months) - 1)]
    R["spy"] = spy
    for k, r in R.items():
        res["stats"][k] = stats(r)
        yr = {}
        for x, m in zip(r, hold):
            yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
        res["yearly"][k] = {y: round(v - 1, 4) for y, v in yr.items()}
        st = res["stats"][k]
        print(k, f"CAGR {st['cagr']:.2%} MDD {st['mdd']:.2%} SD {st['sd']:.2%}")
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/pv12.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
