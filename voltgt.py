#!/usr/bin/env python3
"""변동성 타깃팅 테스트: 현행 50:50 최종 비중의 예상 변동성(직전 3개월 일별, 연환산)이
 목표를 넘으면 전체를 k = 목표/예상변동성 으로 줄이고 나머지는 SGOV
 A) 2022-05~2026-10 실제 12종목  B) 2015-04~2026-10 장기 연결 (stack.py와 동일 연결)
"""
import json
import math
import sys

import numpy as np
import pandas as pd

from erc import aaa_weights, stats
from stack import T12, CASH, LAST, splice

TARGETS = [None, 0.06, 0.08, 0.10, 0.12]


def run(adj, months, target):
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    r, hold, ks = [], [], []
    for j in range(len(months) - 1):
        p, p1 = months[j], months[j + 1]
        i = mpx.index.get_loc(p)
        s = (mpx.iloc[i][T12] / mpx.iloc[i - 6][T12] - 1).dropna()
        top = sorted(s.index, key=lambda x: (-s[x], T12.index(x)))[:5]
        wa = aaa_weights(dret, top, p, "erc")
        w = {t: 0.5 * v for t, v in wa.items()}
        for t in top:
            k = t if s[t] >= 0 else CASH
            w[k] = w.get(k, 0) + 0.1
        k = 1.0
        if target:
            win = dret.loc[(p - 2).start_time:p.end_time, list(w)].dropna()
            pv = float((win.values @ np.array(list(w.values()))).std(ddof=1) * math.sqrt(252))
            k = min(1.0, target / pv) if pv > 0 else 1.0
        wf = {t: v * k for t, v in w.items()}
        wf[CASH] = wf.get(CASH, 0) + (1 - k)
        nxt = mpx.loc[p1] / mpx.loc[p] - 1
        r.append(float(sum(v * nxt[t] for t, v in wf.items()))); hold.append(str(p1)); ks.append(k)
    return r, hold, ks


def main():
    import yfinance as yf
    need = sorted(set(T12) | {"AQMIX", "BIL", "DBC", "BTC-USD"})
    raw = yf.download(need, start="2009-06-01", auto_adjust=True, progress=False)["Close"]
    days = raw["SPY"].dropna().index
    raw = raw.reindex(days).ffill(limit=5)
    a = raw[T12].copy()
    mA = [p for p in a.groupby(a.index.to_period("M")).tail(1).index.to_period("M") if pd.Period("2022-04") <= p <= LAST]
    b = raw[T12].copy()
    b["DBMF"] = splice(raw, "DBMF", "AQMIX"); b["SGOV"] = splice(raw, "SGOV", "BIL")
    b["PDBC"] = splice(raw, "PDBC", "DBC"); b["BITO"] = raw["BTC-USD"]
    first = max(b[t].first_valid_index() for t in T12).to_period("M") + 6
    mB = [p for p in b.groupby(b.index.to_period("M")).tail(1).index.to_period("M") if first <= p <= LAST]
    res = {}
    for tag, df, ms in (("A", a, mA), ("B", b, mB)):
        for tg in TARGETS:
            r, hold, ks = run(df, ms, tg)
            name = f"{tag}_{'base' if tg is None else int(tg * 100)}"
            st = {"full": stats(r), "5y": stats(r[-60:])}
            if len(r) >= 120:
                st["10y"] = stats(r[-120:])
            yr = {}
            for x, m in zip(r, hold):
                yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
            res[name] = {"period": [hold[0], hold[-1]], "stats": st, "avg_k": round(sum(ks) / len(ks), 3),
                         "cut_months": sum(k < 1 for k in ks), "yearly": {y: round(v - 1, 4) for y, v in yr.items()}}
            print(name, " | ".join(f"{k} {v['cagr']:.2%}/{v['mdd']:.2%}" for k, v in st.items()), "avg_k", res[name]["avg_k"])
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/voltgt.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
