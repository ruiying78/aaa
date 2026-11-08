#!/usr/bin/env python3
"""12종목 PV 방식 장기 백테스트 (상장 전 구간은 대체 자산으로 연결)
 - DBMF ← AQMIX (상장 전), BITO → BTC-USD (전 기간), SGOV ← BIL, PDBC ← DBC
 - AAA(ERC 리스크패리티) · DM(각 20%, 음수→SGOV) · 50:50, 월말 리밸런싱, 배당 포함
"""
import json
import sys

import pandas as pd

from erc import aaa_weights, stats

T12 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "IEF", "TLT", "SGOV", "DBMF", "BITO"]
SPLICE = {"DBMF": "AQMIX", "SGOV": "BIL", "PDBC": "DBC"}   # 실제 상품 상장 전은 대체 자산 수익률 사용
REPLACE = {"BITO": "BTC-USD"}                              # 전 기간 대체
CASH = "SGOV"
LAST = pd.Period("2026-10")


def build():
    import yfinance as yf
    need = sorted(set(T12) | set(SPLICE.values()) | set(REPLACE.values()))
    raw = yf.download(need, start="2009-06-01", auto_adjust=True, progress=False)["Close"]
    days = raw["SPY"].dropna().index                      # 미국 거래일 기준 (비트코인 주말 가격 제외)
    raw = raw.reindex(days).ffill(limit=5)
    out = pd.DataFrame(index=days)
    info = {}
    for t in T12:
        if t in REPLACE:
            s = raw[REPLACE[t]]
            info[t] = f"{REPLACE[t]} 전 기간"
        elif t in SPLICE:
            a, p = raw[t], raw[SPLICE[t]]
            start = a.first_valid_index()
            r = p.pct_change().where(p.index < start, a.pct_change())
            r.loc[start] = p.loc[start] / p.shift(1).loc[start] - 1 if pd.notna(p.shift(1).loc[start]) else 0.0
            first = p.first_valid_index()
            s = (1 + r.loc[first:].fillna(0)).cumprod() * 100
            s.iloc[0] = 100
            info[t] = f"{start.date()} 이전은 {SPLICE[t]}"
        else:
            s = raw[t]
        out[t] = s
    return out, info


def main():
    adj, info = build()
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    first = max(adj[t].first_valid_index() for t in T12).to_period("M") + 6
    months = [p for p in mpx.index if first <= p <= LAST]
    R = {k: [] for k in ("mix", "aaa", "dm", "spy")}
    hold, rows = [], []
    for j in range(len(months) - 1):
        p, p1 = months[j], months[j + 1]
        i = mpx.index.get_loc(p)
        s = (mpx.iloc[i][T12] / mpx.iloc[i - 6][T12] - 1).dropna()
        top = sorted(s.index, key=lambda x: (-s[x], T12.index(x)))[:5]
        nxt = mpx.loc[p1] / mpx.loc[p] - 1
        wa = aaa_weights(dret, top, p, "erc")
        ra = sum(w * nxt[t] for t, w in wa.items())
        wd = {}
        for t in top:
            k = t if s[t] >= 0 else CASH
            wd[k] = wd.get(k, 0) + 0.2
        rd = sum(w * nxt[t] for t, w in wd.items())
        for k, v in (("mix", (ra + rd) / 2), ("aaa", ra), ("dm", rd), ("spy", nxt["SPY"])):
            R[k].append(float(v))
        hold.append(str(p1))
        rows.append({"hold": str(p1), "top": top, "r": {k: round(R[k][-1], 5) for k in R}})
    res = {"period": [hold[0], hold[-1]], "info": info, "rows": rows, "stats": {}, "yearly": {}}
    for k, r in R.items():
        res["stats"][k] = {"full": stats(r), "10y": stats(r[-120:]), "5y": stats(r[-60:])}
        yr = {}
        for x, m in zip(r, hold):
            yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
        res["yearly"][k] = {y: round(v - 1, 4) for y, v in yr.items()}
        st = res["stats"][k]
        print(k, " | ".join(f"{a} {st[a]['cagr']:.2%}/{st[a]['mdd']:.2%}" for a in st))
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/pv12x.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
