#!/usr/bin/env python3
"""매매 시점 비교: 월말 종가 신호 → (a) 월말 종가 매매  vs  (b) 다음 달 첫 거래일 시가 매매
 현행 전략 (6M 모멘텀 Top5, AAA=ERC, DM 50:50)
 A) 12종목 실제 상품 2022-05~2026-10
 B) 장기 10종목 (SPY VGK EWJ EEM IYR GLD DBC IEF TLT BIL) 2008~2026-10 : 시가 데이터가 모두 있는 상품만
"""
import json
import sys

import pandas as pd

from erc import aaa_weights, stats
from stack import T12, LAST

U10 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "DBC", "IEF", "TLT", "BIL"]


def run(close, opn, uni, cash, months):
    mc = close.groupby(close.index.to_period("M")).tail(1); mc.index = mc.index.to_period("M")
    mo = opn.groupby(opn.index.to_period("M")).head(1); mo.index = mo.index.to_period("M")
    dret = close.pct_change()
    rc, ro, hold, gaps = [], [], [], []
    for j in range(len(months) - 2):
        p, p1, p2 = months[j], months[j + 1], months[j + 2]
        i = mc.index.get_loc(p)
        s = (mc.iloc[i][uni] / mc.iloc[i - 6][uni] - 1).dropna()
        top = sorted(s.index, key=lambda x: (-s[x], uni.index(x)))[:5]
        wa = aaa_weights(dret, top, p, "erc")
        w = {t: 0.5 * v for t, v in wa.items()}
        for t in top:
            k = t if s[t] >= 0 else cash
            w[k] = w.get(k, 0) + 0.1
        rc.append(float(sum(v * (mc.loc[p1, t] / mc.loc[p, t] - 1) for t, v in w.items())))      # 종가→종가
        ro.append(float(sum(v * (mo.loc[p2, t] / mo.loc[p1, t] - 1) for t, v in w.items())))     # 다음달 첫 시가→그 다음달 첫 시가
        hold.append(str(p1))
    return rc, ro, hold


def pack(r, hold):
    st = {"full": stats(r), "5y": stats(r[-60:])}
    if len(r) >= 120:
        st["10y"] = stats(r[-120:])
    return {"period": [hold[0], hold[-1]], "stats": st}


def main():
    import yfinance as yf
    need = sorted(set(T12) | set(U10))
    df = None
    for _ in range(3):
        df = yf.download(need, start="2007-01-01", auto_adjust=True, progress=False, threads=False)
        if all(df["Close"][t].notna().any() for t in need):
            break
    days = df["Close"]["SPY"].dropna().index
    close = df["Close"].reindex(days).ffill(limit=5)
    opn = df["Open"].reindex(days)
    opn = opn.where(opn > 0).fillna(close.shift(1))
    res = {}
    for tag, uni, cash, s0 in (("A", T12, "SGOV", pd.Period("2022-04")), ("B", U10, "BIL", None)):
        c, o = close[uni], opn[uni]
        if s0 is None:
            s0 = max(c[t].first_valid_index() for t in uni).to_period("M") + 6
        ms = [p for p in c.groupby(c.index.to_period("M")).tail(1).index.to_period("M") if s0 <= p <= LAST + 1]
        rc, ro, hold = run(c, o, uni, cash, ms)
        res[f"{tag}_close"] = pack(rc, hold)
        res[f"{tag}_open"] = pack(ro, hold)
        diff = [a - b for a, b in zip(ro, rc)]
        res[f"{tag}_diff"] = {"mean_monthly": sum(diff) / len(diff), "pos_months": sum(d > 0 for d in diff), "n": len(diff),
                              "worst": min(diff), "best": max(diff)}
        for k in ("close", "open"):
            st = res[f"{tag}_{k}"]["stats"]
            print(tag, k, res[f"{tag}_{k}"]["period"], " | ".join(f"{a} {v['cagr']:.2%}/{v['mdd']:.2%}" for a, v in st.items()), flush=True)
        print(tag, "diff", res[f"{tag}_diff"])
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/openexec.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
