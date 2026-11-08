#!/usr/bin/env python3
"""리턴 스태킹 테스트: SPY 자리를 'SPY 100% + 관리형선물 100%' 합성 자산(SPY+MF)으로 교체
 합성 일수익률 = SPY + (관리형선물 - 단기국채) - 추가보수(연 0.86% = 스태킹 ETF 0.95% - SPY 0.09%)
 관리형선물 = DBMF (상장 전 AQMIX). 단기국채 = BIL
 A) 2022-05~2026-10: 실제 12종목 (BITO 실물)
 B) 2015-04~2026-10: 장기 연결 (BITO→BTC-USD, DBMF←AQMIX, SGOV←BIL, PDBC←DBC)
"""
import json
import sys

import pandas as pd

from erc import aaa_weights, stats

T12 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "IEF", "TLT", "SGOV", "DBMF", "BITO"]
CASH = "SGOV"
FEE = 0.0086 / 252
LAST = pd.Period("2026-10")


def splice(raw, actual, proxy):
    a, p = raw[actual], raw[proxy]
    start = a.first_valid_index()
    r = p.pct_change().where(p.index < start, a.pct_change())
    r.loc[start] = p.pct_change().loc[start]
    first = p.first_valid_index()
    return (1 + r.loc[first:].fillna(0)).cumprod() * 100


def run(adj, months):
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    r, hold, nin = [], [], 0
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
        r.append(float((ra + rd) / 2)); hold.append(str(p1))
        nin += "SPY" in top
    return r, hold, nin


def stacked(spy, mf, bil):
    rs = spy.pct_change() + (mf.pct_change() - bil.pct_change()) - FEE
    first = max(spy.first_valid_index(), mf.first_valid_index(), bil.first_valid_index())
    return (1 + rs.loc[first:].fillna(0)).cumprod() * 100


def summarize(name, r, hold, nin, out):
    st = {"full": stats(r), "5y": stats(r[-60:])}
    if len(r) >= 120:
        st["10y"] = stats(r[-120:])
    yr = {}
    for x, m in zip(r, hold):
        yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
    out[name] = {"period": [hold[0], hold[-1]], "stats": st, "spy_slot_months": nin,
                 "yearly": {y: round(v - 1, 4) for y, v in yr.items()}}
    print(name, out[name]["period"], " | ".join(f"{k} {v['cagr']:.2%}/{v['mdd']:.2%}" for k, v in st.items()), "SPY슬롯", nin)


def main():
    import yfinance as yf
    need = sorted(set(T12) | {"AQMIX", "BIL", "DBC", "BTC-USD"})
    raw = yf.download(need, start="2009-06-01", auto_adjust=True, progress=False)["Close"]
    days = raw["SPY"].dropna().index
    raw = raw.reindex(days).ffill(limit=5)
    mf = splice(raw, "DBMF", "AQMIX")
    res = {}
    # A) 실제 상품
    a = raw[T12].copy()
    mA = [p for p in a.groupby(a.index.to_period("M")).tail(1).index.to_period("M") if pd.Period("2022-04") <= p <= LAST]
    summarize("A_base", *run(a, mA), res)
    a2 = a.copy(); a2["SPY"] = stacked(raw["SPY"], mf, raw["BIL"])
    summarize("A_stack", *run(a2, mA), res)
    # B) 장기 연결
    b = raw[T12].copy()
    b["DBMF"] = mf
    b["SGOV"] = splice(raw, "SGOV", "BIL")
    b["PDBC"] = splice(raw, "PDBC", "DBC")
    b["BITO"] = raw["BTC-USD"]
    first = max(b[t].first_valid_index() for t in T12).to_period("M") + 6
    mB = [p for p in b.groupby(b.index.to_period("M")).tail(1).index.to_period("M") if first <= p <= LAST]
    summarize("B_base", *run(b, mB), res)
    b2 = b.copy(); b2["SPY"] = stacked(raw["SPY"], mf, raw["BIL"])
    summarize("B_stack", *run(b2, mB), res)
    # 참고: 합성 자산 단독 vs SPY 단독 (B 기간)
    for nm, ser in (("SPY", b["SPY"]), ("SPY+MF", b2["SPY"])):
        m = ser.groupby(ser.index.to_period("M")).last()
        rr = (m / m.shift(1) - 1).loc[mB[1]:mB[-1]].tolist()
        res["asset_" + nm] = stats(rr)
        print("asset", nm, f"{res['asset_' + nm]['cagr']:.2%}/{res['asset_' + nm]['mdd']:.2%}")
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/stack.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
