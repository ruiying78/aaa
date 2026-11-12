#!/usr/bin/env python3
"""CLSE(롱숏 주식) 추가 효과: 13종목 추가 / SPY→CLSE 교체 / SPY↔CLSE 게이트 (A: 실제 상품, B: MARNX(전신 펀드) 연결 가능 시), 현행 전략(PV 6M Top5, AAA=ERC, DM, 50:50)
 A) 2022-05~2026-10 실제 상품  B) 2015-04~2026-10 장기 연결 (stack.py와 동일 연결)
"""
import json
import sys

import pandas as pd

from erc import aaa_weights, stats
from stack import T12, CASH, LAST, splice

EXTRA = ["CLSE"]
T13 = T12 + ["CLSE"]


def run(adj, months, uni, gate=()):
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    mix, aaa, dm, hold, nin = [], [], [], [], 0
    for j in range(len(months) - 1):
        p, p1 = months[j], months[j + 1]
        i = mpx.index.get_loc(p)
        s = (mpx.iloc[i][uni] / mpx.iloc[i - 6][uni] - 1).dropna()
        top = sorted(s.index, key=lambda x: (-s[x], uni.index(x)))[:5]
        if gate and "SPY" in top:
            best, bs = "SPY", s["SPY"]
            for g in gate:
                q = mpx.iloc[i][g] / mpx.iloc[i - 6][g] - 1
                if pd.notna(q) and q > bs:
                    best, bs = g, q
            if best != "SPY":
                top[top.index("SPY")] = best; s[best] = bs
        nxt = mpx.loc[p1] / mpx.loc[p] - 1
        wa = aaa_weights(dret, top, p, "erc")
        ra = sum(w * nxt[t] for t, w in wa.items())
        wd = {}
        for t in top:
            k = t if s[t] >= 0 else CASH
            wd[k] = wd.get(k, 0) + 0.2
        rd = sum(w * nxt[t] for t, w in wd.items())
        mix.append(float((ra + rd) / 2)); aaa.append(float(ra)); dm.append(float(rd)); hold.append(str(p1))
        nin += any(t in top for t in EXTRA)
    return {"mix": mix, "aaa": aaa, "dm": dm}, hold, nin


def pack(r, hold):
    st = {"full": stats(r), "5y": stats(r[-60:])}
    if len(r) >= 120:
        st["10y"] = stats(r[-120:])
    yr = {}
    for x, m in zip(r, hold):
        yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
    return {"stats": st, "yearly": {y: round(v - 1, 4) for y, v in yr.items()}}


def main():
    import yfinance as yf
    need = sorted(set(T12) | set(EXTRA) | {"AQMIX", "BIL", "DBC", "BTC-USD", "MARNX"})
    raw = None
    for _ in range(3):
        raw = yf.download(need, start="2009-06-01", auto_adjust=True, progress=False, threads=False)["Close"]
        if all(raw[t].notna().any() for t in need if t != "MARNX"):
            break
    days = raw["SPY"].dropna().index
    raw = raw.reindex(days).ffill(limit=5)
    a = raw[T12 + EXTRA].copy()
    fa = max(a[t].first_valid_index() for t in T12 + EXTRA).to_period("M") + 6
    mA = [p for p in a.groupby(a.index.to_period("M")).tail(1).index.to_period("M") if fa <= p <= LAST]
    has_m = "MARNX" in raw and raw["MARNX"].notna().sum() > 500
    b = raw[T12 + EXTRA].copy()
    if has_m:
        b["CLSE"] = splice(raw, "CLSE", "MARNX")
    b["DBMF"] = splice(raw, "DBMF", "AQMIX"); b["SGOV"] = splice(raw, "SGOV", "BIL")
    b["PDBC"] = splice(raw, "PDBC", "DBC"); b["BITO"] = raw["BTC-USD"]
    first = max(b[t].first_valid_index() for t in T12 + EXTRA).to_period("M") + 6
    mB = [p for p in b.groupby(b.index.to_period("M")).tail(1).index.to_period("M") if first <= p <= LAST]
    res = {"clse_start": str(raw["CLSE"].first_valid_index().date()),
           "marnx": [str(raw["MARNX"].first_valid_index().date()), str(raw["MARNX"].last_valid_index().date())] if has_m else None}
    sets = [("A", a, mA)] + ([("B", b, mB)] if has_m else [])
    for tag, df, ms in sets:
        for nm, uni, g in (("12", T12, ()), ("13", T13, ()), ("rep", ["CLSE" if t == "SPY" else t for t in T12], ()),
                           ("gate", T12, ("CLSE",))):
            R, hold, nin = run(df, ms, uni, g)
            for k, r in R.items():
                res[f"{tag}_{nm}_{k}"] = pack(r, hold)
            res[f"{tag}_{nm}_mix"]["period"] = [hold[0], hold[-1]]
            res[f"{tag}_{nm}_mix"]["swap_months"] = nin
            st = res[f"{tag}_{nm}_mix"]["stats"]
            print(tag, nm, " | ".join(f"{k} {v['cagr']:.2%}/{v['mdd']:.2%}" for k, v in st.items()), "swap", nin, flush=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/clse.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
