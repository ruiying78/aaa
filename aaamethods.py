#!/usr/bin/env python3
"""AAA 비중 방식 비교: ERC(현행) / 역변동성 / 최소분산(원조 AAA 방식) / 동일비중
 각각 AAA 단독과 AAA+DM 50:50, A) 2022-05~2026-10 실제 12종목  B) 2015-04~2026-10 장기 연결
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

from erc import erc_weights, stats
from stack import T12, CASH, LAST, splice

METHODS = ["erc", "inv", "minvar", "equal"]


def minvar_weights(cov):
    """롱온리 최소분산: 모든 부분집합의 무제약 해 중 음수 없는 해에서 분산 최소 (5종목이면 31개 → 정확해)"""
    n = cov.shape[0]
    best, bw = None, None
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            c = cov[np.ix_(S, S)]
            try:
                x = np.linalg.solve(c, np.ones(k))
            except np.linalg.LinAlgError:
                continue
            if x.sum() <= 0:
                continue
            x = x / x.sum()
            if (x < -1e-12).any():
                continue
            w = np.zeros(n); w[list(S)] = np.clip(x, 0, None)
            v = w @ cov @ w
            if best is None or v < best - 1e-18:
                best, bw = v, w
    return bw / bw.sum()


def weights(dret, top, p, method):
    r = dret.loc[(p - 2).start_time:p.end_time, top].dropna()
    if method == "equal":
        return {t: 1 / len(top) for t in top}
    if method == "inv":
        inv = 1 / r.std(ddof=1)
        return (inv / inv.sum()).to_dict()
    cov = r.cov().values
    w = erc_weights(cov) if method == "erc" else minvar_weights(cov)
    return dict(zip(top, w))


def run(adj, months, method):
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    ra_l, mix_l, hold, conc = [], [], [], []
    for j in range(len(months) - 1):
        p, p1 = months[j], months[j + 1]
        i = mpx.index.get_loc(p)
        s = (mpx.iloc[i][T12] / mpx.iloc[i - 6][T12] - 1).dropna()
        top = sorted(s.index, key=lambda x: (-s[x], T12.index(x)))[:5]
        nxt = mpx.loc[p1] / mpx.loc[p] - 1
        wa = weights(dret, top, p, method)
        ra = sum(w * nxt[t] for t, w in wa.items())
        wd = {}
        for t in top:
            k = t if s[t] >= 0 else CASH
            wd[k] = wd.get(k, 0) + 0.2
        rd = sum(w * nxt[t] for t, w in wd.items())
        ra_l.append(float(ra)); mix_l.append(float((ra + rd) / 2)); hold.append(str(p1))
        conc.append(max(wa.values()))
    return ra_l, mix_l, hold, float(np.mean(conc))


def pack(r, hold):
    st = {"full": stats(r), "5y": stats(r[-60:])}
    if len(r) >= 120:
        st["10y"] = stats(r[-120:])
    yr = {}
    for x, m in zip(r, hold):
        yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
    return {"period": [hold[0], hold[-1]], "stats": st, "yearly": {y: round(v - 1, 4) for y, v in yr.items()}}


def main():
    import yfinance as yf
    need = sorted(set(T12) | {"AQMIX", "BIL", "DBC", "BTC-USD"})
    raw = None
    for _ in range(3):
        raw = yf.download(need, start="2009-06-01", auto_adjust=True, progress=False, threads=False)["Close"]
        if all(raw[t].notna().any() for t in need):
            break
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
        for m in METHODS:
            ra, mix, hold, conc = run(df, ms, m)
            res[f"{tag}_{m}_aaa"] = pack(ra, hold)
            res[f"{tag}_{m}_mix"] = pack(mix, hold)
            res[f"{tag}_{m}_aaa"]["avg_max_weight"] = round(conc, 3)
            for kind in ("aaa", "mix"):
                st = res[f"{tag}_{m}_{kind}"]["stats"]
                print(tag, m, kind, " | ".join(f"{k} {v['cagr']:.2%}/{v['mdd']:.2%}" for k, v in st.items()), flush=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/aaamethods.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
