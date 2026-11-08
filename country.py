#!/usr/bin/env python3
"""VGK→선진국 개별국가 모멘텀 1위, EEM→신흥국 개별국가 모멘텀 1위로 대체 시 50:50 전략 성과 비교"""
import json
import sys

import pandas as pd

from erc import aaa_weights, split_stats

BASE = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "DBC", "IEF", "TLT", "BIL"]
DEV = ["EWA", "EWC", "EWG", "EWQ", "EWU", "EWL", "EWI", "EWP", "EWD", "EWN", "EWH", "EWS", "EWK", "EWO",
       "EDEN", "ENZL", "EIS", "NORW"]
EMG = ["EWZ", "EWY", "EWT", "EWW", "EZA", "FXI", "EWM", "INDA", "TUR", "THD", "EIDO", "EPOL", "ECH", "EPHE"]
ALL = sorted(set(BASE) | set(DEV) | set(EMG))


def mom(mpx, i, t):
    a, b = mpx.iloc[i][t], mpx.iloc[i - 6][t]
    return a / b - 1 if pd.notna(a) and pd.notna(b) else None


def best_of(mpx, dret, pool, p, i):
    """그 시점에 6개월 모멘텀 + 3개월 변동성 데이터가 모두 있는 후보 중 1위"""
    vs = (p - 2).start_time
    cands = {}
    for t in pool:
        s = mom(mpx, i, t)
        if s is None:
            continue
        r = dret.loc[vs:p.end_time, t]
        if r.notna().sum() < 40:
            continue
        cands[t] = s
    return max(cands, key=cands.get) if cands else None


def run(mpx, dret, months, mode):
    mix, hold, picks = [], [], []
    for j in range(len(months) - 1):
        p, p1 = months[j], months[j + 1]
        i = mpx.index.get_loc(p)
        uni = list(BASE)
        pk = {}
        if mode in ("dev", "both"):
            b = best_of(mpx, dret, DEV, p, i)
            if b:
                uni[uni.index("VGK")] = b; pk["DEV"] = b
        if mode in ("em", "both"):
            b = best_of(mpx, dret, EMG, p, i)
            if b:
                uni[uni.index("EEM")] = b; pk["EM"] = b
        s = {t: mom(mpx, i, t) for t in uni}
        s = {t: v for t, v in s.items() if v is not None}
        top = sorted(s, key=lambda x: (-s[x], uni.index(x)))[:5]
        if mode.startswith("g"):  # 게이트: VGK/EEM이 Top5에 들었을 때만 국가 1위로 교체
            for base_t, pool, key, on in (("VGK", DEV, "DEV", mode in ("gdev", "gboth")),
                                          ("EEM", EMG, "EM", mode in ("gem", "gboth"))):
                if on and base_t in top:
                    b = best_of(mpx, dret, pool, p, i)
                    if b:
                        top[top.index(base_t)] = b; s[b] = mom(mpx, i, b); pk[key] = b
        nxt = mpx.loc[p1] / mpx.loc[p] - 1
        wa = aaa_weights(dret, top, p, "erc")
        ra = sum(w * nxt[t] for t, w in wa.items())
        wd = {}
        for t in top:
            k = t if s[t] >= 0 else "BIL"
            wd[k] = wd.get(k, 0) + 0.2
        rd = sum(w * nxt[t] for t, w in wd.items())
        mix.append((ra + rd) / 2); hold.append(str(p1))
        picks.append({"hold": str(p1), **pk, "in_top": [t for t in top if t in pk.values()]})
    return mix, hold, picks


def main():
    import yfinance as yf
    adj = yf.download(ALL, start="2004-01-01", auto_adjust=True, progress=False)["Close"]
    adj = adj[[t for t in ALL if t in adj.columns]].sort_index()
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    first = max(adj[t].first_valid_index() for t in BASE).to_period("M") + 7
    months = [p for p in mpx.index if first <= p <= pd.Period("2026-10")]
    res = {"available": {t: str(adj[t].first_valid_index().date()) for t in adj.columns if adj[t].notna().any()}}
    for mode in ["base", "dev", "em", "both", "gdev", "gem", "gboth"]:
        r, h, pk = run(mpx, dret, months, mode)
        yr = {}
        for x, m in zip(r, h):
            yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
        res[mode] = {**split_stats(r, h), "yearly": {k: round(v - 1, 4) for k, v in yr.items()},
                     "picks": pk, "period": [h[0], h[-1]]}
        f = res[mode]["full"]
        print(mode, f"CAGR {f['cagr']:.2%} MDD {f['mdd']:.2%}")
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/country.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
