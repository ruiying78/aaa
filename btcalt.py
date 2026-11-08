#!/usr/bin/env python3
"""비트코인 슬롯 대체 비교: BITO / BTC-USD / COIN / MSTR / 제외(11종목)
 현행 전략 그대로 (PV 6개월 Top5, AAA=ERC, DM 각 20% 음수→SGOV, 50:50), 2022-05 ~ 2026-10
"""
import json
import sys

import pandas as pd

from erc import aaa_weights, stats

BASE = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "IEF", "TLT", "SGOV", "DBMF"]
ALTS = {"BITO": "BITO", "BTC현물": "BTC-USD", "COIN": "COIN", "MSTR": "MSTR", "제외": None}
CASH = "SGOV"
FIRST, LAST = pd.Period("2022-04"), pd.Period("2026-10")


def main():
    import yfinance as yf
    need = BASE + [t for t in ALTS.values() if t]
    raw = yf.download(need, start="2021-06-01", auto_adjust=True, progress=False)["Close"]
    days = raw["SPY"].dropna().index
    adj = raw.reindex(days).ffill(limit=5)
    mpx = adj.groupby(adj.index.to_period("M")).tail(1)
    mpx.index = mpx.index.to_period("M")
    dret = adj.pct_change()
    months = [p for p in mpx.index if FIRST <= p <= LAST]
    res = {"period": None, "stats": {}, "yearly": {}, "slot": {}}
    for name, tk in ALTS.items():
        uni = BASE + ([tk] if tk else [])
        r, hold, w_aaa, w_dm, n_in = [], [], [], [], 0
        for j in range(len(months) - 1):
            p, p1 = months[j], months[j + 1]
            i = mpx.index.get_loc(p)
            s = (mpx.iloc[i][uni] / mpx.iloc[i - 6][uni] - 1).dropna()
            top = sorted(s.index, key=lambda x: (-s[x], uni.index(x)))[:5]
            nxt = mpx.loc[p1] / mpx.loc[p] - 1
            wa = aaa_weights(dret, top, p, "erc")
            ra = sum(w * nxt[t] for t, w in wa.items())
            wd = {}
            for t in top:
                k = t if s[t] >= 0 else CASH
                wd[k] = wd.get(k, 0) + 0.2
            rd = sum(w * nxt[t] for t, w in wd.items())
            r.append(float((ra + rd) / 2)); hold.append(str(p1))
            if tk and tk in top:
                n_in += 1
                w_aaa.append(wa[tk]); w_dm.append(wd.get(tk, 0))
        res["period"] = [hold[0], hold[-1]]
        res["stats"][name] = stats(r)
        yr = {}
        for x, m in zip(r, hold):
            yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
        res["yearly"][name] = {y: round(v - 1, 4) for y, v in yr.items()}
        res["slot"][name] = {"months_in_top5": n_in, "of": len(r),
                             "avg_final_w_when_in": round(sum((a + d) / 2 for a, d in zip(w_aaa, w_dm)) / n_in, 4) if n_in else 0}
        st = res["stats"][name]
        print(name, f"CAGR {st['cagr']:.2%} MDD {st['mdd']:.2%} SD {st['sd']:.2%}", res["slot"][name])
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/btcalt.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
