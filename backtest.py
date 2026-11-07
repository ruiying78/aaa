#!/usr/bin/env python3
"""AAA 전략 백테스트: PV 방식 vs 엑셀 방식 (월말 신호 → 월말 종가 매매, 성과는 배당 포함 총수익률)"""
import json
import math
import os
import sys

import pandas as pd

T10 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "SGOV", "DBMF", "BITO"]
T12 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "IEF", "TLT", "SGOV", "DBMF", "BITO"]
START, END = "2022-04", "2026-10"   # 첫 신호 2022-04말 → 성과 2022-05 ~ 2026-10
ANNUAL = math.sqrt(252)


def target_vol(pv):
    if pv <= 0.18:
        return 1.0
    if pv >= 0.22:
        return 0.15
    return 1.0 - (pv - 0.18) * (0.85 / 0.04)


def load():
    import yfinance as yf
    adj = yf.download(T12, start="2021-09-01", auto_adjust=True, progress=False)["Close"]
    raw = yf.download(T12, start="2021-09-01", auto_adjust=False, progress=False)["Close"]
    return adj[T12].sort_index(), raw[T12].sort_index()


def signal(px, tickers, t, mode):
    """t 월말 기준 (top, weights dict, cash)"""
    sub = px.loc[:t, tickers].dropna(how="all")
    m = sub.groupby(sub.index.to_period("M")).tail(1)
    cur = m.iloc[-1]
    if mode == "pv":
        score = cur / m.iloc[-7] - 1
        elig = score.dropna()
    else:
        score = sum((cur / m.iloc[-1 - k] - 1) for k in range(1, 7)) / 6
        elig = score.dropna()
        elig = elig[elig >= -0.02]
    top = sorted(elig.index, key=lambda x: (-elig[x], tickers.index(x)))[:5]
    if not top:
        return [], {}, 1.0
    endp = t.to_period("M")
    vs = (endp - 2).start_time
    rets = sub[top].pct_change().dropna()
    rets = rets[(rets.index >= vs) & (rets.index <= endp.end_time)]
    vol = rets.std(ddof=1)
    inv = vol.apply(lambda v: 0 if v == 0 else 1 / v)
    w = inv / inv.sum()
    k = 1.0
    if mode == "excel":
        pvv = (rets * w).sum(axis=1).std(ddof=1) * ANNUAL
        k = min(1.0, target_vol(pvv) / pvv) if pvv > 0 else 1.0
    wd = {x: float(w[x] * k) for x in top}
    return top, wd, 1 - sum(wd.values())


def run(adj, raw, tickers, mode):
    sig_px = adj if mode == "pv" else raw
    me = adj.groupby(adj.index.to_period("M")).tail(1).index
    me = [d for d in me if START <= str(d.to_period("M")) <= END]
    rows, bal = [], 1.0
    for i in range(len(me) - 1):
        t, t1 = me[i], me[i + 1]
        top, w, cash = signal(sig_px, tickers, t, mode)
        r = sum(w[x] * (adj.at[t1, x] / adj.at[t, x] - 1) for x in top)
        bal *= 1 + r
        rows.append({"month": str(t1.to_period("M")), "ret": r, "bal": bal,
                     "w": {x: round(w[x], 4) for x in top}, "cash": round(cash, 4)})
    rets = pd.Series([x["ret"] for x in rows])
    n = len(rets)
    cagr = bal ** (12 / n) - 1
    eq = (1 + rets).cumprod()
    mdd = float((eq / eq.cummax().clip(lower=1) - 1).min())
    sd = float(rets.std(ddof=1) * math.sqrt(12))
    return {"cagr": cagr, "mdd": mdd, "stdev": sd, "end": bal * 10000, "months": n,
            "best": float(rets.max()), "worst": float(rets.min()), "rows": rows}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/backtest.json"
    adj, raw = load()
    res = {
        "A_PV_10": run(adj, raw, T10, "pv"),
        "B_PV_12": run(adj, raw, T12, "pv"),
        "C_Excel_12": run(adj, raw, T12, "excel"),
        "D_Excel_10": run(adj, raw, T10, "excel"),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    for k, v in res.items():
        print(f"{k:12s} CAGR {v['cagr']:.2%}  MDD {v['mdd']:.2%}  SD {v['stdev']:.2%}  End ${v['end']:,.0f}")


if __name__ == "__main__":
    main()
