#!/usr/bin/env python3
"""장기 백테스트: DBMF·BITO 제외, PDBC→DBC, SGOV→BIL. PV 방식 vs 엑셀 방식, 전체기간 + 최근 5년"""
import json
import math
import sys

import pandas as pd

from backtest import signal

U8 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "DBC", "BIL"]
U10 = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "DBC", "IEF", "TLT", "BIL"]
ALL = U10
END = "2026-10"


def load():
    import yfinance as yf
    adj = yf.download(ALL, start="2004-01-01", auto_adjust=True, progress=False)["Close"][ALL].sort_index()
    raw = yf.download(ALL, start="2004-01-01", auto_adjust=False, progress=False)["Close"][ALL].sort_index()
    return adj, raw


def stats(rets):
    rets = pd.Series(rets)
    n = len(rets)
    eq = (1 + rets).cumprod()
    cagr = eq.iloc[-1] ** (12 / n) - 1
    mdd = float((eq / eq.cummax().clip(lower=1) - 1).min())
    sd = float(rets.std(ddof=1) * math.sqrt(12))
    return {"cagr": float(cagr), "mdd": mdd, "stdev": sd, "months": n,
            "worst_month": float(rets.min()), "growth": float(eq.iloc[-1])}


def dm_signal(px, tickers, t, base):
    """듀얼 모멘텀: 상위 5종목 각 20%, 점수 < 0 이면 그 몫을 BIL로"""
    sub = px.loc[:t, tickers].dropna(how="all")
    m = sub.groupby(sub.index.to_period("M")).tail(1)
    cur = m.iloc[-1]
    if base == "pv":
        score = (cur / m.iloc[-7] - 1).dropna()
    else:
        score = (sum((cur / m.iloc[-1 - k] - 1) for k in range(1, 7)) / 6).dropna()
    top = sorted(score.index, key=lambda x: (-score[x], tickers.index(x)))[:5]
    w = {}
    for x in top:
        k = x if score[x] >= 0 else "BIL"
        w[k] = w.get(k, 0) + 0.2
    return list(w), w, 0.0


def run(adj, raw, tickers, mode, first_sig):
    sig_px = adj if mode in ("pv", "dm_pv") else raw
    me = adj.groupby(adj.index.to_period("M")).tail(1).index
    me = [d for d in me if first_sig <= d.to_period("M") <= pd.Period(END)]
    rows = []
    for i in range(len(me) - 1):
        t, t1 = me[i], me[i + 1]
        if mode.startswith("dm_"):
            top, w, cash = dm_signal(sig_px, tickers, t, mode[3:])
        else:
            top, w, cash = signal(sig_px, tickers, t, mode)
        r = sum(w[x] * (adj.at[t1, x] / adj.at[t, x] - 1) for x in top)
        rows.append({"month": str(t1.to_period("M")), "ret": float(r), "top": top})
    return rows


def yearly(rows):
    y = {}
    for r in rows:
        y.setdefault(r["month"][:4], 1.0)
        y[r["month"][:4]] *= 1 + r["ret"]
    return {k: round(v - 1, 4) for k, v in y.items()}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/backtest_long.json"
    adj, raw = load()
    first_valid = max(adj[t].first_valid_index() for t in ALL)
    first_sig = first_valid.to_period("M") + 7          # 6개월 모멘텀 + 첫 달 온전한 데이터
    res = {"first_holding_month": str(first_sig + 1)}
    for name, tk in [("U10", U10)]:
        for mode in ["pv", "excel", "dm_pv", "dm_excel"]:
            rows = run(adj, raw, tk, mode, first_sig)
            rets = [r["ret"] for r in rows]
            res[f"{mode}_{name}"] = {"full": stats(rets), "last5y": stats(rets[-60:]),
                                     "period": [rows[0]["month"], rows[-1]["month"]],
                                     "yearly": yearly(rows)}
    # 벤치마크: SPY 보유
    me = adj.groupby(adj.index.to_period("M")).tail(1)
    me = me[(me.index.to_period("M") >= first_sig) & (me.index.to_period("M") <= pd.Period(END))]
    spy = me["SPY"].pct_change().dropna().tolist()
    res["SPY_buyhold"] = {"full": stats(spy), "last5y": stats(spy[-60:]),
                          "yearly": {str(k): round(v - 1, 4) for k, v in
                                     (1 + me["SPY"].pct_change().dropna()).groupby(me.index[1:].year).prod().items()}}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    for k, v in res.items():
        if isinstance(v, dict) and "full" in v:
            print(f"{k:12s} FULL CAGR {v['full']['cagr']:.2%} MDD {v['full']['mdd']:.2%} | 5Y CAGR {v['last5y']['cagr']:.2%} MDD {v['last5y']['mdd']:.2%}")


if __name__ == "__main__":
    main()
