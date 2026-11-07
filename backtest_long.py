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


def run(adj, raw, tickers, mode, first_sig):
    sig_px = adj if mode == "pv" else raw
    me = adj.groupby(adj.index.to_period("M")).tail(1).index
    me = [d for d in me if first_sig <= d.to_period("M") <= pd.Period(END)]
    rows = []
    for i in range(len(me) - 1):
        t, t1 = me[i], me[i + 1]
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
        for mode in ["pv", "excel"]:
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
