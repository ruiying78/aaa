#!/usr/bin/env python3
"""개선안 3종 백테스트 (일별 시뮬레이션, 현행 50:50 = AAA(ERC)+DM, 6M 모멘텀 Top5, 12종목)
 1) 리밸런싱 날짜 분산: 자금을 N등분해 서로 다른 날(월말/월중 등) 리밸런싱
 3) 카나리아: EEM·AGG 13612W 모멘텀이 음수면 방어자산(IEF/TLT/GLD/DBMF/SGOV 중 1위)으로 전환
 2) 매매 최소화 밴드: 목표와 현재 비중 차이가 밴드 이내면 그대로 두고 차이는 SGOV로 흡수
 A) 2022-05~2026-10 실제 상품  B) 2015-04~2026-10 장기 연결
"""
import json
import math
import sys

import numpy as np
import pandas as pd

from erc import erc_weights, stats
from stack import T12, CASH, LAST, splice

DEF = ["IEF", "TLT", "GLD", "DBMF", "SGOV"]
CANARY = ["EEM", "AGG"]


def sig(px, dret, d, canary=None):
    """d 종가 기준 목표비중 (dict)"""
    p0 = px.loc[:d - pd.DateOffset(months=6)].iloc[-1]
    s = (px.loc[d, T12] / p0[T12] - 1).dropna()
    top = sorted(s.index, key=lambda x: (-s[x], T12.index(x)))[:5]
    win = dret.loc[(d - pd.DateOffset(months=3)) + pd.Timedelta(days=1):d, top].dropna()
    wa = dict(zip(top, erc_weights(win.cov().values)))
    w = {t: 0.5 * v for t, v in wa.items()}
    for t in top:
        k = t if s[t] >= 0 else CASH
        w[k] = w.get(k, 0) + 0.1
    if canary:
        mp = px.loc[:d]
        def m13612(t):
            c = mp[t].iloc[-1]
            r = [c / mp[t].loc[:d - pd.DateOffset(months=k)].iloc[-1] - 1 for k in (1, 3, 6, 12)]
            return (12 * r[0] + 4 * r[1] + 2 * r[2] + r[3]) / 19
        bad = sum(m13612(t) < 0 for t in CANARY)
        frac = min(1.0, bad / len(CANARY)) if canary == "partial" else (1.0 if bad else 0.0)
        if frac > 0:
            ds = {t: px.loc[d, t] / p0[t] - 1 for t in DEF}
            best = max(ds, key=ds.get)
            if ds[best] < ds["SGOV"]:
                best = "SGOV"
            w = {t: v * (1 - frac) for t, v in w.items()}
            w[best] = w.get(best, 0) + frac
    return w


def schedule(days, kind):
    """리밸런싱 날짜 목록"""
    s = pd.Series(days, index=days)
    out = []
    for _, g in s.groupby(days.to_period("M")):
        if kind == "end":
            out.append(g.iloc[-1])
        else:
            sub = g[g.dt.day <= kind]
            if len(sub):
                out.append(sub.iloc[-1])
    return out


def simulate(px, dret, start, end, dates, canary=None, band=0.0):
    """한 트랜치: start 종가에 매수 후 dates마다 리밸런싱. (일별 평가액, 회전율 합, 매매 건수)"""
    days = px.loc[start:end].index
    rb = set(d for d in dates if start < d <= end)
    w = sig(px, dret, start, canary)
    hold = {t: v for t, v in w.items()}          # 평가액(자본 1 기준)
    eq, turn, ntr = [], 0.0, 0
    prev = start
    for d in days:
        if d != start:
            for t in hold:
                hold[t] *= px.loc[d, t] / px.loc[prev, t]
        tot = sum(hold.values())
        if d in rb:
            cur = {t: v / tot for t, v in hold.items()}
            tgt = sig(px, dret, d, canary)
            keys = set(cur) | set(tgt)
            new = {}
            for t in keys:
                c, g = cur.get(t, 0.0), tgt.get(t, 0.0)
                new[t] = c if (band and abs(g - c) < band and g > 0 and c > 0) else g
            new[CASH] = new.get(CASH, 0.0) + (1 - sum(new.values()))
            if new[CASH] < 0:
                new = dict(tgt)
            turn += sum(abs(new.get(t, 0) - cur.get(t, 0)) for t in keys | set(new)) / 2
            ntr += sum(abs(new.get(t, 0) - cur.get(t, 0)) > 1e-9 for t in keys | set(new))
            hold = {t: v * tot for t, v in new.items() if v > 1e-12}
        eq.append(tot)
        prev = d
    return pd.Series(eq, index=days), turn, ntr


def monthly(eq):
    m = eq.groupby(eq.index.to_period("M")).last()
    return (m / m.shift(1) - 1).dropna().tolist(), [str(p) for p in m.index[1:]]


def pack(r, hold, extra=None):
    st = {"full": stats(r), "5y": stats(r[-60:])}
    if len(r) >= 120:
        st["10y"] = stats(r[-120:])
    yr = {}
    for x, m in zip(r, hold):
        yr.setdefault(m[:4], 1.0); yr[m[:4]] *= 1 + x
    o = {"period": [hold[0], hold[-1]], "stats": st, "yearly": {y: round(v - 1, 4) for y, v in yr.items()}}
    if extra:
        o.update(extra)
    return o


def main():
    import yfinance as yf
    need = sorted(set(T12) | {"AQMIX", "BIL", "DBC", "BTC-USD", "AGG"})
    raw = None
    for _ in range(3):
        raw = yf.download(need, start="2009-06-01", auto_adjust=True, progress=False, threads=False)["Close"]
        if all(raw[t].notna().any() for t in need):
            break
    days = raw["SPY"].dropna().index
    raw = raw.reindex(days).ffill(limit=5)
    A = raw[T12 + ["AGG"]].copy()
    B = raw[T12 + ["AGG"]].copy()
    B["DBMF"] = splice(raw, "DBMF", "AQMIX"); B["SGOV"] = splice(raw, "SGOV", "BIL")
    B["PDBC"] = splice(raw, "PDBC", "DBC"); B["BITO"] = raw["BTC-USD"]
    res = {}
    for tag, px, s0 in (("A", A, "2022-04"), ("B", B, None)):
        px = px.dropna(how="all")
        dret = px.pct_change()
        if s0 is None:
            first = max(px[t].first_valid_index() for t in T12).to_period("M") + 6
            s0 = str(first)
        ends = schedule(px.index, "end")
        start = [d for d in ends if d.to_period("M") == pd.Period(s0)][0]
        end = [d for d in ends if d.to_period("M") == LAST][0]
        yrs = (end - start).days / 365.25
        runs = {
            "base": [("end", None, 0.0)],
            "tr2": [("end", None, 0.0), (15, None, 0.0)],
            "tr4": [("end", None, 0.0), (7, None, 0.0), (15, None, 0.0), (22, None, 0.0)],
            "can_full": [("end", "full", 0.0)],
            "can_part": [("end", "partial", 0.0)],
            "band2": [("end", None, 0.02)],
            "band3": [("end", None, 0.03)],
            "band5": [("end", None, 0.05)],
        }
        for name, trs in runs.items():
            eqs, turn, ntr = [], 0.0, 0
            for kind, can, band in trs:
                e, tv, n = simulate(px, dret, start, end, schedule(px.index, kind), can, band)
                eqs.append(e); turn += tv / len(trs); ntr += n
            eq = sum(eqs) / len(eqs)
            r, hold = monthly(eq)
            res[f"{tag}_{name}"] = pack(r, hold, {"turnover_per_year": round(turn / yrs, 3),
                                                  "trades_per_year": round(ntr / yrs, 1)})
            st = res[f"{tag}_{name}"]["stats"]
            print(tag, name, " | ".join(f"{k} {v['cagr']:.2%}/{v['mdd']:.2%}" for k, v in st.items()),
                  "turn", res[f"{tag}_{name}"]["turnover_per_year"], "trades", res[f"{tag}_{name}"]["trades_per_year"], flush=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/improve.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
