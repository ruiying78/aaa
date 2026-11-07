#!/usr/bin/env python3
"""
국장 AAA v6 자동 리밸런싱
 - stooq(1순위) / Yahoo Finance(백업)에서 종가를 자동으로 받아
 - 엑셀(AAA 시트) 수식과 동일한 로직으로 Top5 + 리스크패리티 + 변동성조정 비중을 산출하고
 - 결과를 웹페이지(report.html)로 만들어 띄웁니다.

설정은 config.json 한 곳에서만 합니다 (투자금, 보유수량 등). 실행할 때 입력할 것은 없습니다.

  python aaa_rebalance.py                 # 계산 + report.html 생성 + 브라우저로 열기
  python aaa_rebalance.py --no-open       # 브라우저 안 열기 (GitHub 자동화용)
  python aaa_rebalance.py --excel 내전략.xlsx   # 엑셀 M_Price/D_Price도 최신 데이터로 채운 사본 저장
"""
import argparse
import html
import io
import json
import math
import os
import sys
import webbrowser
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))

# ───────────── 전략 파라미터 (엑셀과 동일) ─────────────
UNIVERSE = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "IEF", "TLT", "SGOV", "DBMF", "BITO"]
LOOKBACK_MONTHS = 6
# 엑셀 실제 수식은 1~6M 각 1/6 (동일가중). 상단 설명의 18.5185%/14.8148%를 쓰려면 아래 주석 교체
MOM_WEIGHTS = [1 / 6] * 6
# MOM_WEIGHTS = [0.185185, 0.185185, 0.185185, 0.148148, 0.148148, 0.148148]
CUTOFF = -0.02          # 모멘텀 컷오프 (H3)
TOP_N = 5
VOL_MONTHS = 3          # 변동성: 직전 완료된 달력상 3개월 (D월 1일 ~ D+2월 말일)
ANNUAL = math.sqrt(252)


def target_vol(port_vol):
    """엑셀 P13 수식 그대로"""
    if port_vol <= 0.18:
        return 1.0
    if port_vol >= 0.22:
        return 0.15
    return 1.0 - (port_vol - 0.18) * (0.85 / 0.04)


# ───────────── 데이터 수집 ─────────────
def fetch_stooq(symbol, start):
    url = "https://stooq.com/q/d/l/"
    params = {"s": symbol, "i": "d", "d1": start.strftime("%Y%m%d"), "d2": date.today().strftime("%Y%m%d")}
    r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    txt = r.text.strip()
    if not txt.lower().startswith("date"):
        raise ValueError(f"응답 이상: {txt[:60]}")
    df = pd.read_csv(io.StringIO(txt), parse_dates=["Date"]).set_index("Date")
    return df["Close"].astype(float)


def fetch_yahoo(tickers, start):
    import yfinance as yf
    df = yf.download(tickers, start=start, auto_adjust=False, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df


def load_prices(tickers, start):
    closes, failed, source = {}, [], {}
    for t in tickers:
        try:
            closes[t] = fetch_stooq(f"{t.lower()}.us", start)
            source[t] = "stooq"
            print(f"  [stooq] {t}: 최종 {closes[t].index[-1].date()} {closes[t].iloc[-1]}")
        except Exception as e:
            print(f"  [stooq 실패] {t}: {e}")
            failed.append(t)
    if failed:
        print(f"  → Yahoo로 재시도: {failed}")
        ydf = fetch_yahoo(failed, start)
        for t in failed:
            closes[t] = ydf[t].dropna()
            source[t] = "Yahoo"
            print(f"  [yahoo] {t}: 최종 {closes[t].index[-1].date()}")
    df = pd.DataFrame(closes)[tickers].sort_index()
    df.attrs["source"] = source
    return df


def load_fx(start):
    try:
        return float(fetch_stooq("usdkrw", start).iloc[-1])
    except Exception:
        try:
            import yfinance as yf
            s = yf.download("KRW=X", start=start, progress=False)["Close"]
            return float(s.squeeze().dropna().iloc[-1])
        except Exception:
            return None


# ───────────── 엑셀 로직 ─────────────
def to_monthly(daily):
    """월말 종가 + 마지막 행은 최신 종가 (엑셀 M_Price 입력 방식과 동일)"""
    return daily.groupby(daily.index.to_period("M")).tail(1)


def momentum_scores(monthly):
    cur = monthly.iloc[-1]
    return sum(w * (cur / monthly.iloc[-1 - k] - 1) for k, w in zip(range(1, LOOKBACK_MONTHS + 1), MOM_WEIGHTS))


def select_top(score):
    elig = score[score >= CUTOFF]
    order = sorted(elig.index, key=lambda t: (-elig[t], UNIVERSE.index(t)))  # 동점 시 목록 위쪽 우선
    return order[:TOP_N]


def vol_window(daily, today=None):
    """실행일(베이징 날짜) 기준 달력상 VOL_MONTHS개월.
    실행일이 말일이면 그 달까지 포함, 아니면 직전 달까지.
    예) 11/7 → 8/1~10/31, 11/30 → 9/1~11/30, 12/1 → 9/1~11/30"""
    if today is None:
        today = datetime.now(timezone(timedelta(hours=8))).date()
    today = pd.Timestamp(today)
    end_month = today.to_period("M")
    if today.day != today.days_in_month:
        end_month -= 1
    start = (end_month - (VOL_MONTHS - 1)).start_time
    end = end_month.end_time
    return start, end


def risk_parity(daily, top):
    start, end = vol_window(daily)
    rets = daily[top].pct_change().dropna()
    rets = rets[(rets.index >= start) & (rets.index <= end)]
    vol = rets.std(ddof=1)
    inv = vol.apply(lambda v: 0 if v == 0 else 1 / v)
    w = inv / inv.sum()
    port = (rets * w).sum(axis=1)
    port_vol = port.std(ddof=1) * ANNUAL
    tv = target_vol(port_vol)
    k = min(1.0, tv / port_vol) if port_vol > 0 else 1.0
    return vol, w, port_vol, tv, k, (rets.index[0], rets.index[-1], len(rets))


def compute(daily):
    monthly = to_monthly(daily)
    if len(monthly) < LOOKBACK_MONTHS + 1:
        raise SystemExit("월 데이터가 7개월 미만입니다.")
    score = momentum_scores(monthly)
    top = select_top(score)
    if not top:
        return dict(monthly=monthly, score=score, top=[], vol=pd.Series(dtype=float), w=pd.Series(dtype=float),
                    port_vol=0.0, target=1.0, k=1.0, final=pd.Series(dtype=float), cash=1.0, vwin=None)
    vol, w, port_vol, tv, k, vwin = risk_parity(daily, top)
    final = w * k
    return dict(monthly=monthly, score=score, top=top, vol=vol, w=w,
                port_vol=port_vol, target=tv, k=k, final=final, cash=1 - final.sum(), vwin=vwin)


def monthly_signals(daily, n=12):
    """지난 n개월 각 월말 기준 신호 (월말 리밸런싱 → 다음 달 보유)"""
    out = []
    month_ends = daily.groupby(daily.index.to_period("M")).tail(1).index
    month_ends = [d for d in month_ends if d.to_period("M") < daily.index[-1].to_period("M")]
    for me in month_ends[-n:]:
        sub = daily[daily.index <= me]
        if len(to_monthly(sub)) < LOOKBACK_MONTHS + 1:
            continue
        cal_end = me.to_period("M").end_time.normalize()
        score = momentum_scores(to_monthly(sub))
        top = select_top(score)
        row = {"rebal": me, "hold": (me.to_period("M") + 1).strftime("%Y-%m"), "top": top, "w": {}, "cash": 1.0}
        if top:
            rets = sub[top].pct_change().dropna()
            vs, ve = vol_window(sub, today=cal_end)
            rets = rets[(rets.index >= vs) & (rets.index <= ve)]
            vol = rets.std(ddof=1)
            inv = vol.apply(lambda v: 0 if v == 0 else 1 / v)
            w = inv / inv.sum()
            pv = (rets * w).sum(axis=1).std(ddof=1) * ANNUAL
            k = min(1.0, target_vol(pv) / pv) if pv > 0 else 1.0
            row["w"] = {t: float(w[t] * k) for t in top}
            row["cash"] = 1 - sum(row["w"].values())
        out.append(row)
    return out


def order_table(res, daily, capital, fx, holdings):
    last = daily.iloc[-1]
    rows = []
    for t in res["top"]:
        row = {"ticker": t, "vol": res["vol"][t], "rp": res["w"][t], "final": res["final"][t], "price": last[t]}
        if capital and fx:
            amt = capital * res["final"][t] / fx
            row["amount_usd"] = amt
            row["target_qty"] = math.floor(amt / last[t])
            if holdings:
                row["held"] = int(holdings.get(t, 0))
                row["trade"] = row["target_qty"] - row["held"]
        rows.append(row)
    if capital and fx and holdings:
        for t, q in holdings.items():
            if t not in res["top"] and int(q):
                rows.append({"ticker": t, "final": 0.0, "price": last.get(t, float("nan")),
                             "target_qty": 0, "held": int(q), "trade": -int(q)})
    return rows


# ───────────── 이력 ─────────────
def update_history(path, asof, res):
    hist = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
    hist = [h for h in hist if h["date"] != asof]
    hist.append({"date": asof, "top": res["top"],
                 "weights": {t: round(float(res["final"][t]), 4) for t in res["top"]}})
    hist = sorted(hist, key=lambda h: h["date"])[-60:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)
    return hist


# ───────────── HTML 리포트 ─────────────
def pct(v, d=2):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:.{d}f}%"


def build_html(res, daily, rows, hist, capital, fx, msig=None):
    asof = daily.index[-1].strftime("%Y-%m-%d")
    kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    src = daily.attrs.get("source", {})
    src_txt = ", ".join(sorted(set(src.values()))) or "-"
    e = html.escape

    # Top5 비중 막대
    cards = ""
    for t in res["top"]:
        w = res["final"][t]
        cards += f"""<div class="wrow"><span class="tk">{e(t)}</span>
<div class="bar"><i style="width:{w*100:.1f}%"></i></div><span class="num">{pct(w,1)}</span></div>"""
    if res["cash"] > 0.0005:
        cards += f"""<div class="wrow"><span class="tk muted">현금</span>
<div class="bar"><i class="cash" style="width:{res['cash']*100:.1f}%"></i></div><span class="num">{pct(res['cash'],1)}</span></div>"""

    # 모멘텀 표
    smax = max(abs(res["score"]).max(), 0.01)
    mom = ""
    for t in sorted(UNIVERSE, key=lambda t: -res["score"][t]):
        s = res["score"][t]
        cls = "sel" if t in res["top"] else ("out" if s < CUTOFF else "")
        tag = "선정" if t in res["top"] else ("컷오프" if s < CUTOFF else "")
        width = abs(s) / smax * 50
        side = f"left:50%;width:{width:.1f}%" if s >= 0 else f"left:{50-width:.1f}%;width:{width:.1f}%"
        mom += f"""<tr class="{cls}"><td class="tk">{e(t)}</td><td class="num">{s*100:+.2f}%</td>
<td class="mbar"><div><b style="{side}" class="{'neg' if s<0 else 'pos'}"></b><u></u></div></td><td class="tag">{tag}</td></tr>"""

    # 주문표
    orders = ""
    if capital and fx:
        has_h = any("held" in r for r in rows)
        head = "<th>종목</th><th>최종비중</th><th>현재가($)</th><th>목표금액($)</th><th>목표수량</th>"
        if has_h:
            head += "<th>보유</th><th>매매</th>"
        body = ""
        for r in rows:
            body += f"<tr><td class='tk'>{e(r['ticker'])}</td><td class='num'>{pct(r.get('final'),1)}</td>" \
                    f"<td class='num'>{r['price']:,.2f}</td><td class='num'>{r.get('amount_usd',0):,.0f}</td>" \
                    f"<td class='num'>{r.get('target_qty',0):,}</td>"
            if has_h:
                tr = r.get("trade", 0)
                c = "buy" if tr > 0 else ("sell" if tr < 0 else "")
                body += f"<td class='num'>{r.get('held',0):,}</td><td class='num {c}'>{tr:+,}</td>"
            body += "</tr>"
        orders = f"""<section><h2>매매 수량</h2>
<p class="muted">투자금 {capital:,.0f}원 · 환율 {fx:,.2f}원 · 소수점 이하 버림</p>
<div class="scroll"><table class="grid"><tr>{head}</tr>{body}</table></div></section>"""

    # 변동성 상세
    vw = res.get("vwin")
    vtxt = f"{vw[0]:%Y-%m-%d} ~ {vw[1]:%Y-%m-%d}, {vw[2]}영업일" if vw else "-"
    detail = "".join(f"<tr><td class='tk'>{e(t)}</td><td class='num'>{pct(res['vol'][t],3)}</td>"
                     f"<td class='num'>{pct(res['w'][t],1)}</td></tr>" for t in res["top"])

    # 월별 신호 표
    mtab = ""
    for m in reversed(msig or []):
        cells = " · ".join(f"{e(t)} {m['w'].get(t,0)*100:.1f}%" for t in m["top"]) or "전액 현금"
        if m["cash"] > 0.0005 and m["top"]:
            cells += f" · 현금 {m['cash']*100:.1f}%"
        mtab += f"<tr><td style='white-space:nowrap'><b>{m['hold']}</b></td><td style='text-align:left'>{cells}</td></tr>"

    # 이력
    hrows = ""
    for h in reversed(hist[-15:]):
        hrows += f"<tr><td>{h['date']}</td><td>{', '.join(h['top'])}</td></tr>"
    change_note = ""
    if len(hist) >= 2 and set(hist[-1]["top"]) != set(hist[-2]["top"]):
        added = sorted(set(hist[-1]["top"]) - set(hist[-2]["top"]))
        removed = sorted(set(hist[-2]["top"]) - set(hist[-1]["top"]))
        change_note = f"<div class='alert'>직전 실행({hist[-2]['date']}) 대비 종목 변경 — 편입: {', '.join(added) or '-'} / 편출: {', '.join(removed) or '-'}</div>"

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AAA 리밸런싱 {asof}</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mut:#6b7280;--line:#e5e7eb;--acc:#2563eb;--pos:#16a34a;--neg:#dc2626}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1115;--card:#181b22;--ink:#e6e8ec;--mut:#9aa1ad;--line:#2a2f3a;--acc:#60a5fa;--pos:#4ade80;--neg:#f87171}}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;background:var(--bg);color:var(--ink)}}
main{{max-width:760px;margin:auto;padding:20px 16px 48px}}
header h1{{font-size:20px;margin:0 0 4px}}.muted{{color:var(--mut);font-size:13px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}}
h2{{font-size:15px;margin:0 0 12px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}
.kpi b{{display:block;font-size:20px;font-variant-numeric:tabular-nums}}.kpi span{{font-size:12px;color:var(--mut)}}
.wrow{{display:flex;align-items:center;gap:10px;margin:8px 0}}.tk{{font-weight:600;min-width:48px}}
.bar{{flex:1;height:14px;background:var(--line);border-radius:7px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--acc)}}.bar i.cash{{background:var(--mut)}}
.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:6px 4px;border-bottom:1px solid var(--line)}}th{{text-align:right;color:var(--mut);font-weight:500;font-size:12px}}th:first-child{{text-align:left}}
.mbar{{width:45%}}.mbar div{{position:relative;height:10px}}.mbar b{{position:absolute;top:0;height:10px;border-radius:2px}}.mbar u{{position:absolute;left:50%;top:-3px;height:16px;border-left:1px solid var(--mut)}}
b.pos{{background:var(--pos)}}b.neg{{background:var(--neg)}}tr.out{{opacity:.45}}tr.sel .tk{{color:var(--acc)}}
.tag{{font-size:12px;color:var(--mut);text-align:right;width:52px}}.buy{{color:var(--pos);font-weight:600}}.sell{{color:var(--neg);font-weight:600}}
.scroll{{overflow-x:auto}}.alert{{margin-top:14px;padding:10px 12px;border-radius:10px;background:#fef3c7;color:#92400e;font-size:14px}}
details summary{{cursor:pointer;font-size:14px;color:var(--mut)}}
</style></head><body><main>
<header><h1>국장 AAA v6 · 리밸런싱 신호</h1>
<div class="muted">가격 기준일 {asof} · 갱신 {kst} KST · 데이터 {e(src_txt)}</div></header>
{change_note}
<div class="kpis"><div class="kpi"><span>포트 변동성(연)</span><b>{pct(res['port_vol'],1)}</b></div>
<div class="kpi"><span>조정계수 k</span><b>{res['k']:.2f}</b></div>
<div class="kpi"><span>현금비중</span><b>{pct(res['cash'],1)}</b></div></div>
<section><h2>Top{TOP_N} 최종 매수비중 (RP × k)</h2>{cards or '<p>컷오프 통과 종목 없음 → 전액 현금</p>'}</section>
{orders}
<section><h2>모멘텀 스코어 (1~6M 평균, 컷오프 {CUTOFF*100:.0f}%)</h2><table>{mom}</table></section>
<section><h2>월별 매수 종목 (전월 말 리밸런싱 기준)</h2>
<div class="scroll"><table><tr><th style="text-align:left">보유월</th><th style="text-align:left">Top5 · 최종비중</th></tr>{mtab}</table></div>
<p class="muted">예) 2026-10 = 9월 말 종가로 계산해 10월 한 달 보유한 종목</p></section>
<section><details><summary>리스크패리티 상세 (변동성 기간 {vtxt})</summary>
<table><tr><th>종목</th><th>일변동성</th><th>RP비중</th></tr>{detail}</table>
<p class="muted">Target {pct(res['target'],0)} / 포트변동성 {pct(res['port_vol'],2)} → k = min(1, Target/포트변동성)</p></details></section>
<section><details><summary>실행 이력</summary><table><tr><th>날짜</th><th style="text-align:left">Top5</th></tr>{hrows}</table></details></section>
<p class="muted">투자 판단의 책임은 본인에게 있습니다. 데이터 지연·오류 가능성이 있으니 주문 전 현재가를 확인하세요.</p>
</main></body></html>"""


# ───────────── 엑셀 갱신 ─────────────
def update_excel(path, daily, res, out_path):
    import openpyxl
    wb = openpyxl.load_workbook(path)
    m_ws, d_ws, a_ws = wb["M_Price"], wb["D_Price"], wb["AAA"]
    monthly = res["monthly"].tail(LOOKBACK_MONTHS + 1)
    for row in m_ws.iter_rows(min_row=4, max_row=m_ws.max_row):
        for c in row:
            c.value = None
    header = [m_ws.cell(2, c).value for c in range(2, 14)]
    for i, (dt, r) in enumerate(monthly.iterrows()):
        m_ws.cell(4 + i, 1, dt.to_pydatetime())
        for j, t in enumerate(header):
            m_ws.cell(4 + i, 2 + j, float(r[t]))
    a_ws["B3"] = monthly.index[-1].to_pydatetime()
    vs, _ = vol_window(daily)
    d = daily[res["top"]].dropna()
    prev = d[d.index < vs].tail(1)  # 첫 수익률 계산용 직전 종가 (엑셀 D_Price 3행)
    d = pd.concat([prev, d[(d.index >= vs) & (d.index <= res["vwin"][1])]])
    for rr in range(3, d_ws.max_row + 1):
        for cc in range(1, 7):
            d_ws.cell(rr, cc).value = None
    for i, (dt, r) in enumerate(d.iterrows()):
        d_ws.cell(3 + i, 1, dt.to_pydatetime())
        for j, t in enumerate(res["top"]):
            d_ws.cell(3 + i, 2 + j, float(r[t]))
    wb.save(out_path)


# ───────────── 실행 ─────────────
def load_config():
    path = os.path.join(HERE, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["holdings"] = {k.upper(): v for k, v in (cfg.get("holdings") or {}).items()}
    return cfg


def main():
    ap = argparse.ArgumentParser(description="국장 AAA v6 자동 리밸런싱")
    ap.add_argument("--out", default=os.path.join(HERE, "report.html"), help="결과 HTML 경로")
    ap.add_argument("--no-open", action="store_true", help="브라우저로 열지 않음")
    ap.add_argument("--public", action="store_true", help="공개 페이지용: 투자금·보유수량 표시 안 함")
    ap.add_argument("--excel", help="원본 엑셀 경로 (최신 데이터로 갱신한 사본 저장)")
    a = ap.parse_args()

    cfg = load_config()
    capital = None if a.public else cfg.get("capital_krw")
    holdings = None if a.public else cfg.get("holdings")
    fx = cfg.get("fx_krw") or None

    start = date.today() - timedelta(days=640)
    print("가격 데이터 수집 중...")
    daily = load_prices(UNIVERSE, start)
    src = daily.attrs.get("source", {})
    daily = daily.ffill().dropna()
    daily.attrs["source"] = src
    res = compute(daily)
    if capital and not fx:
        fx = load_fx(start)
        if fx is None:
            print("환율 조회 실패 → config.json의 fx_krw에 직접 입력하면 매매수량이 표시됩니다.")

    rows = order_table(res, daily, capital, fx, holdings)
    asof = daily.index[-1].strftime("%Y-%m-%d")
    hist = update_history(os.path.join(os.path.dirname(os.path.abspath(a.out)), "history.json"), asof, res)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build_html(res, daily, rows, hist, capital, fx, monthly_signals(daily)))
    print(f"\n기준일 {asof}  Top5: {', '.join(res['top'])}")
    for t in res["top"]:
        print(f"  {t:5s} {res['final'][t]:7.2%}")
    print(f"  현금  {res['cash']:7.2%}")
    print(f"\n결과 페이지: {a.out}")

    if a.excel:
        dst = a.excel.replace(".xlsx", f"_updated_{asof}.xlsx")
        update_excel(a.excel, daily, res, dst)
        print(f"엑셀 갱신본: {dst}")
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(a.out))


if __name__ == "__main__":
    main()
