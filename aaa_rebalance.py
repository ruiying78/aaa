#!/usr/bin/env python3
"""
AAA + 듀얼모멘텀 50:50 자동 리밸런싱 (Portfolio Visualizer 방식)
 - 모멘텀: 6개월 총수익률(배당 포함), 상위 5종목
 - AAA(50%): 상위 5종목을 리스크패리티(공분산 기반 ERC 해찾기, 3개월 일별 수익률) 비중
 - 듀얼모멘텀(50%): 상위 5종목 각 20%, 6개월 수익률이 마이너스면 그 몫은 SGOV
 - 두 전략 비중을 합쳐 최종 매수 비중 산출 → 웹페이지(report.html)

설정은 config.json (투자금, 보유수량). 실행할 때 입력할 것은 없습니다.
  python aaa_rebalance.py              # 계산 + report.html 생성 + 브라우저로 열기
  python aaa_rebalance.py --no-open    # 브라우저 안 열기 (GitHub 자동화용)
  python aaa_rebalance.py --public     # 공개 페이지용 (투자금·보유수량 숨김)
"""
import argparse
import html
import io
import json
import math
import os
import webbrowser
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))

# ───────────── 전략 파라미터 ─────────────
UNIVERSE = ["SPY", "VGK", "EWJ", "EEM", "IYR", "GLD", "PDBC", "IEF", "TLT", "SGOV", "DBMF", "BITO"]
CASH = "SGOV"           # 듀얼모멘텀에서 마이너스 종목 대체
LOOKBACK = 6            # 모멘텀: 6개월 총수익률
TOP_N = 5
VOL_MONTHS = 3          # 변동성: 달력상 3개월 (실행일이 말일이면 그 달 포함, 아니면 직전 달까지)
MIX = {"AAA": 0.5, "DM": 0.5}


# ───────────── 데이터 ─────────────
def load_prices(tickers, start):
    """(배당조정 종가, 실제 종가, 출처)"""
    try:
        import yfinance as yf
        df = yf.download(tickers, start=start, auto_adjust=False, progress=False)
        adj, raw = df["Adj Close"][tickers], df["Close"][tickers]
        if adj.dropna(how="all").empty:
            raise ValueError("empty")
        return adj.sort_index(), raw.sort_index(), "Yahoo"
    except Exception as e:
        print(f"Yahoo 실패({e}) → stooq 시도")
    closes = {}
    for t in tickers:
        r = requests.get("https://stooq.com/q/d/l/", params={"s": f"{t.lower()}.us", "i": "d",
                         "d1": start.strftime("%Y%m%d"), "d2": date.today().strftime("%Y%m%d")},
                         timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        closes[t] = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")["Close"]
    px = pd.DataFrame(closes)[tickers].sort_index()
    return px, px, "stooq"


def load_fx(start):
    try:
        import yfinance as yf
        s = yf.download("KRW=X", start=start, progress=False)["Close"]
        return float(s.squeeze().dropna().iloc[-1])
    except Exception:
        return None


# ───────────── 전략 로직 ─────────────
def month_ends(px):
    return px.groupby(px.index.to_period("M")).tail(1)


def momentum(adj):
    """최신 종가 / 6개월 전 월말 - 1 (총수익률)"""
    m = month_ends(adj)
    return m.iloc[-1] / m.iloc[-1 - LOOKBACK] - 1


def select_top(score):
    s = score.dropna()
    return sorted(s.index, key=lambda t: (-s[t], UNIVERSE.index(t)))[:TOP_N]


def vol_window(today):
    today = pd.Timestamp(today)
    end_m = today.to_period("M")
    if today.day != today.days_in_month:
        end_m -= 1
    return (end_m - (VOL_MONTHS - 1)).start_time, end_m.end_time


def rp_weights(adj, top, today):
    vs, ve = vol_window(today)
    rets = adj[top].pct_change().dropna()
    rets = rets[(rets.index >= vs) & (rets.index <= ve)]
    vol = rets.std(ddof=1)
    w = pd.Series(erc_weights(rets.cov().values), index=top)
    return w, vol, (rets.index[0], rets.index[-1], len(rets))


def erc_weights(cov, iters=1000, tol=1e-12):
    """공분산 기반 동일위험기여(ERC) 해찾기: 각 종목의 w_i*(Σw)_i 가 같아지도록 (순환 좌표하강법)"""
    import math
    import numpy as np
    n = cov.shape[0]
    x = 1 / np.sqrt(np.diag(cov))
    x = x / x.sum()
    b = 1.0 / n
    for _ in range(iters):
        x_old = x.copy()
        for i in range(n):
            s = cov[i] @ x - cov[i, i] * x[i]
            x[i] = (-s + math.sqrt(s * s + 4 * cov[i, i] * b)) / (2 * cov[i, i])
        if np.max(np.abs(x - x_old)) < tol:
            break
    return x / x.sum()


def compute(adj, today=None):
    if today is None:
        today = datetime.now(timezone(timedelta(hours=8))).date()
    score = momentum(adj)
    top = select_top(score)
    aaa, vol, vwin = rp_weights(adj, top, today)
    dm = {}
    for t in top:
        k = t if score[t] >= 0 else CASH
        dm[k] = dm.get(k, 0) + 1 / TOP_N
    final = {}
    for t, w in aaa.items():
        final[t] = final.get(t, 0) + MIX["AAA"] * float(w)
    for t, w in dm.items():
        final[t] = final.get(t, 0) + MIX["DM"] * w
    final = dict(sorted(final.items(), key=lambda kv: -kv[1]))
    return dict(score=score, top=top, aaa={t: float(aaa[t]) for t in top}, vol=vol,
                dm=dm, final=final, vwin=vwin)


def monthly_signals(adj, n=12):
    """지난 n개월: 각 월말 종가로 계산 → 다음 달 보유"""
    out = []
    allme = list(month_ends(adj).index)
    me = [d for d in allme if d.to_period("M") < adj.index[-1].to_period("M")]
    for d in me[-n:]:
        sub = adj[adj.index <= d]
        if len(month_ends(sub)) < LOOKBACK + 1:
            continue
        r = compute(sub, today=d.to_period("M").end_time.normalize())
        d1 = allme[allme.index(d) + 1]
        ret = sum(w * (adj.at[d1, t] / adj.at[d, t] - 1) for t, w in r["final"].items())
        out.append({"hold": str(d.to_period("M") + 1), "final": r["final"], "top": r["top"],
                    "ret": float(ret), "partial": d1 == adj.index[-1] and d1.to_period("M") == adj.index[-1].to_period("M")
                    and d1 != (d1 + pd.offsets.BMonthEnd(0))})
    return out


def month_to_date(adj, msig):
    """이번 달 보유분(전월 말 신호)의 월초 이후 수익률"""
    if not msig:
        return None
    cur = msig[-1]
    if cur["hold"] != str(adj.index[-1].to_period("M")):
        return None
    d0 = month_ends(adj).index[-2]
    r = {t: float(adj[t].iloc[-1] / adj.at[d0, t] - 1) for t in cur["final"]}
    total = sum(w * r[t] for t, w in cur["final"].items())
    return {"hold": cur["hold"], "from": d0, "to": adj.index[-1], "w": cur["final"], "r": r, "total": total}


def track_stats(msig):
    """월별 수익률 요약 (진행 중인 달은 낙폭에만 포함)"""
    done = [m["ret"] for m in msig if not m["partial"]]
    if not done:
        return None
    eq, peak, dd, cur = 1.0, 1.0, 0.0, 0.0
    for m in msig:
        eq *= 1 + m["ret"]
        peak = max(peak, eq)
        cur = eq / peak - 1
        dd = min(dd, cur)
    cum = 1.0
    for x in done:
        cum *= 1 + x
    return {"n": len(done), "cum": cum - 1, "ann": cum ** (12 / len(done)) - 1,
            "best": max(done), "worst": min(done), "mdd": dd, "cur_dd": cur,
            "pos": sum(x >= 0 for x in done), "neg": sum(x < 0 for x in done)}


def write_xlsx(path, msig):
    """월별 수익률 엑셀 (누적·낙폭·요약은 수식)"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter as L
    F = "Arial"
    wb = Workbook()
    ws = wb.active
    ws.title = "월별 수익률"
    ws["A1"] = "AAA + 듀얼모멘텀 50:50 · 월별 수익률"
    ws["A1"].font = Font(name=F, size=14, bold=True)
    ws["A2"] = "6개월 총수익률 Top5 · AAA(리스크패리티·ERC) 50% + 듀얼모멘텀(각 20%, 음수→SGOV) 50% · 배당 포함 · 파란색=입력값, 검은색=수식"
    ws["A2"].font = Font(name=F, size=9, color="6B7280")
    hd = ["보유월", "월 수익률", "상태", "누적 수익률", "고점 대비"] + UNIVERSE
    for j, h in enumerate(hd, 1):
        c = ws.cell(4, j, h)
        c.font = Font(name=F, size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1F2937")
        c.alignment = Alignment(horizontal="center")
    first = 5
    blue, base = Font(name=F, size=10, color="0000FF"), Font(name=F, size=10)
    yel = PatternFill("solid", fgColor="FEF3C7")
    for i, m in enumerate(msig):
        r = first + i
        ws.cell(r, 1, m["hold"]).font = base
        c = ws.cell(r, 2, round(m["ret"], 6)); c.font = blue; c.number_format = "0.00%"
        ws.cell(r, 3, "진행 중" if m["partial"] else "완료").font = base
        prev = f"D{r-1}" if i else "0"
        c = ws.cell(r, 4, f"=(1+{prev})*(1+B{r})-1"); c.font = base; c.number_format = "0.00%"
        c = ws.cell(r, 5, f"=(1+D{r})/MAX(1,1+MAX($D${first}:D{r}))-1"); c.font = base; c.number_format = "0.00%"
        for j, t in enumerate(UNIVERSE, 6):
            v = m["final"].get(t, 0)
            c = ws.cell(r, j, round(v, 4) if v else 0); c.font = blue; c.number_format = "0.0%;-0.0%;-"
        if m["partial"]:
            for j in range(1, len(hd) + 1):
                ws.cell(r, j).fill = yel
    last = first + len(msig) - 1
    s0 = last + 2
    rng_b, rng_c = f"B{first}:B{last}", f"C{first}:C{last}"
    items = [("완료 개월 수", f'=COUNTIF({rng_c},"완료")', "0"),
             ("누적 수익률 (완료)", f'=SUMPRODUCT((({rng_c})="완료")*LN(1+{rng_b}))', None),
             ("월평균 수익률", f'=AVERAGEIF({rng_c},"완료",{rng_b})', "0.00%"),
             ("플러스 개월", f'=COUNTIFS({rng_c},"완료",{rng_b},">=0")', "0"),
             ("마이너스 개월", f'=COUNTIFS({rng_c},"완료",{rng_b},"<0")', "0"),
             ("최고의 달", f'=_xlfn.MAXIFS({rng_b},{rng_c},"완료")', "0.00%"),
             ("최악의 달", f'=_xlfn.MINIFS({rng_b},{rng_c},"완료")', "0.00%"),
             ("최대 낙폭", f"=MIN(E{first}:E{last})", "0.00%"),
             ("현재 고점 대비", f"=E{last}", "0.00%")]
    ws.cell(s0, 1, "요약").font = Font(name=F, size=10, bold=True)
    for k, (a, f, nf) in enumerate(items, 1):
        ws.cell(s0 + k, 1, a).font = base
        c = ws.cell(s0 + k, 2, f); c.font = base
        if nf:
            c.number_format = nf
    # 누적은 로그합 → 지수 변환
    lr = s0 + 2
    ws.cell(lr, 2).value = f'=EXP(SUMPRODUCT((({rng_c})="완료")*LN(1+{rng_b})))-1'
    ws.cell(lr, 2).number_format = "0.00%"
    ws.cell(s0 + len(items) + 1, 1, "연환산 수익률").font = base
    c = ws.cell(s0 + len(items) + 1, 2, f"=(1+B{lr})^(12/B{s0+1})-1"); c.font = base; c.number_format = "0.00%"
    ws.column_dimensions["A"].width = 18
    for j in range(2, len(hd) + 1):
        ws.column_dimensions[L(j)].width = 10 if j <= 5 else 7
    ws.freeze_panes = "B5"
    wb.save(path)


def order_table(res, raw, capital, fx, holdings):
    last = raw.ffill().iloc[-1]
    rows = []
    for t, w in res["final"].items():
        row = {"ticker": t, "final": w, "price": float(last[t])}
        if capital and fx:
            amt = capital * w / fx
            row["amount_usd"] = amt
            row["target_qty"] = math.floor(amt / last[t])
            if holdings:
                row["held"] = int(holdings.get(t, 0))
                row["trade"] = row["target_qty"] - row["held"]
        rows.append(row)
    if capital and fx and holdings:
        for t, q in holdings.items():
            if t not in res["final"] and int(q):
                rows.append({"ticker": t, "final": 0.0, "price": float(last.get(t, float("nan"))),
                             "target_qty": 0, "held": int(q), "trade": -int(q)})
    return rows


def update_history(path, asof, res):
    hist = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            hist = json.load(f)
    hist = [h for h in hist if h["date"] != asof]
    hist.append({"date": asof, "top": res["top"], "weights": {t: round(w, 4) for t, w in res["final"].items()}})
    hist = sorted(hist, key=lambda h: h["date"])[-60:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=1)
    return hist


# ───────────── HTML ─────────────
def pct(v, d=1):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v*100:.{d}f}%"


def bars(weights, cls=""):
    e = html.escape
    return "".join(f"""<div class="wrow"><span class="tk">{e(t)}</span>
<div class="bar"><i class="{cls}" style="width:{w*100:.1f}%"></i></div><span class="num">{pct(w)}</span></div>"""
                   for t, w in weights.items())


def build_html(res, adj, rows, hist, msig, capital, fx, src, mtd=None):
    e = html.escape
    asof = adj.index[-1].strftime("%Y-%m-%d")
    kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")

    smax = max(float(res["score"].abs().max()), 0.01)
    mom = ""
    for t in sorted(UNIVERSE, key=lambda t: -res["score"][t]):
        s = float(res["score"][t])
        sel = t in res["top"]
        tag = ("선정" if s >= 0 else f"선정→{CASH}") if sel else ""
        width = abs(s) / smax * 50
        side = f"left:50%;width:{width:.1f}%" if s >= 0 else f"left:{50-width:.1f}%;width:{width:.1f}%"
        mom += f"""<tr class="{'sel' if sel else ''}"><td class="tk">{e(t)}</td><td class="num">{s*100:+.2f}%</td>
<td class="mbar"><div><b style="{side}" class="{'neg' if s<0 else 'pos'}"></b><u></u></div></td><td class="tag">{tag}</td></tr>"""

    orders = ""
    if capital and fx:
        has_h = any("held" in r for r in rows)
        head = "<th>종목</th><th>비중</th><th>현재가($)</th><th>목표금액($)</th><th>목표수량</th>"
        if has_h:
            head += "<th>보유</th><th>매매</th>"
        body = ""
        for r in rows:
            body += (f"<tr><td class='tk'>{e(r['ticker'])}</td><td class='num'>{pct(r.get('final'))}</td>"
                     f"<td class='num'>{r['price']:,.2f}</td><td class='num'>{r.get('amount_usd',0):,.0f}</td>"
                     f"<td class='num'>{r.get('target_qty',0):,}</td>")
            if has_h:
                tr = r.get("trade", 0)
                c = "buy" if tr > 0 else ("sell" if tr < 0 else "")
                body += f"<td class='num'>{r.get('held',0):,}</td><td class='num {c}'>{tr:+,}</td>"
            body += "</tr>"
        orders = f"""<section><h2>매매 수량</h2><p class="muted">투자금 {capital:,.0f}원 · 환율 {fx:,.2f}원 · 소수점 이하 버림</p>
<div class="scroll"><table><tr>{head}</tr>{body}</table></div></section>"""

    vw = res["vwin"]
    vtxt = f"{vw[0]:%Y-%m-%d} ~ {vw[1]:%Y-%m-%d}, {vw[2]}영업일"
    detail = "".join(f"<tr><td class='tk'>{e(t)}</td><td class='num'>{pct(float(res['vol'][t]),3)}</td>"
                     f"<td class='num'>{pct(res['aaa'][t])}</td></tr>" for t in res["top"])
    mtab = "".join(f"<tr><td style='white-space:nowrap'><b>{m['hold']}</b></td>"
                   f"<td class='num {'buy' if m['ret'] >= 0 else 'sell'}'>{m['ret']*100:+.2f}%{'*' if m['partial'] else ''}</td>"
                   "<td style='text-align:left;font-size:13px'>"
                   + " · ".join(f"{e(t)} {w*100:.1f}%" for t, w in m["final"].items()) + "</td></tr>"
                   for m in reversed(msig))
    st = track_stats(msig)
    msum = ""
    if st:
        kp = [("누적 (완료 " + str(st["n"]) + "개월)", st["cum"]), ("연환산", st["ann"]), ("최고의 달", st["best"]),
              ("최악의 달", st["worst"]), ("최대 낙폭", st["mdd"]), ("현재 고점 대비", st["cur_dd"])]
        msum = ("<div class='kpis'>" + "".join(
            f"<div class='kpi'><span>{a}</span><b class='{'buy' if v >= 0 else 'sell'}'>{v*100:+.2f}%</b></div>" for a, v in kp)
            + f"</div><p class='muted'>플러스 {st['pos']}개월 / 마이너스 {st['neg']}개월 · * 진행 중인 달 · "
            "<a href='monthly.xlsx' download>엑셀 다운로드</a></p>")
    used = [t for t in UNIVERSE if any(t in m["final"] for m in msig)]
    mat = ("<tr><th style='text-align:left'>보유월</th>" + "".join(f"<th>{e(t)}</th>" for t in used) + "</tr>"
           + "".join("<tr><td style='white-space:nowrap'><b>" + m["hold"][2:] + "</b></td>" + "".join(
               f"<td class='num'>{m['final'][t]*100:.0f}</td>" if t in m["final"] else "<td class='num muted'>·</td>"
               for t in used) + "</tr>" for m in reversed(msig))
           + "<tr><td><b>편입</b></td>" + "".join(
               f"<td class='num'><b>{sum(t in m['final'] for m in msig)}</b></td>" for t in used) + "</tr>")
    change = ""
    if len(hist) >= 2 and set(hist[-1]["top"]) != set(hist[-2]["top"]):
        a = sorted(set(hist[-1]["top"]) - set(hist[-2]["top"]))
        r_ = sorted(set(hist[-2]["top"]) - set(hist[-1]["top"]))
        change = f"<div class='alert'>직전 실행({hist[-2]['date']}) 대비 Top5 변경 — 편입: {', '.join(a) or '-'} / 편출: {', '.join(r_) or '-'}</div>"

    mtd_html = ""
    if mtd:
        rws = "".join(f"<tr><td class='tk'>{e(t)}</td><td class='num'>{pct(w)}</td>"
                      f"<td class='num {'buy' if mtd['r'][t] >= 0 else 'sell'}'>{mtd['r'][t]*100:+.2f}%</td>"
                      f"<td class='num'>{w*mtd['r'][t]*100:+.2f}%p</td></tr>" for t, w in mtd["w"].items())
        tc = "buy" if mtd["total"] >= 0 else "sell"
        mtd_html = (f"<section><h2>{mtd['hold']} 보유분 수익률 <span class='{tc}'>{mtd['total']*100:+.2f}%</span></h2>"
                    f"<p class='muted'>{mtd['from']:%Y-%m-%d} 종가 → {mtd['to']:%Y-%m-%d} 종가 · 배당 포함</p>"
                    f"<table><tr><th>종목</th><th>비중</th><th>수익률</th><th>기여도</th></tr>{rws}</table></section>")

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AAA+DM 리밸런싱 {asof}</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mut:#6b7280;--line:#e5e7eb;--acc:#2563eb;--acc2:#0d9488;--pos:#16a34a;--neg:#dc2626}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1115;--card:#181b22;--ink:#e6e8ec;--mut:#9aa1ad;--line:#2a2f3a;--acc:#60a5fa;--acc2:#2dd4bf;--pos:#4ade80;--neg:#f87171}}}}
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;background:var(--bg);color:var(--ink)}}
main{{max-width:760px;margin:auto;padding:20px 16px 48px}}header h1{{font-size:20px;margin:0 0 4px}}.muted{{color:var(--mut);font-size:13px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}}h2{{font-size:15px;margin:0 0 12px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media(max-width:560px){{.grid2{{grid-template-columns:1fr}}}}
.grid2 section{{margin-top:0}}.wrap{{margin-top:14px}}
.wrow{{display:flex;align-items:center;gap:10px;margin:8px 0}}.tk{{font-weight:600;min-width:48px}}
.bar{{flex:1;height:14px;background:var(--line);border-radius:7px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--acc)}}.bar i.t2{{background:var(--acc2)}}
.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:6px 4px;border-bottom:1px solid var(--line)}}th{{text-align:right;color:var(--mut);font-weight:500;font-size:12px}}th:first-child{{text-align:left}}
.mbar{{width:45%}}.mbar div{{position:relative;height:10px}}.mbar b{{position:absolute;top:0;height:10px;border-radius:2px}}.mbar u{{position:absolute;left:50%;top:-3px;height:16px;border-left:1px solid var(--mut)}}
b.pos{{background:var(--pos)}}b.neg{{background:var(--neg)}}tr.sel .tk{{color:var(--acc)}}
.tag{{font-size:12px;color:var(--mut);text-align:right;width:80px}}.buy{{color:var(--pos);font-weight:600}}.sell{{color:var(--neg);font-weight:600}}
.scroll{{overflow-x:auto}}.alert{{margin-top:14px;padding:10px 12px;border-radius:10px;background:#fef3c7;color:#92400e;font-size:14px}}
details summary{{cursor:pointer;font-size:14px;color:var(--mut)}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px}}.kpi{{border:1px solid var(--line);border-radius:10px;padding:10px}}
.kpi span{{display:block;font-size:12px;color:var(--mut)}}.kpi b{{font-size:18px;font-variant-numeric:tabular-nums}}
table.mat td,table.mat th{{padding:5px 3px;font-size:13px}}a{{color:var(--acc)}}
</style></head><body><main>
<header><h1>AAA + 듀얼모멘텀 50:50 · 리밸런싱 신호</h1>
<div class="muted">가격 기준일 {asof} · 갱신 {kst} KST · 데이터 {e(src)} · 6개월 총수익률 · Top{TOP_N}</div></header>
{change}
{mtd_html}
<section><h2>다음 리밸런싱 예상 비중</h2>{bars(res['final'])}</section>
{orders}
<div class="grid2 wrap"><section><h2>AAA 50% (리스크패리티·ERC)</h2>{bars(res['aaa'])}</section>
<section><h2>듀얼모멘텀 50% (각 20%)</h2>{bars(res['dm'], 't2')}</section></div>
<section><h2>모멘텀 (6개월 총수익률)</h2><table>{mom}</table>
<p class="muted">상위 {TOP_N}개 선정. 듀얼모멘텀에서는 마이너스 종목의 20%를 {CASH}로 대체</p></section>
<section><h2>월별 수익률 · 매수 종목 (전월 말 기준)</h2>{msum}<div class="scroll"><table><tr><th style="text-align:left">보유월</th><th>수익률</th><th style="text-align:left">최종 비중</th></tr>{mtab}</table></div>
<p class="muted">예) 2026-10 = 9월 말 종가로 계산해 10월 한 달 보유</p></section>
<section><h2>종목별 비중 (%)</h2><div class="scroll"><table class="mat">{mat}</table></div>
<p class="muted">마지막 행은 지난 12개월 중 편입된 횟수</p></section>
<section><details><summary>AAA 리스크패리티(ERC, 공분산 해찾기) 상세 (기간 {vtxt})</summary>
<table><tr><th>종목</th><th>일변동성</th><th>AAA 비중</th></tr>{detail}</table></details></section>
<p class="muted">맨 위 비중은 가격 기준일 종가로 미리 계산한 값이며, 월말 종가로 확정됩니다. 투자 판단의 책임은 본인에게 있습니다.</p>
</main></body></html>"""


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
    ap = argparse.ArgumentParser(description="AAA + 듀얼모멘텀 50:50 자동 리밸런싱")
    ap.add_argument("--out", default=os.path.join(HERE, "report.html"))
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--public", action="store_true", help="투자금·보유수량 표시 안 함")
    a = ap.parse_args()

    cfg = load_config()
    capital = None if a.public else cfg.get("capital_krw")
    holdings = None if a.public else cfg.get("holdings")
    fx = cfg.get("fx_krw") or None

    start = date.today() - timedelta(days=640)
    print("가격 데이터 수집 중...")
    adj, raw, src = load_prices(UNIVERSE, start)
    adj = adj.ffill().dropna()
    res = compute(adj)
    if capital and not fx:
        fx = load_fx(start)
    rows = order_table(res, raw, capital, fx, holdings)
    asof = adj.index[-1].strftime("%Y-%m-%d")
    hist = update_history(os.path.join(os.path.dirname(os.path.abspath(a.out)), "history.json"), asof, res)
    msig = monthly_signals(adj)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build_html(res, adj, rows, hist, msig, capital, fx, src, month_to_date(adj, msig)))
    try:
        write_xlsx(os.path.join(os.path.dirname(os.path.abspath(a.out)), "monthly.xlsx"), msig)
    except Exception as ex:
        print(f"엑셀 생성 실패: {ex}")
    print(f"\n기준일 {asof}  Top5: {', '.join(res['top'])}")
    for t, w in res["final"].items():
        print(f"  {t:5s} {w:7.2%}")
    print(f"\n결과 페이지: {a.out}")
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(a.out))


if __name__ == "__main__":
    main()
