"""여름 손실 · 간격 what-if — **서버 없이 열리는** 판.

두 기능이 API 로만 있었다. 나머지 뷰 22개는 파일만 열면 도는데 가장 최근
것 둘만 서버를 요구해서, 심사장에서 서버를 못 띄우면 그 둘이 안 보였다.
"서버 없이도 전부 열린다" 가 이 프로젝트의 성질이므로 여기서 되돌린다.

**계산은 여기서 하지 않는다.** `server/routers/season.compute()` 와
`capacity.interval_whatif()` 를 그대로 부른다 — 라우터가 쓰는 그 함수다.
구울 때 한 번 부르고 값을 박아 넣을 뿐이라, 서버로 본 화면과 파일로 본
화면이 같은 수를 말한다. 산식을 이 파일에 옮겨 적으면 언젠가 갈린다.

박아 넣는 값이므로 **입력을 바꿀 수 없다.** 우리 농장 값을 넣으려면 서버를
띄워야 하고, 화면이 그 사실을 먼저 말한다.

    python competition/src/build_season_interval.py
출력: competition/dashboard/season_interval.html
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../competition
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))         # competition.server 용

import psy_priority as pp  # noqa: E402

OUT = os.path.join(ROOT, "dashboard", "season_interval.html")

# 예시 농장 — 300두 기본 구성. **실제 농장이 아니다.**
# 시연 규모의 정본은 `psy_priority.DEMO_SOWS` 다 — 화면마다 다른 두수를
# 보이면 같은 농장을 보는 것처럼 읽히지 않는다.
DEMO_SOWS = pp.DEMO_SOWS
DEMO_PERF = {"weaned": 11.0, "farrowing_rate": 85.0, "survival": 94.0}

C_ACC, C_BAD, C_GOOD, C_WARN = "#2a78d6", "#d03b3b", "#1baf7a", "#e8a33d"


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def bold(s: str) -> str:
    """서버 문구의 **강조** 표기 — 그대로 넣으면 별표가 보인다."""
    out, parts = [], esc(s).split("**")
    for i, p in enumerate(parts):
        out.append(f"<b>{p}</b>" if i % 2 else p)
    return "".join(out)


def man(won: float) -> str:
    return f"{round(won / 1e4):,}만원"


def tag(kind: str, text: str) -> str:
    color = {"실측": C_GOOD, "계산": C_ACC, "가정": C_WARN, "유도": C_WARN}[kind]
    return (f'<span class="tag" style="--tc:{color}">{kind}</span>'
            f'<span class="tagtxt">{esc(text)}</span>')


# -- 자료 ------------------------------------------------------------------
def gather() -> dict:
    """라우터가 쓰는 함수를 그대로 부른다 — 여기서 재구현하지 않는다."""
    from competition.server.routers import capacity, season
    from competition.server.schemas import FarmSetup

    import batch_flow as bf

    stub = FarmSetup(interval_days=21, lactation_days=24,
                     pre_farrow_days=7, washout_days=7)
    barns = bf.design_barns(
        DEMO_SOWS, 21.0, lactation=24, pre_farrow=7, washdown=7,
        weaned_per_litter=DEMO_PERF["weaned"],
        extra_rooms=capacity._extra_rooms(stub))
    setup = FarmSetup(name="예시 농장", n_sows=DEMO_SOWS, interval_days=21,
                      lactation_days=24, pre_farrow_days=7, washout_days=7,
                      barns=barns, performance=DEMO_PERF)
    return {"season": season.compute(DEMO_SOWS),
            "interval": capacity.interval_whatif(setup),
            "barns": [b.model_dump() for b in setup.barns]}


# -- 분포 띠 ---------------------------------------------------------------
def strip(q: dict) -> str:
    """**중앙값 하나로 말하면 안 된다** — 농장마다 갈리는 게 요지다."""
    lo, hi = q["p10"], q["p90"]
    span = max(1e-9, hi - lo)
    x0, x1 = lo - span * .18, hi + span * .18

    def px(x):
        return (x - x0) / (x1 - x0) * 100

    return f"""<div class="strip">
  <div class="band" style="left:{px(lo):.1f}%;width:{px(hi) - px(lo):.1f}%"></div>
  <div class="band2" style="left:{px(q['p25']):.1f}%;
       width:{px(q['p75']) - px(q['p25']):.1f}%"></div>
  <div class="med" style="left:{px(q['median']):.1f}%"></div>
  <div class="lb" style="left:{px(lo):.1f}%">하위10% {lo}</div>
  <div class="lb" style="left:{px(q['median']):.1f}%">중앙 {q['median']}</div>
  <div class="lb" style="left:{px(hi):.1f}%">상위10% {hi}</div></div>"""


def kpis(items) -> str:
    return ('<div class="kpis">' + "".join(
        f'<div class="kpi"><span class="v">{v}</span>'
        f'<span class="k">{esc(k)}</span>'
        f'<span class="d">{bold(d)}</span></div>' for v, k, d in items)
        + "</div>")


# -- 패널 1 · 여름 손실 ------------------------------------------------------
def panel_season(d: dict) -> str:
    s = d["season"]
    sc, acc = s["scenario"], s["accidents"]
    lo, hi = s["implantation_window"]

    body = kpis([
        (man(sc["median"]["won_year"]), "중앙 농장이라면",
         f"여름 손실 +{sc['median']['loss_pp']}%p · 가정"),
        (man(sc["p90"]["won_year"]), "취약 상위10% 라면",
         f"여름 손실 +{sc['p90']['loss_pp']}%p · 가정"),
        (f"{s['n_sows']:,}두", "환산 규모",
         f"PSY {s['psy_used']} · {s['psy_source']}"),
    ]) + strip(s["loss"])

    body += (f'<p class="warn"><b>위는 특정 농장의 값이 아닙니다</b> — '
             f'국내 {s["n_farms"]}농장 분포를 {s["n_sows"]:,}두 규모로 환산한 '
             f'범위입니다. 우리 농장이 어느 쪽인지 알려면 월별 분만율 12개월이 '
             f'필요하고, 그건 <b>서버를 띄워야</b> 넣을 수 있습니다.</p>'
             f'<p class="note">패널 실측 기준(농장마다 <b>자기</b> PSY·'
             f'<b>자기</b> 겨울로 낸 금액)의 중앙은 '
             f'<b>{man(s["panel_won_ref"]["median"])}</b>으로 위와 다릅니다 — '
             f'곱의 중앙값 ≠ 중앙값의 곱.</p>')

    body += kpis([
        (f"{s['spread']['true_share'] * 100:.0f}%", "진짜 농장 차이",
         "관측 분산 중. 나머지는 표본 오차"),
        (f"{s['loss_shrunk']['p10']} ~ {s['loss_shrunk']['p90']}",
         "축소 후 분포 %p",
         f"관측 {s['loss']['p10']} ~ {s['loss']['p90']}"),
        (f"ρ {s['join']['PSY']['rho']}", "PSY 와의 상관",
         "연간 성적으로는 못 맞힙니다"),
    ])

    # 무너지는 경로 — 사양이 아니라 착상
    key = "임신사고(1차)"
    top = sorted(acc["delta"].items(), key=lambda x: -x[1])[:4]
    rows = "".join(
        f'<tr{" class=hi" if k == key else ""}>'
        f'<td>{esc(k.replace("임신사고", "").strip("()"))}</td>'
        f'<td class="d">{acc["winter"][k] * 100:.1f}%</td>'
        f'<td class="d">{acc["summer"][k] * 100:.1f}%</td>'
        f'<td class="d"><span class="pill {"stop" if v > .02 else "mute"}">'
        f'{"+" if v > 0 else ""}{v * 100:.1f}%p</span></td></tr>'
        for k, v in top)
    body += (f'<h3>무너지는 건 사양이 아니라 착상입니다</h3>'
             f'<div class="tw"><table><thead><tr><th>임신사고 구성</th>'
             f'<th>겨울</th><th>여름</th><th>차이</th></tr></thead>'
             f'<tbody>{rows}</tbody></table></div>'
             f'<p class="note">여름에 이유두수·재귀율은 거의 그대로인데 '
             f'임신사고 구성이 1차 재발 쪽으로 '
             f'<b>{acc["delta"][key] * 100:.1f}%p</b> 기웁니다. 겨냥할 시점은 '
             f'<b>교배 후 {lo}~{hi}일 착상기</b>이고, 이 구간 축사의 THI 를 '
             f'낮추는 것이 처방입니다.</p>')

    body += ('<div class="caveat">'
             + "<br>".join("· " + bold(c) for c in s["caveats"]) + "</div>")
    return body


# -- 패널 2 · 간격 what-if ---------------------------------------------------
def panel_interval(d: dict) -> str:
    iw = d["interval"]
    top = max((r["n_sows"] for r in iw["rows"]), default=1) or 1
    rows = []
    for r in iw["rows"]:
        cur, blocked = r["current"], r["n_sows"] == 0
        marks = ('<span class="pill">지금</span>' if cur else "") + (
            '<span class="pill good">최대 규모</span>'
            if not cur and r["interval_days"] == iw["best"] else "")
        if blocked:
            sows = '<span class="pill stop">막힘</span>'
            why = esc("·".join(r["blocked"])) + " 방 부족"
            ceil = svc = "—"
        else:
            sows = (f'<span class="qbar"><i style="width:'
                    f'{r["n_sows"] / top * 100:.0f}%"></i></span>'
                    f'<span class="d">{r["n_sows"]:,}두</span>')
            why = esc(r["binding"])
            ceil = f'{r["ceiling_year"]:,}두'
            svc = f'{r["services_per_batch"]}두'
        rows.append(f'<tr{" class=cur" if cur else ""}>'
                    f'<td><b>{esc(r["name"])}</b> '
                    f'<span class="d">{r["interval_days"]:g}일</span>{marks}</td>'
                    f'<td>{sows}</td><td class="d">{why}</td>'
                    f'<td class="d">{ceil}</td><td class="d">{svc}</td>'
                    f'<td class="d">×{r["peak_ratio"]:g}</td></tr>')

    return (f'<div class="tw"><table><thead><tr><th>간격</th>'
            f'<th>받을 수 있는 모돈</th><th>병목</th><th>연간 출하 상한</th>'
            f'<th>배치당 교배</th><th>한 날 집중도</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="note">{esc(iw["note"])}</p>'
            f'<p class="note"><b>막힌 간격은 두수를 줄여서 풀리지 '
            f'않습니다</b> — 회전이 안 되는 것이라 방을 늘리거나 간격을 '
            f'넓혀야 합니다.</p>')


# -- 조립 ------------------------------------------------------------------
def build(d: dict) -> str:
    s, iw = d["season"], d["interval"]
    barns = " · ".join(f'{b["stage"]} {b["rooms"]}방×{b["per"]}'
                       for b in d["barns"])
    sec = [
        ("1", "여름 손실은 농장마다 갈립니다", "실측",
         f"국내 {s['n_farms']}농장 월별 실측 · 교배월 기준 · 중복 제거 후",
         f"전체로는 여름 교배분 분만율이 겨울보다 중앙 "
         f"{s['overall']['summer_minus_winter']}%p 떨어집니다. 그런데 하위10% "
         f"{s['loss']['p10']} ~ 상위10% +{s['loss']['p90']}%p 로 갈립니다 — "
         f"<b>공통 처방이 아니라 선별 처방</b>입니다.", panel_season(d)),
        ("2", "같은 돈사로 간격만 바꾸면", "계산",
         f"예시 구성 {DEMO_SOWS}두 기본 · 지금 {iw['current_interval']:g}일",
         "짓기 전 질문이 아니라 <b>지은 뒤</b> 질문입니다. 좁히면 같은 "
         "분만틀이 더 자주 돌아 규모가 늘지만 <b>방 수 요구가 커져 "
         "막힙니다</b>. 넓히면 방은 넉넉해지는데 <b>배치가 커져 한 날에 "
         "몰립니다</b>.", panel_interval(d)),
    ]
    cards = "".join(
        f'<section class="card"><div class="ch"><span class="cn">{n}</span>'
        f'<h2>{t}</h2></div><div class="prov">{tag(kind, src)}</div>'
        f'<p class="lead">{lead}</p><div class="body">{body}</div></section>'
        for n, t, kind, src, lead, body in sec)

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>여름 손실 · 간격 what-if</title><style>
:root{{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--surface2:#f2f2ee;
--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--border:rgba(11,11,11,.12);
--band:rgba(11,11,11,.09);--band2:rgba(42,120,214,.22);--acc:{C_ACC};
--accs:rgba(42,120,214,.10);--stop:{C_BAD};--stops:rgba(208,59,59,.12);
--good:{C_GOOD};--goods:rgba(27,175,122,.14);--warn:{C_WARN}}}
@media(prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{
--page:#0d0d0d;--surface:#1a1a19;--surface2:#242422;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--border:rgba(255,255,255,.14);--band:rgba(255,255,255,.11);
--band2:rgba(57,135,229,.30);--acc:#5fa8f0;--accs:rgba(95,168,240,.14);
--stop:#f07070;--stops:rgba(240,112,112,.15);--good:#4fd6a0;
--goods:rgba(79,214,160,.16);--warn:#e8b45e}}}}
:root[data-theme=dark]{{--page:#0d0d0d;--surface:#1a1a19;--surface2:#242422;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--border:rgba(255,255,255,.14);
--band:rgba(255,255,255,.11);--band2:rgba(57,135,229,.30);--acc:#5fa8f0;
--accs:rgba(95,168,240,.14);--stop:#f07070;--stops:rgba(240,112,112,.15);
--good:#4fd6a0;--goods:rgba(79,214,160,.16);--warn:#e8b45e}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif;
background:var(--page);color:var(--ink);line-height:1.55;padding:22px}}
.wrap{{max-width:860px;margin:0 auto}}
h1{{font-size:1.55rem;letter-spacing:-.02em}}
.sub{{color:var(--ink2);font-size:.92rem;margin:6px 0}}
.demo{{font-size:.82rem;color:var(--ink2);background:var(--surface2);
border:1px solid var(--border);border-radius:9px;padding:9px 12px;margin:12px 0 18px}}
.card{{background:var(--surface);border:1px solid var(--border);
border-radius:13px;padding:17px 18px;margin-bottom:15px}}
.ch{{display:flex;align-items:baseline;gap:9px}}
.cn{{font-size:.72rem;font-weight:700;color:var(--page);background:var(--ink);
border-radius:6px;padding:2px 7px}}
h2{{font-size:1.05rem;letter-spacing:-.01em}}
h3{{font-size:.92rem;margin:18px 0 8px}}
.prov{{display:flex;align-items:center;gap:7px;margin:7px 0 2px;flex-wrap:wrap}}
.tag{{font-size:.68rem;font-weight:700;color:#fff;background:var(--tc);
border-radius:5px;padding:1px 7px}}
.tagtxt{{font-size:.74rem;color:var(--muted)}}
.lead{{font-size:.87rem;color:var(--ink2);margin:9px 0 4px}}
.body{{margin-top:12px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:1px;background:var(--border);border:1px solid var(--border);
border-radius:10px;overflow:hidden;margin:12px 0}}
.kpi{{background:var(--surface2);padding:11px 13px;display:flex;
flex-direction:column;gap:2px}}
.kpi .v{{font-family:ui-monospace,Menlo,monospace;font-size:1.15rem;
font-weight:600;letter-spacing:-.01em}}
.kpi .k{{font-size:.78rem;color:var(--ink2)}}
.kpi .d{{font-size:.72rem;color:var(--muted)}}
.strip{{position:relative;height:46px;margin:22px 0 6px}}
.strip .band{{position:absolute;top:14px;height:9px;border-radius:5px;
background:var(--band)}}
.strip .band2{{position:absolute;top:14px;height:9px;border-radius:5px;
background:var(--band2)}}
.strip .med{{position:absolute;top:8px;height:21px;width:2px;background:var(--ink)}}
.strip .lb{{position:absolute;top:30px;transform:translateX(-50%);
font-family:ui-monospace,Menlo,monospace;font-size:.63rem;color:var(--muted);
white-space:nowrap}}
.note{{font-size:.8rem;color:var(--ink2);margin:9px 0}}
.warn{{font-size:.82rem;color:var(--warn);margin:11px 0;
background:var(--surface2);border-left:3px solid var(--warn);
border-radius:0 7px 7px 0;padding:9px 12px}}
.caveat{{font-size:.76rem;color:var(--muted);margin-top:16px;
padding-top:12px;border-top:1px solid var(--border);line-height:1.75}}
.tw{{overflow-x:auto;margin:10px 0}}
table{{width:100%;border-collapse:collapse;font-size:.84rem;min-width:520px}}
th{{text-align:left;font-size:.65rem;letter-spacing:.06em;color:var(--muted);
font-weight:500;padding:0 11px 7px 0;border-bottom:1px solid var(--border)}}
td{{padding:9px 11px 9px 0;border-bottom:1px solid var(--border)}}
tr:last-child td{{border-bottom:0}}
td.d{{font-family:ui-monospace,Menlo,monospace;white-space:nowrap}}
tr.hi td{{font-weight:600}}
tr.cur td{{background:var(--accs)}}
.pill{{font-family:ui-monospace,Menlo,monospace;font-size:.6rem;font-weight:500;
border-radius:4px;padding:1px 6px;margin-left:6px;background:var(--accs);
color:var(--acc);white-space:nowrap}}
.pill.stop{{background:var(--stops);color:var(--stop);margin-left:0}}
.pill.good{{background:var(--goods);color:var(--good)}}
.pill.mute{{background:var(--band);color:var(--muted);margin-left:0}}
.qbar{{display:inline-block;vertical-align:middle;width:60px;height:6px;
border-radius:3px;background:var(--band);margin-right:8px;overflow:hidden}}
.qbar i{{display:block;height:100%;background:var(--acc);border-radius:3px}}
footer{{font-size:.76rem;color:var(--muted);margin-top:18px;line-height:1.7}}
</style></head><body><div class="wrap">
<h1>여름 손실 · 간격 what-if</h1>
<p class="sub">두 판 다 <b>서버 없이</b> 열립니다. 값은 구울 때 한 번 계산해
박아 넣은 것입니다.</p>
<div class="demo"><b>예시 농장입니다 — 실제 농장이 아닙니다.</b><br>
모돈 {DEMO_SOWS}두 기본 구성({esc(barns)}) · 분만율
{DEMO_PERF["farrowing_rate"]}% · 복당 이유 {DEMO_PERF["weaned"]}두 ·
육성률 {DEMO_PERF["survival"]}%.<br>
<b>우리 농장 값을 넣으려면 서버를 띄워야 합니다</b> —
<code>python -m uvicorn competition.server.app:app --port 8000</code>
</div>
{cards}
<footer>산식은 <code>src/</code> 의 도메인 모듈이 정본이고, 이 화면은 그걸
구울 때 한 번 부른 것입니다 — <b>여기서 다시 계산하지 않습니다.</b><br>
서버로 본 금액과 이 파일로 본 금액은 같은 함수에서 나옵니다
(<code>season.compute</code> · <code>capacity.interval_whatif</code>).
</footer></div></body></html>"""


def main() -> int:
    d = gather()
    html = build(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    s, iw = d["season"], d["interval"]
    print(f"여름 손실 · 간격 what-if 생성: {OUT} "
          f"({len(html) / 1024:.0f}KB)")
    print(f"  여름 손실  {s['n_farms']}농장 · 중앙 시나리오 "
          f"{man(s['scenario']['median']['won_year'])} · 패널 실측 중앙 "
          f"{man(s['panel_won_ref']['median'])}")
    ok = [r for r in iw["rows"] if r["n_sows"] > 0]
    print(f"  간격       {len(iw['rows'])}개 중 {len(ok)}개 성립 · "
          f"규모 최대 {iw['best']:g}일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
