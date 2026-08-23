"""교배 배정 · 환경 알람 — **서버 없이 열리는** 판.

두 기능이 CLI 와 API 에만 있었다. 심사장에서 화면으로 보는데 최근 기능이
화면에 없으면 없는 기능이다 — `season_interval` 을 만든 것과 같은 이유로
여기서 되돌린다.

**계산은 여기서 하지 않는다.** `mating_plan.plan()` 과
`barn_env_control.assess()` 를 그대로 부른다 — 라우터가 쓰는 그 함수다.
구울 때 한 번 부르고 값을 박아 넣을 뿐이라, 서버로 본 화면과 파일로 본
화면이 같은 수를 말한다.

박아 넣는 값이므로 **입력을 바꿀 수 없다.** 우리 농장 값을 넣으려면 서버를
띄워야 하고(`POST /api/ops/mating`·`/env`), 화면이 그 사실을 먼저 말한다.

이 판이 보여 주려는 것 둘:

- **근친이 인덱스를 이긴다** — 인덱스 최고 웅돈이 반형매 모돈에는 막혀
  차선으로 비켜간다. 모돈별 최고를 주는 탐욕이 아니라 농장 전체 최적이다.
- **센서 차이를 사육환경 차이로 읽지 않는다** — 같은 환경인데 센서만
  치우친 돈사는 원값으로 더워 보이지만 편차(z)에는 흔적이 없다.

    python competition/src/build_ops_console.py
출력: competition/dashboard/ops_console.html
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # .../competition
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

OUT = os.path.join(ROOT, "dashboard", "ops_console.html")

C_ACC, C_BAD, C_GOOD, C_WARN = "#2a78d6", "#d03b3b", "#1baf7a", "#e8a33d"


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def bold(s: str) -> str:
    out, parts = [], esc(s).split("**")
    for i, p in enumerate(parts):
        out.append(f"<b>{p}</b>" if i % 2 else p)
    return "".join(out)


def gather() -> dict:
    """모듈의 시연 구성을 그대로 쓴다 — 여기서 수를 짓지 않는다."""
    import barn_env_control as ec
    import mating_plan as mp

    log, stages = ec._demo()
    return {"mating": mp._demo(), "env": ec.assess(log, stages),
            "guide_nh3": ec.NH3_LIMIT, "guide_h2s": ec.H2S_LIMIT}


def mating_card(r: dict) -> str:
    rows = "".join(
        f"<tr><td>{esc(x['모돈번호'])}</td><td class=n>{x['모돈인덱스']:g}</td>"
        f"<td>{esc(x['웅돈번호'])}</td><td class=n>{x['웅돈인덱스']:g}</td>"
        f"<td class='n hi'>{x['후손의 예상인덱스']:g}</td>"
        f"<td class=n>{x['근친율(%)']:g}%</td>"
        f"<td class=n>{x['교배횟수']}</td></tr>" for x in r["rows"])
    un = "".join(
        f'<div class="warn">미배정 {esc(u["모돈번호"])} — {esc(u["사유"])}</div>'
        for u in r["unassigned"])
    use = " · ".join(f"{esc(b)} {v['배정']}/{v['상한']}"
                     for b, v in r["boar_use"].items())
    notes = "".join(f"<li>{bold(n)}</li>" for n in r["notes"])
    return f"""<div class="card">
<div class="ch"><span class="cn">교배 배정</span>
<b>근친 한도 {r['max_f'] * 100:g}% 아래에서 예상인덱스 합 최대화</b></div>
<p class="sub">평균 예상인덱스 <b>{r['mean_expected_index']}</b> ·
웅돈 사용 {esc(use)}</p>
<table><thead><tr><th>모돈번호</th><th>모돈인덱스</th><th>웅돈번호</th>
<th>웅돈인덱스</th><th>후손 예상인덱스</th><th>근친율</th><th>교배횟수</th>
</tr></thead><tbody>{rows}</tbody></table>{un}
<p class="lead">인덱스가 가장 높은 웅돈이 반형매 모돈에게는 근친
한도에 걸려 차선으로 비켜갔다 — 모돈별 최고가 아니라 <b>농장 전체
최적</b>이다. 수치는 위 표가 정본이다.</p>
<ul class="notes">{notes}</ul></div>"""


def env_card(r: dict, nh3: float, h2s: float) -> str:
    cells = []
    for barn, d in r["barns"].items():
        svs = []
        for k, v in d["sensors"].items():
            zt = (" · 기준선 미형성" if v["z"] is None
                  else " · z %+.1f" % v["z"])
            bad = "bad" if v["guide_state"] != "적정" else ""
            svs.append(f'<div class="sv"><span class="sk">{esc(k)}</span>'
                       f'<span class="svv">{v["now"]:.1f}</span>'
                       f'<span class="sz {bad}">{esc(v["guide_state"])}'
                       f'{esc(zt)}</span></div>')
        sens = "".join(svs)
        al = "".join(
            f'<div class="al {"lv1" if a["수준"] == "위험" else "lv2"}">'
            f'[{esc(a["수준"])}] {esc(a["내용"])}</div>' for a in d["alarms"])
        cells.append(f'<div class="barn"><div class="bh">{esc(barn)}'
                     f'<span class="stg">{esc(d["stage"])}</span></div>'
                     f'{sens}{al or "<div class=ok>알람 없음</div>"}</div>')
    rank = " · ".join(f"{esc(t['돈사'])}/{esc(t['센서'])} {t['z']:+.1f}"
                      for t in r["ranking"][:3])
    notes = "".join(f"<li>{bold(n)}</li>" for n in r["notes"])
    return f"""<div class="card">
<div class="ch"><span class="cn">환경 알람</span>
<b>위험은 지침 위반 · 주의는 평소와 다름 — 제어 지시는 내지 않는다</b></div>
<p class="sub">지침 NH₃ {nh3:g}ppm · H₂S {h2s:g}ppm ·
편차 순위(|z|) {esc(rank)}</p>
<div class="barns">{''.join(cells)}</div>
<p class="lead">2동은 1동과 <b>같은 환경</b>인데 센서만 치우쳐 있다.
원값으로 비교하면 더워 보이지만 자기 기준선 편차(z)에는 흔적이 없고,
알람에 <b>“센서 치우침, 교정 확인”</b> 이 붙는다 — 센서 차이를 사육환경
차이로 읽지 않는 자리다. 수치는 위 돈사 칸이 정본이다.</p>
<ul class="notes">{notes}</ul></div>"""


def page(d: dict) -> str:
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>교배 배정 · 환경 알람</title><style>
:root{{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--surface2:#f2f2ee;
--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--border:rgba(11,11,11,.12);
--acc:{C_ACC};--accs:rgba(42,120,214,.10);--stop:{C_BAD};
--stops:rgba(208,59,59,.12);--good:{C_GOOD};--goods:rgba(27,175,122,.14);
--warn:{C_WARN};--warns:rgba(232,163,61,.14)}}
@media(prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{
--page:#0d0d0d;--surface:#1a1a19;--surface2:#242422;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--border:rgba(255,255,255,.14);--acc:#5fa8f0;
--accs:rgba(95,168,240,.14);--stop:#f07070;--stops:rgba(240,112,112,.15);
--good:#4fd6a0;--goods:rgba(79,214,160,.16);--warn:#e8b45e;
--warns:rgba(232,180,94,.16)}}}}
:root[data-theme=dark]{{--page:#0d0d0d;--surface:#1a1a19;--surface2:#242422;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--border:rgba(255,255,255,.14);
--acc:#5fa8f0;--accs:rgba(95,168,240,.14);--stop:#f07070;
--stops:rgba(240,112,112,.15);--good:#4fd6a0;--goods:rgba(79,214,160,.16);
--warn:#e8b45e;--warns:rgba(232,180,94,.16)}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif;
background:var(--page);color:var(--ink);line-height:1.55;padding:22px}}
.wrap{{max-width:860px;margin:0 auto}}
h1{{font-size:1.55rem;letter-spacing:-.02em}}
.sub{{color:var(--ink2);font-size:.92rem;margin:6px 0}}
.demo{{font-size:.82rem;color:var(--ink2);background:var(--surface2);
border:1px solid var(--border);border-radius:9px;padding:9px 12px;
margin:12px 0 18px}}
.card{{background:var(--surface);border:1px solid var(--border);
border-radius:13px;padding:17px 18px;margin-bottom:15px}}
.ch{{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}}
.cn{{font-size:.72rem;font-weight:700;color:var(--page);background:var(--ink);
border-radius:5px;padding:2px 7px}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9rem}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--border);text-align:left}}
th{{font-size:.78rem;color:var(--ink2);font-weight:600}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
td.hi{{color:var(--acc);font-weight:700}}
.warn{{background:var(--warns);border-radius:8px;padding:7px 10px;
font-size:.86rem;margin-top:6px}}
.lead{{margin-top:11px;font-size:.9rem;color:var(--ink2);
background:var(--accs);border-radius:9px;padding:10px 12px}}
.notes{{margin:11px 0 0 18px;font-size:.82rem;color:var(--muted)}}
.notes li{{margin-top:4px}}
.barns{{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
margin-top:12px}}
.barn{{border:1px solid var(--border);border-radius:10px;padding:11px 12px;
background:var(--surface2)}}
.bh{{font-weight:700;display:flex;justify-content:space-between;
align-items:baseline;margin-bottom:6px}}
.stg{{font-size:.74rem;color:var(--muted);font-weight:400}}
.sv{{display:flex;gap:7px;align-items:baseline;font-size:.84rem;
padding:2px 0;flex-wrap:wrap}}
.sk{{color:var(--ink2);min-width:66px}}
.svv{{font-variant-numeric:tabular-nums;font-weight:600}}
.sz{{font-size:.76rem;color:var(--muted)}}
.sz.bad{{color:var(--stop);font-weight:600}}
.al{{margin-top:7px;border-radius:8px;padding:7px 9px;font-size:.82rem}}
.al.lv1{{background:var(--stops);color:var(--stop)}}
.al.lv2{{background:var(--warns)}}
.ok{{margin-top:7px;font-size:.8rem;color:var(--good)}}
</style><div class="wrap">
<h1>교배 배정 · 환경 알람</h1>
<p class="sub">근친 한도 아래 전체 최적 배정 · 지침 위반과 자기 기준선
편차를 겹으로 본 알람</p>
<div class="demo"><b>등급 합성</b> — 예시 구성으로 구운 화면이다. 실제
농장 성적이 아니고, <b>입력을 바꾸려면 서버가 필요하다</b>
(<code>POST /api/ops/mating</code> · <code>/api/ops/env</code>).
계산은 <code>mating_plan</code> · <code>barn_env_control</code> 이 하고
이 파일은 값을 받아 그리기만 한다.</div>
{mating_card(d['mating'])}
{env_card(d['env'], d['guide_nh3'], d['guide_h2s'])}
</div></html>"""


def main() -> int:
    d = gather()
    html = page(d)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  교배 배정 {len(d['mating']['rows'])}건 · "
          f"환경 돈사 {len(d['env']['barns'])}개 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
