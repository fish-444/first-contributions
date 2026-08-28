"""환경 이상치 화면 — 온·습·환기 사분위 밴드 + 날짜별 추이 + 알림 목록.

`env_anomaly.py` 가 낸 판정을 **한 장**으로 본다. 그래프는 필드마다 따로
그린다 — 온도·습도·환기율은 단위가 달라서 한 축에 겹쳐 그리면 눈금이 뜻을
잃는다(환기율 1~3 이 습도 40~90 옆에서 납작해진다).

각 그래프에 세 가지가 같이 있다:

  · 사분위 밴드   Q1~Q4 구간을 배경 음영으로. "지금 어느 수준인가"
  · 날짜별 평균선 하루 평균. "어디로 가고 있나"
  · 이상치 점     그날 알림이 난 날. "언제 튀었나"

수준(사분위)과 튐(z)을 같은 그림에 두는 게 요점이다. 같은 +3z 도 환기 Q1 에서
난 것과 Q4 에서 난 것은 다른 조치라서, 둘을 따로 보면 판단이 안 된다.

**발송하지 않는다.** 화면 위 배너와 목록까지가 이 파일의 일이다.

    python competition/src/build_env_anomaly.py
출력: competition/dashboard/env_anomaly.html  (외부 연결·라이브러리 불필요)
"""
from __future__ import annotations

import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

SRC = os.path.join(ROOT, "competition", "data", "env_anomaly.json")
OUT = os.path.join(ROOT, "competition", "dashboard", "env_anomaly.html")

# 사분위 배경색 — Q1 이 낮고 Q4 가 높다. 색으로 좋고 나쁨을 말하지 않는다.
# 환기율은 낮은 쪽이 문제고 온도는 높은 쪽이 문제라, 같은 색을 좋다/나쁘다로
# 읽히게 두면 필드마다 뜻이 뒤집힌다.
Q_FILL = ["#eef2f7", "#e3eaf3", "#d8e2ef", "#cddaeb"]
LINE = "#2b6cb0"
BAD = "#d03b3b"
W, H, PADL, PADR, PADT, PADB = 720, 150, 52, 12, 14, 26


def _fmt_date(d: str) -> str:
    """YYMMDD → YY-MM-DD. 6자리가 아니면 그대로 둔다(지어내지 않는다)."""
    d = str(d)
    return "%s-%s-%s" % (d[:2], d[2:4], d[4:6]) if len(d) == 6 else d


def chart(field: str, meta: dict, daily: list, alert_dates: set) -> str:
    """필드 하나의 SVG — 사분위 밴드 + 날짜별 평균선 + 이상치 점.

    점이 찍히는 곳은 **그 필드에 알림이 난 날**이다. 다른 필드의 알림은 그
    필드 그래프에만 찍힌다 — 한 그래프에 다 모으면 어느 값이 튄 건지 모른다.
    """
    pts = [(r["date"], r.get(field)) for r in daily
           if r.get(field) is not None]
    if len(pts) < 2:
        return ('<div class="nodata">날짜별 추이를 그릴 수 없습니다 — '
                '날짜가 붙은 관측이 2일 미만입니다. 순서를 시간축으로 '
                '대신 쓰지 않습니다.</div>')

    cuts = meta.get("quartile_cuts") or []
    ys = [v for _, v in pts] + [c for c in cuts if c is not None]
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    def sx(i):
        return PADL + i * (W - PADL - PADR) / max(1, len(pts) - 1)

    def sy(v):
        return PADT + (hi - v) / (hi - lo) * (H - PADT - PADB)

    out = ['<svg viewBox="0 0 %d %d" class="chart" '
           'preserveAspectRatio="none">' % (W, H)]

    # 사분위 밴드 — 경계는 클립 단위 값의 사분위다(일평균의 사분위가 아니다)
    edges = [lo] + [c for c in cuts] + [hi]
    for i in range(len(edges) - 1):
        y0, y1 = sy(edges[i + 1]), sy(edges[i])
        out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                   'fill="%s"/>' % (PADL, y0, W - PADL - PADR,
                                    max(0.0, y1 - y0),
                                    Q_FILL[min(i, len(Q_FILL) - 1)]))
        if y1 - y0 > 13:
            out.append('<text x="%.1f" y="%.1f" class="qlab">Q%d</text>'
                       % (PADL + 4, (y0 + y1) / 2 + 3, i + 1))

    for v in (lo, (lo + hi) / 2, hi):
        out.append('<text x="%d" y="%.1f" class="ax">%.1f</text>'
                   % (PADL - 6, sy(v) + 3, v))

    line = " ".join("%.1f,%.1f" % (sx(i), sy(v))
                    for i, (_, v) in enumerate(pts))
    out.append('<polyline points="%s" fill="none" stroke="%s" '
               'stroke-width="1.6"/>' % (line, LINE))

    for i, (d, v) in enumerate(pts):
        if d in alert_dates:
            out.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="%s"/>'
                       % (sx(i), sy(v), BAD))

    step = max(1, len(pts) // 8)
    for i in range(0, len(pts), step):
        out.append('<text x="%.1f" y="%d" class="ax mid">%s</text>'
                   % (sx(i), H - 6, _fmt_date(pts[i][0])))
    out.append("</svg>")
    return "".join(out)


def build(r: dict) -> str:
    fields = r.get("fields", {})
    daily = r.get("daily", [])
    alerts = r.get("alerts", [])
    by_field_dates: dict = {}
    for a in alerts:
        if a.get("date"):
            by_field_dates.setdefault(a["field"], set()).add(a["date"])

    synth = "합성" in str(r.get("source", ""))
    banner = ""
    if synth:
        banner += ('<div class="warn"><b>합성 자료입니다.</b> 배관 검증용이며 '
                   '현장 숫자가 아닙니다 — 71763 실라벨로 다시 돌려야 '
                   '화면의 값이 뜻을 갖습니다.</div>')
    n = r.get("n_alerts", 0)
    banner += ('<div class="%s"><b>알림 %d건</b> — 클립 %s개 중. '
               '이 화면은 목록과 표시까지입니다. <b>발송하지 않습니다.</b></div>'
               % ("alertbar" if n else "okbar", n,
                  format(r.get("n_clips", 0), ",")))

    cards = []
    for f, m in fields.items():
        if "z" not in m:
            cards.append('<div class="card"><h3>%s</h3><div class="nodata">'
                         '판정 불가 — %s (n=%d)</div></div>'
                         % (html.escape(m["label"]),
                            html.escape(m.get("usable", "")), m.get("n", 0)))
            continue
        qrows = "".join(
            '<tr><td><b>%s</b></td><td>%s</td><td>%s</td>'
            '<td class="%s">%s건 (%.1f%%)</td></tr>'
            % (q["q"], format(q["n"], ","), q["value_mean"],
               "bad" if q["n_flagged"] else "", format(q["n_flagged"], ","),
               q["flagged_pct"])
            for q in m["by_quartile"])
        note = ""
        if m["usable"] != "쓸만함":
            note = ('<div class="note bad">문턱 판정: <b>%s</b> — 알림률 %.1f%% '
                    '가 목표 대역(0.5~5%%) 밖입니다. 이 필드의 알림은 그대로 '
                    '쓰지 마십시오.</div>' % (m["usable"], m["flagged_pct"]))
        if m["n_bins"] < 4:
            note += ('<div class="note">사분위가 %d칸만 나왔습니다 — 같은 값이 '
                     '4분의 1을 넘게 차지해 경계가 겹칩니다.</div>'
                     % m["n_bins"])
        cards.append(
            '<div class="card"><h3>%s</h3>'
            '<div class="meta">기준 <b>%s</b>(설명력 %.1f%%) · 평균 %s · '
            '로버스트 SD %s · 문턱 z <b>%.1f</b> · 알림 %s건(%.1f%%) '
            '<span class="tag %s">%s</span></div>'
            '%s%s'
            '<table class="q"><thead><tr><th>사분위</th><th>관측</th>'
            '<th>평균값</th><th>이상치</th></tr></thead><tbody>%s</tbody>'
            '</table></div>'
            % (html.escape(m["label"]), html.escape(str(m["center"])),
               m["center_explains_pct"], m["mean"], m["robust_sd"], m["z"],
               format(m["n_flagged"], ","), m["flagged_pct"],
               "ok" if m["usable"] == "쓸만함" else "bad", m["usable"],
               chart(f, m, daily, by_field_dates.get(f, set())), note, qrows))

    rows = "".join(
        '<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
        '<td class="%s">%+.2f</td><td>%s</td><td>%s</td></tr>'
        % (_fmt_date(a.get("date") or "-"), html.escape(a.get("chamber") or "-"),
           html.escape(a["label"]), a["value"],
           "bad" if a["z"] > 0 else "low", a["z"], a["quartile"],
           a["direction"])
        for a in alerts[:40])

    more = ("" if len(alerts) <= 40 else
            '<div class="note">상위 40건만 표시했습니다 — 전체 %s건은 '
            '<code>outputs/env_alerts.csv</code> 에 있습니다.</div>'
            % format(len(alerts), ","))

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>환경 이상치 — 온·습·환기</title><style>
:root{{--bg:#f7f7f5;--fg:#22201d;--muted:#7c7a75;--line:#e3e1dc;--bad:#d03b3b;
--good:#1baf7a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.6 -apple-system,"Segoe UI","Malgun Gothic",sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:22px 16px 60px}}
h1{{font-size:1.32rem;margin:0 0 4px}}
h3{{font-size:1rem;margin:0 0 8px}}
.sub{{color:var(--muted);font-size:.86rem;margin-bottom:16px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:9px;
padding:14px 15px;margin-bottom:14px}}
.meta{{color:var(--muted);font-size:.82rem;margin-bottom:9px}}
.chart{{width:100%;height:150px;display:block;margin:6px 0 4px;
border:1px solid var(--line);border-radius:5px;background:#fff}}
.ax{{font-size:9px;fill:#8b8983;text-anchor:end}}
.ax.mid{{text-anchor:middle}}
.qlab{{font-size:9px;fill:#9b9992}}
table{{width:100%;border-collapse:collapse;font-size:.83rem}}
th,td{{text-align:left;padding:5px 7px;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-weight:600}}
td.bad,.bad{{color:var(--bad)}}
td.low{{color:#2b6cb0}}
.tag{{padding:1px 6px;border-radius:9px;font-size:.74rem;border:1px solid}}
.tag.ok{{color:var(--good);border-color:var(--good)}}
.tag.bad{{color:var(--bad);border-color:var(--bad)}}
.alertbar,.okbar,.warn{{padding:10px 13px;border-radius:8px;margin-bottom:11px;
font-size:.88rem}}
.alertbar{{background:#fdecec;border:1px solid #f0b8b8}}
.okbar{{background:#eaf7f1;border:1px solid #b6e2cd}}
.warn{{background:#fff6e5;border:1px solid #f0d9a8}}
.note{{color:var(--muted);font-size:.8rem;margin:6px 0}}
.nodata{{color:var(--muted);font-size:.83rem;padding:10px 0}}
code{{background:#efedea;padding:1px 4px;border-radius:3px;font-size:.9em}}
</style></head><body><div class="wrap">
<h1>환경 이상치 — 온도 · 습도 · 환기율</h1>
<div class="sub">기준 집단 안의 편차로 판정합니다(절대 문턱 아님). 문턱은
필드마다 따로 잡습니다 — 전 필드에 3σ 를 일괄로 물리면 어떤 필드는 한 건도
안 걸리고 어떤 필드는 10% 가 걸립니다. 자료: {html.escape(str(r.get('source','-')))}</div>
{banner}
{''.join(cards)}
<div class="card"><h3>알림 대상</h3>
<table><thead><tr><th>날짜</th><th>방</th><th>필드</th><th>값</th><th>z</th>
<th>사분위</th><th>방향</th></tr></thead><tbody>{rows or
'<tr><td colspan="7" class="nodata">알림 없음</td></tr>'}</tbody></table>
{more}
<div class="note">{html.escape(str(r.get('note','')))}</div>
</div></div></body></html>"""


def main() -> int:
    if not os.path.exists(SRC):
        print("먼저 env_anomaly.py 를 돌려 %s 를 만들어야 합니다." % SRC)
        return 2
    r = json.load(open(SRC, encoding="utf-8"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(build(r))
    print("%s · 알림 %d건 · 필드 %d개"
          % (OUT, r.get("n_alerts", 0), len(r.get("fields", {}))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
