# -*- coding: utf-8 -*-
"""농장 화이트보드 교배표 → HTML 대시보드.

현장 보드의 스키마를 그대로 따른다:
  배치# · 교배 시작일 · 교배 그룹(이유돈/재교배) · 임신 주차 1~17 잔존 두수
  (재발 발견 주에 빨간 숫자) · 포유 1~4주 · 이유 모돈 · 이유 두수

계산 재구현 금지 — 데이터는 synth_farm.generate()(실측 분포 재현·검증),
상수는 breeding_timing(임신 115·포유 28·재발주기 21)에서 읽는다.
재발 '발견' 주는 같은 모돈의 다음 발정일(데이터)이고, 다음 기록이 없으면
교배+21일(지침)로 대신하며 그 건수를 각주에 적는다.
"""
import math
import os
import sys
from collections import defaultdict
from datetime import date, timedelta

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(COMP, "src"))
import breeding_timing as bt  # noqa: E402
import synth_farm as sf       # noqa: E402

GEST_WEEKS = math.ceil(bt.GESTATION / 7)      # 17
LACT_WEEKS = math.ceil(bt.LACTATION / 7)      # 4
N_BATCH = 18

df = sf.generate(n_sows=300, years=1.0, start="2025-01-01", seed=0)
df = df.sort_values(["sow_id", "service"]).reset_index(drop=True)

# 재발 발견일: 같은 모돈의 다음 발정(데이터). 없으면 교배+21일(지침).
next_estrus = {}
fallback_n = 0
for sid, g in df.groupby("sow_id"):
    rows = g.reset_index()
    for k in range(len(rows)):
        r = rows.loc[k]
        if r["outcome"] != "재발":
            continue
        if k + 1 < len(rows):
            next_estrus[r["index"]] = rows.loc[k + 1, "estrus"]
        else:
            next_estrus[r["index"]] = r["service"] + timedelta(days=bt.RETURN_CYCLE)
            fallback_n += 1

# 직전 주기가 재발이면 이번 교배는 '재교배'
prev_outcome = df.groupby("sow_id")["outcome"].shift(1)
df["group"] = ["재교배" if p == "재발" else "이유돈" for p in prev_outcome]

# 주 단위 배치(월요일 시작)
df["batch_week"] = df["service"].map(lambda d: d - timedelta(days=d.weekday()))
weeks = sorted(df["batch_week"].unique())[:N_BATCH]

batches = []
for bi, wk in enumerate(weeks, start=1):
    b = df[df["batch_week"] == wk]
    bred = len(b)
    n_wean_grp = int((b["group"] == "이유돈").sum())
    n_ret_grp = int((b["group"] == "재교배").sum())

    # 재발 발견 주차별 이탈 목록
    drops = defaultdict(list)
    for idx, r in b.iterrows():
        if r["outcome"] != "재발":
            continue
        det = next_estrus[idx]
        wno = max(1, min(GEST_WEEKS, (det - r["service"]).days // 7 + 1))
        drops[wno].append(f'{r["sow_id"]}({r["return_type"]})')

    cells, remain = [], bred
    for w in range(1, GEST_WEEKS + 1):
        d = drops.get(w, [])
        remain -= len(d)
        cells.append((remain, len(d), d))

    far = b[b["outcome"] == "분만"]
    lact_cells = []
    for lw in range(1, LACT_WEEKS + 1):
        nursing = int(sum(1 for _, r in far.iterrows()
                          if (r["wean"] - r["farrow"]).days > (lw - 1) * 7))
        lact_cells.append(nursing)
    batches.append(dict(no=bi, start=wk, bred=bred, wean_grp=n_wean_grp,
                        ret_grp=n_ret_grp, cells=cells, lact=lact_cells,
                        weaned_sows=len(far),
                        pigs_weaned=int(far["weaned"].sum())))

v = sf.validate(df)
ok = bool(v.get("ok"))
chk = " · ".join(f'{c["name"]} {c["got"]}/{c["want"]}{c.get("unit","")}'
                 for c in v["checks"])

wk_heads = "".join(f"<th>{w}</th>" for w in range(1, GEST_WEEKS + 1))
lact_heads = "".join(f"<th>{w}</th>" for w in range(1, LACT_WEEKS + 1))
rows_html = []
for b in batches:
    tds = []
    for remain, ndrop, d in b["cells"]:
        if ndrop:
            tip = " · ".join(d)
            tds.append(f'<td class="drop" title="재발 발견 {ndrop}두: {tip}">{remain}</td>')
        else:
            tds.append(f"<td>{remain}</td>")
    lact_tds = "".join(f'<td class="lac">{n if n else ""}</td>' for n in b["lact"])
    rows_html.append(
        f'<tr><td class="bn">{b["no"]}</td>'
        f'<td class="dt">{b["start"].strftime("%m/%d")}</td>'
        f'<td class="grp">{b["wean_grp"]}</td><td class="grp r">{b["ret_grp"]}</td>'
        f'<td class="tot">{b["bred"]}</td>'
        + "".join(tds) + lact_tds +
        f'<td class="ws">{b["weaned_sows"]}</td><td class="pw">{b["pigs_weaned"]}</td></tr>')

html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>교배표 보드 — 배치×주차 (합성 시연)</title>
<style>
:root{{--bd:#2b2b2b;--blue:#1d4ed8;--red:#d03b3b;--muted:#6b7280}}
*{{box-sizing:border-box}}
body{{margin:0;background:#e8e6e0;font:14px/1.5 -apple-system,'Malgun Gothic',sans-serif;color:#111}}
.wrap{{max-width:1280px;margin:18px auto;padding:0 14px}}
h1{{font-size:1.05rem;margin:0 0 2px}}
.sub{{color:var(--muted);font-size:.8rem;margin-bottom:10px}}
.board{{background:#fdfdfb;border:3px solid var(--bd);border-radius:6px;
  box-shadow:0 3px 14px rgba(0,0,0,.18);overflow-x:auto;padding:6px}}
table{{border-collapse:collapse;width:100%;min-width:1150px}}
th,td{{border:1px solid var(--bd);text-align:center;padding:3px 4px;
  font-variant-numeric:tabular-nums}}
thead th{{background:#f3f1ea;font-size:.72rem;letter-spacing:.02em}}
thead .sec{{background:#eae7de;font-size:.78rem}}
td{{color:var(--blue);font-weight:600}}
td.bn{{color:#111;background:#f3f1ea}} td.dt{{color:#111;font-weight:400}}
td.grp{{color:#111;font-weight:400}} td.grp.r{{color:var(--red)}}
td.tot{{background:#f8f7f2;color:#111}}
td.drop{{color:var(--red);background:#fdeaea;cursor:help}}
td.lac{{color:#0e7a3d}} td.ws,td.pw{{color:#111;background:#f8f7f2}}
tbody tr:hover td{{background:#fffbe6}}
tbody tr:hover td.drop{{background:#fbdcdc}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:.76rem;color:#333;margin:10px 2px}}
.legend b.b{{color:var(--blue)}} .legend b.r{{color:var(--red)}} .legend b.g{{color:#0e7a3d}}
.note{{font-size:.74rem;color:var(--muted);margin-top:10px;line-height:1.7}}
.badge{{display:inline-block;font-size:.68rem;border:1px solid #999;border-radius:999px;
  padding:0 8px;margin-right:6px;color:#444;background:#fff}}
@media print{{body{{background:#fff}}.board{{box-shadow:none}}.note{{page-break-inside:avoid}}}}
</style></head><body><div class="wrap">
<h1>교배표 보드 <span class="badge">합성 시연 — 실제 농장 아님</span></h1>
<div class="sub">현장 화이트보드 서식 그대로 — 배치(주간) × 임신 주차 잔존, 재발은 발견 주에 붉게.
마우스를 붉은 칸에 올리면 이탈 개체가 나온다.</div>
<div class="board"><table>
<thead>
<tr><th class="sec" colspan="2">배치</th><th class="sec" colspan="3">교배 그룹</th>
<th class="sec" colspan="{GEST_WEEKS}">임신 주차 잔존 두수 (분만 예정 {GEST_WEEKS}주차 · {bt.GESTATION}일)</th>
<th class="sec" colspan="{LACT_WEEKS}">포유(주)</th><th class="sec" colspan="2">이유</th></tr>
<tr><th>#</th><th>시작일</th><th>이유돈</th><th>재교배</th><th>계</th>
{wk_heads}{lact_heads}<th>모돈</th><th>자돈</th></tr>
</thead>
<tbody>{"".join(rows_html)}</tbody>
</table></div>
<div class="legend">
<span><b class="b">파랑</b> 임신 유지 두수</span>
<span><b class="r">빨강</b> 그 주에 재발 발견 → 줄어든 값 (올리면 개체·유형)</span>
<span><b class="g">초록</b> 포유 중 모돈</span>
<span>재교배 열이 <b class="r">붉은</b> 것은 직전 주기가 재발이었던 교배</span>
</div>
<div class="note">
<span class="badge">합성</span>수치는 synth_farm 이 국내 실측 분포(재귀발정일·분만율·복당이유·임신사고
구성)를 재현해 만든 것이며 생성 후 자동 검증을 {"통과했다" if ok else "통과하지 못했다"} — {chk}.
개체 이질성·산차 분포는 실측에 없는 가정이다. 실제 농장 이력(여섯 열 CSV)을 넣으면
같은 보드가 그 농장의 값으로 채워진다.<br>
<span class="badge">지침·계산</span>임신 {bt.GESTATION}일 → {GEST_WEEKS}주차 분만 · 포유 {bt.LACTATION}일 →
{LACT_WEEKS}주 · 재발 발견 주는 같은 개체의 다음 발정일(데이터)이고, 다음 기록이 없는
{fallback_n}건은 교배 +{bt.RETURN_CYCLE}일(재발정 주기, 지침)로 대신했다.<br>
외부 연결 0 · 농장 식별자 0 — 개체 번호는 합성 일련번호다. 생성: build_breeding_board(등록 뷰 아님 —
수치 동결로 뷰 25 레지스트리는 그대로 두었다).
</div>
</div></body></html>"""

out = os.path.join(COMP, "dashboard", "breeding_board.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"OK {out} · 배치 {len(batches)} · 재발 fallback {fallback_n}건")
print("검증:", "통과" if ok else "실패", "|", chk)
