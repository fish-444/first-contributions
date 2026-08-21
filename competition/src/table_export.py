"""표 내보내기 — 화면에서 본 것을 파일로 가져간다.

농가·심사위원이 결과를 엑셀로 열어 보려면 CSV 가 필요하다. 그런데 CSV 는
**서식이 없어서 등급과 각주가 통째로 사라진다** — 화면에서는 "실측" 배지와
"개입 효과가 아니다" 각주가 늘 붙어 다니는데, 표만 뽑으면 숫자만 남는다.
그러면 격차 분해가 개입 효과처럼, 유도값이 실측처럼 읽힌다. 이 프로젝트가
가장 조심해 온 것이 그 오독이다.

그래서 두 가지를 강제한다:

  1. **머리말이 붙는다** — 무엇을·언제·어느 등급으로 낸 표인지, 그리고
     그 축의 각주를 `#` 줄로 파일 맨 위에 적는다. 엑셀에서 A열 위쪽에
     그대로 보인다.
  2. **행마다 등급 열이 있다** — 머리말을 지우고 붙여넣어도 등급은 남는다.

`bare=True` 로 머리말을 뺄 수 있지만 등급 열은 못 뺀다. 기계가 먹을 때만
쓰는 것이고, 사람이 보는 파일에서 등급을 지우는 길은 열어 두지 않는다.

**여기서 계산하지 않는다.** 각 표는 기존 모듈·라우터의 출력을 평평하게
펴기만 한다.

    python competition/src/table_export.py --sheet capacity
"""
from __future__ import annotations

import csv
import io
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

BOM = "﻿"     # 엑셀이 UTF-8 한글을 깨뜨리지 않게

SHEETS = ("capacity", "interval", "diagnosis", "priority", "season", "targets")


def to_csv(rows: list, meta: dict, bare: bool = False) -> str:
    """행 + 머리말 → CSV 문자열.

    머리말은 `#` 로 시작해 엑셀에서 텍스트 줄로 보인다. 거슬리지만 **각주가
    파일을 떠나지 않는 것**이 더 중요하다 — 화면에서 각주와 함께 읽던 표가
    파일로 옮겨지면서 각주만 떨어져 나가는 게 이 프로젝트가 막으려는 일이다.
    """
    buf = io.StringIO()
    if not bare:
        buf.write(f"# {meta['title']}\n")
        buf.write(f"# 등급 {meta['grade']} · {meta['source']}\n")
        buf.write(f"# 생성 {meta.get('on') or date.today().isoformat()} · "
                  f"양돈 AI (competition/src/{meta['module']})\n")
        for c in meta.get("caveats", ()):
            buf.write(f"# ⚠ {c}\n")
        buf.write("#\n")
    if not rows:
        buf.write("(행 없음)\n")
        return BOM + buf.getvalue()
    cols = list(rows[0])
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n",
                       extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return BOM + buf.getvalue()


def _g(grade: str, rows: list) -> list:
    """등급 열을 **맨 앞에** 박는다. 머리말을 지워도 이건 남는다."""
    return [{"등급": grade, **r} for r in rows]


# --------------------------------------------------------------------------
def capacity_sheet(d: dict) -> tuple:
    """돈사별 지지 두수와 병목 — `capacity.compute()` 출력을 편다."""
    cap, tp = d["capacity"], d["throughput"]
    rows = [{
        "돈사": r["stage"], "방식": r["kind"], "방수": r["rooms"],
        "필요방수": r.get("need_rooms"), "방당자리": r["per"],
        "분만틀": r.get("crates"),
        "지지 모돈두수": r["sows"],
        "병목": "●" if r["stage"] == cap["binding"] else "",
        "막힘사유": r.get("why") or "",
    } for r in cap["rows"]]
    return _g("계산", rows), {
        "title": f"돈사별 지지 두수와 병목 — 이 농장 규모 {cap['n_sows']}두 "
                 f"(병목 {cap['binding']})",
        "grade": "계산", "module": "batch_flow.capacity_from_rooms",
        "source": f"등록 돈사 · 배치 간격 {cap['interval_days']:g}일",
        "caveats": [
            "가장 작은 칸이 이 농장의 규모다 — 돈방은 돈사를 건너뛰어 쓸 수 없다.",
            "막힌 돈사는 두수를 줄여서 풀리지 않는다. 방을 늘리거나 간격을 넓혀야 한다.",
            f"연간 출하 상한 {tp['ceiling_year']:,}두 · 지금 {tp['now_year']:,}두"
            + ("" if d["given"] else " (성적을 비웠으므로 '지금'은 설계 기준값이다)"),
        ]}


def interval_sheet(d: dict) -> tuple:
    """간격별 규모·병목 — `capacity.interval_whatif()` 출력을 편다."""
    rows = [{
        "간격": r["name"], "간격일": r["interval_days"],
        "지금": "●" if r["current"] else "",
        "받을 수 있는 모돈": r["n_sows"] or "",
        "병목": r["binding"] if r["n_sows"] else "막힘",
        "막힌 돈사": "·".join(r["blocked"]),
        "연간 출하 상한": r["ceiling_year"] if r["n_sows"] else "",
        "배치당 교배": r["services_per_batch"] if r["n_sows"] else "",
        "한 날 집중도": r["peak_ratio"],
    } for r in d["rows"]]
    return _g("계산", rows), {
        "title": f"같은 돈사로 간격만 바꾸면 — 지금 {d['current_interval']:g}일",
        "grade": "계산", "module": "batch_flow (server/routers/capacity)",
        "source": "등록 돈사 · 간격 7종",
        "caveats": [
            "규모가 가장 큰 간격이 늘 정답은 아니다 — 좁힐수록 방 수 요구가 "
            "커지고 작업이 잦아진다.",
            "인력·공사비를 계산하지 않는다.",
        ] + ([] if d["given"] else
             ["성적을 비웠으므로 출하는 설계 상한이고 지금 나오는 값이 아니다."])}


def diagnosis_sheet(d: dict) -> tuple:
    """466농장 대비 격차 — `farm_gap.diagnose()` 출력을 편다."""
    g = d["diagnosis"]
    rows = [{
        "지표": r["name_ko"], "내 값": r["value"], "중앙 농장": r["median"],
        "격차": r["gap"], "IQR 단위 거리": r["iqr_z"], "위치": r["band"],
        "중앙으로 되돌리면 PSY": r["psy_recover"],
        "고칠 수 있는 항목": "●" if r.get("actionable") else "",
    } for r in g["rows"]]
    return _g("실측", rows), {
        "title": f"국내 466농장 대비 격차 — PSY {g['psy']} "
                 f"(중앙 농장과 {g['psy_gap']:+.2f}두)",
        "grade": "실측", "module": "farm_gap.diagnose",
        "source": "국내 202농장 × 4년 = 466행 (2020~2023)",
        "caveats": [
            "**개입 효과가 아니다.** 격차의 분해이지 '고치면 오른다' 의 "
            "추정이 아니며, 실농장 개입 실험은 수행하지 않았다.",
            "**항목을 합산하지 말 것** — 지표가 항등식에서 곱해지므로 개별 "
            "회수량의 합은 총 격차와 다르다.",
            "비운 성적은 중앙값으로 채우지 않았다 — 채우면 격차가 늘 0 이 된다.",
        ]}


def priority_sheet(d: dict) -> tuple:
    """처방 순서 — `psy_priority.build()` 출력을 편다. 근거 등급이 행마다 붙는다."""
    p = d["priority"]
    rows = [{
        "순": i + 1, "항목": r["name"], "축": r["axis"],
        "회수/방어량": r["psy"], "원/년": r["won_year"],
        "근거등급": r["grade"], "표적": r["target"],
    } for i, r in enumerate(p["rows"])]
    return _g("실측", rows), {
        "title": f"처방 순서 — {p['n_sows']}두 · PSY {p['psy']} "
                 f"vs 중앙 농장 {p['psy_median_farm']}",
        "grade": "실측", "module": "psy_priority.build",
        "source": "466농장 격차 · 202농장 패널 · 67농장 계절",
        "caveats": [
            p["footer"].replace("\n", " "),
            p["sum_note"].replace("\n", " "),
            "**근거 등급 A/B/C 를 지우지 말 것** — A(농장 내 변화)와 "
            "B(농장 간 횡단면)는 다른 종류의 증거다.",
            "**축이 다르다** — 올리기(격차)·안 떨어지기(기댓값)·"
            "손실 상한(계절)을 한 열로 더하면 안 된다.",
        ]}


def season_sheet(d: dict) -> tuple:
    """여름 손실 분포 — `season.compute()` 출력을 편다."""
    loss, sh, ref = d["loss"], d["loss_shrunk"], d["panel_won_ref"]
    keys = [("p10", "하위10%"), ("p25", "하위25%"), ("median", "중앙"),
            ("p75", "상위25%"), ("p90", "상위10%")]
    rows = [{
        "분위": ko, "관측 손실%p": loss[k],
        "표본오차 제거 후%p": sh.get(k),
        "패널 실측 원/년": ref.get(k),
    } for k, ko in keys]
    return _g("실측", rows), {
        "title": f"여름 손실 분포 — 국내 {d['n_farms']}농장 · "
                 f"{d['n_sows']:,}두 규모 환산",
        "grade": "실측", "module": "farm_monthly_panel (server/routers/season)",
        "source": f"교배월 기준 · 전체 {d['overall']['summer_minus_winter']}%p",
        "caveats": list(d["caveats"]) + [
            "원/년 열은 농장마다 자기 PSY·자기 겨울로 낸 금액의 분위수다 — "
            "손실 분위와 같은 농장이 아니다(곱의 중앙값 ≠ 중앙값의 곱).",
        ]}


def targets_sheet(d: dict) -> tuple:
    """오늘의 영상 겨냥 — `vision_contract.targets()` 출력을 편다."""
    rows = [{
        "헤드": r["kr"], "개체": r["animal_id"], "축사": r["stage"],
        "경과일": r["day"], "창": ("~".join(map(str, r["window"]))
                                  if r["window"] else "상시"),
        "우선순위": r["priority"], "근거": r["why"],
    } for h in d["heads"] for r in d["heads"][h]["rows"]]
    return _g("계산", rows), {
        "title": f"오늘의 영상 겨냥 — {d['on']} · 배치 {d['n_placed']}두 · "
                 f"대상 {d['n_targets']}건",
        "grade": "계산", "module": "vision_contract.targets",
        "source": "개체 이력 스냅숏 + 번식 달력", "on": str(d["on"]),
        "caveats": [
            "**판정이 아니라 겨냥이다.** 이 표는 '이 개체를 오늘 보라' 까지이고, "
            "분만 임박·발정 여부를 말하지 않는다 — 그건 모델의 몫이다.",
            "행동 분류 모델은 아직 없다. 지금 꽂혀 있는 것은 배선을 시험하는 "
            "스텁뿐이다.",
            "질병 헤드는 달력이 없어 전 개체가 대상이라 우선순위가 없다.",
        ]}


BUILDERS = {"capacity": capacity_sheet, "interval": interval_sheet,
            "diagnosis": diagnosis_sheet, "priority": priority_sheet,
            "season": season_sheet, "targets": targets_sheet}


def build(sheet: str, payload: dict, bare: bool = False) -> str:
    if sheet not in BUILDERS:
        raise ValueError(f"알 수 없는 표: {sheet} (가능: {', '.join(SHEETS)})")
    rows, meta = BUILDERS[sheet](payload)
    return to_csv(rows, meta, bare=bare)


def filename(sheet: str, on=None) -> str:
    """내려받을 때 붙는 이름. **농장 이름을 넣지 않는다** — 식별자다."""
    day = str(on or date.today())
    return f"yangdon_{sheet}_{day}.csv"


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="table_export")
    ap.add_argument("--sheet", default="capacity", choices=SHEETS)
    ap.add_argument("--sows", type=int, default=300)
    ap.add_argument("--out", help="파일로 저장 (없으면 화면)")
    ap.add_argument("--bare", action="store_true", help="머리말 없이(기계용)")
    a = ap.parse_args(argv)

    from competition.server.routers import capacity as capr
    from competition.server.routers import season as seasonr
    from competition.server.schemas import FarmSetup

    import batch_flow as bf
    import farm_gap as fg
    import psy_priority as pp

    stub = FarmSetup(interval_days=21, lactation_days=24,
                     pre_farrow_days=7, washout_days=7)
    setup = FarmSetup(
        n_sows=a.sows, interval_days=21, lactation_days=24,
        pre_farrow_days=7, washout_days=7,
        barns=bf.design_barns(a.sows, 21.0, lactation=24, pre_farrow=7,
                              washdown=7, weaned_per_litter=11.0,
                              extra_rooms=capr._extra_rooms(stub)),
        performance={"weaned": 11.0, "farrowing_rate": 85.0, "survival": 94.0})
    perf = {"weaned": 10.0, "npd": 62.0, "farrowing_rate": 74.0}

    if a.sheet in ("capacity",):
        payload = capr.compute(setup)
    elif a.sheet == "interval":
        payload = capr.interval_whatif(setup)
    elif a.sheet == "season":
        payload = seasonr.compute(a.sows)
    elif a.sheet in ("diagnosis", "priority"):
        payload = {"diagnosis": fg.diagnose(dict(perf), n_sows=a.sows),
                   "priority": pp.build(dict(perf), a.sows, None)}
    else:
        import tempfile

        import farm_registry as fr
        import synth_farm as sf
        import vision_contract as vc

        df = sf.generate(a.sows, 1.0, "2025-01-01", 0, sf.Params())
        fd, p = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            sf.to_herd_csv(df, p, "2025-01-01")
            recs, as_of = fr.herd_from_csv(p)
        finally:
            os.unlink(p)
        payload = vc.targets(recs, as_of)

    out = build(a.sheet, payload, bare=a.bare)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            f.write(out)
        print(f"저장: {a.out} ({len(out.splitlines())}줄)")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
