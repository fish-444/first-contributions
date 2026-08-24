"""등록 규모 검산 — **총사육수와 상시모돈은 다른 수다.**

공공데이터포털의 국가 등록 스키마(농림축산식품부 축사 사육시설내역정보,
데이터번호 15151091)는 축사를 이렇게 적는다:

    축종코드 · 축산시설코드 · 사육시설형태(BF_STLE)
    축산허가면적(STKRS_PRMISN_AR) · 축산무허가면적(STKRS_NRT_AR)
    사육수(BRD_CO) · 대표축종여부 · 축종분류코드

우리 등록표는 **운영 단위**(동·방 수·방당 자리·방당 면적)이고 국가 표는
**행정 단위**(동 전체 허가면적·사육수)다. 둘 다 있어야 검산이 된다 —
이 모듈이 그 대조를 한다.

## 섞으면 안 되는 두 수

- **상시모돈**(`n_sows`) — 번식 모돈만. 우리 배치·용량 계산이 쓰는 수다.
- **총사육수**(`n_head_total`) — 모돈·자돈·육성·비육·웅돈 전부. 국가
  스키마의 `BRD_CO` 가 이쪽이고, 법정 사육밀도도 이 수로 잰다.

한 칸에 몰아 받으면 300두 농장이 어떤 화면에선 300, 어떤 화면에선
3,000 이 된다. **그래서 칸을 나눠 받고, 서로 대조한다.**

## 무엇을 검산하나

1. 동별 사육수 합 ↔ 총사육수 (안 맞으면 어느 쪽이 낡았다)
2. 동별 사육수 ↔ 그 동의 자리 수(방×방당 자리)
3. 허가면적 + 무허가면적 ↔ 운영표의 면적(방×방당 면적)
4. **무허가면적 > 0 → 적법화 대상**이라고 말한다
5. 상시모돈 ≤ 총사육수 (넘으면 둘 중 하나가 틀렸다)
6. 밀도는 `growth_flow.density_check` 를 그대로 부른다 — 재구현하지 않는다

**판정하지 않는 것**: 무허가면적이 있다고 위법이라고 말하지 않는다.
적법화 대상일 수 있다는 표시까지다 — 유예·특례는 이 프로그램이 모른다.

    python competition/src/farm_scale.py        # 합성 시연 (등급 합성)
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import legal_density as ld                                     # noqa: E402

# 법정 면적의 정본은 `legal_density` 다 — 「축산법 시행령」[별표 1] 조문표.
# 예전에는 자돈~비육만 보고 "번식사는 기준이 없다" 고 적었는데 **틀렸다**:
# 조문에 웅돈 6.0 · 임신돈 1.4 · 분만돈 3.9 · 종부대기돈 1.4/2.6 ·
# 후보돈 2.3 이 있다. 그동안 번식사 과밀을 아예 못 잡고 있었다.
DENSITY_STAGE = dict(ld.BARN_TO_LAW)
# 국가 스키마 필드명 — 나중에 실데이터가 붙을 때 이 대응표만 보면 된다.
NATIONAL_FIELDS = {
    "permit_area_m2": "STKRS_PRMISN_AR (축산허가면적)",
    "nonpermit_area_m2": "STKRS_NRT_AR (축산무허가면적)",
    "head": "BRD_CO (사육수)",
    "housing": "BF_STLE (사육시설형태)",
}
AREA_TOL = 0.05      # 면적 대조 허용 오차 5% — 통로·벽 두께로 늘 조금 어긋난다


def _barn_checks(b: dict) -> list:
    """동 하나의 검산. 비운 칸은 검사하지 않는다 — 채우라고만 말한다."""
    out = []
    name = b.get("name") or "(이름 없음)"
    head = b.get("head")
    slots = int(b.get("rooms") or 0) * int(b.get("per") or 0)
    permit = b.get("permit_area_m2")
    nonpermit = b.get("nonpermit_area_m2")

    if head is not None and slots and head > slots:
        out.append({"수준": "위험", "동": name,
                    "내용": f"사육수 {head:,}두 > 자리 {slots:,}개 — "
                           "등록표의 방 수·방당 자리를 다시 볼 것"})
    if nonpermit:
        out.append({"수준": "주의", "동": name,
                    "내용": f"무허가면적 {nonpermit:g}㎡ — 적법화 대상일 수 "
                           "있다(유예·특례는 이 프로그램이 모른다)"})
    # 행정 면적 ↔ 운영 면적. 방당 면적을 비웠으면 대조할 게 없다
    op_area = (b.get("area_m2") or 0) * int(b.get("rooms") or 0)
    adm_area = (permit or 0) + (nonpermit or 0)
    if op_area and adm_area:
        gap = abs(op_area - adm_area) / max(op_area, adm_area)
        if gap > AREA_TOL:
            out.append({"수준": "주의", "동": name,
                        "내용": f"면적이 어긋난다 — 운영표 {op_area:g}㎡ "
                               f"(방 {b.get('rooms')}개×{b.get('area_m2'):g}) "
                               f"vs 행정 {adm_area:g}㎡"})
    return out


def barn_density(b: dict) -> dict | None:
    """동 하나의 법정 밀도 — **허가면적 기준**. 기준을 못 정하면 이유를 낸다.

    운영표의 방당 면적으로 재던 검사와 **다른 질문**이다: 저쪽은 "이 방이
    좁은가", 이쪽은 "허가받은 면적에 이만큼 넣어도 되는가"다. 둘 다 낼 수
    있으면 둘 다 낸다 — 하나로 합치면 어느 쪽이 걸린 건지 모른다.

    사육방식이 갈리는 용도(교배사=종부대기돈, 후보사=후보돈)는 방식을
    받아야 값이 정해진다. 방식을 모르면 **판정하지 않고 그 사실을 낸다.**
    """
    head, permit = b.get("head"), b.get("permit_area_m2")
    if not (head and permit):
        return None
    req = ld.for_barn(b.get("stage"), b.get("housing"))
    if req.get("value") is None:
        return {"동": b.get("name"), "stage": b.get("stage"),
                "기준": "허가면적", "regulated": False, "why": req.get("why"),
                "options": req.get("options")}
    n, area, need = int(head), float(permit), float(req["value"])
    per = area / max(1, n)
    return {"동": b.get("name"), "stage": b.get("stage"), "기준": "허가면적",
            "regulated": True, "law_stage": req["law_stage"],
            "housing": req.get("housing"),
            "n_pigs": n, "area_m2": area,
            "per_head_m2": round(per, 3), "required_m2": need,
            "ratio": round(per / need, 2), "overcrowded": per < need,
            "capacity": int(area // need),
            "excess": max(0, n - int(area // need)),
            "source": ld.SOURCE,
            **({"interpreted": req["interpreted"]} if req.get("interpreted")
               else {}),
            **({"note": req["note"]} if req.get("note") else {})}


def reconcile(setup: dict) -> dict:
    """등록 JSON → 규모 검산. `setup` 은 `FarmSetup` 과 같은 모양이다."""
    barns = list(setup.get("barns") or [])
    sows = setup.get("n_sows")
    total = setup.get("n_head_total")

    heads = [b.get("head") for b in barns if b.get("head") is not None]
    head_sum = sum(int(h) for h in heads) if heads else None
    slots_sum = sum(int(b.get("rooms") or 0) * int(b.get("per") or 0)
                    for b in barns)
    permit_sum = sum(float(b.get("permit_area_m2") or 0) for b in barns)
    nonpermit_sum = sum(float(b.get("nonpermit_area_m2") or 0) for b in barns)

    checks = []
    for b in barns:
        checks += _barn_checks(b)

    if head_sum is not None and total is not None and head_sum != total:
        checks.append({"수준": "주의", "동": None,
                       "내용": f"동별 사육수 합 {head_sum:,}두 ≠ 총사육수 "
                              f"{total:,}두 — 어느 쪽이 낡았는지 확인할 것"})
    if sows is not None and total is not None and sows > total:
        checks.append({"수준": "위험", "동": None,
                       "내용": f"상시모돈 {sows:,}두 > 총사육수 {total:,}두 — "
                              "총사육수는 모돈을 포함한 전 두수다"})
    if heads and len(heads) < len(barns):
        checks.append({"수준": "주의", "동": None,
                       "내용": f"사육수를 적은 동이 {len(heads)}/{len(barns)} "
                              "— 합계가 농장 전체가 아니다"})

    return {
        "n_sows": sows, "n_head_total": total,
        "head_by_barn_sum": head_sum, "slots_sum": slots_sum,
        "permit_area_m2": round(permit_sum, 1) or None,
        "nonpermit_area_m2": round(nonpermit_sum, 1) or None,
        "density": [d for d in (barn_density(b) for b in barns) if d],
        "checks": checks,
        "grade": "계산",
        "national_fields": NATIONAL_FIELDS,
        "notes": [
            "**상시모돈과 총사육수는 다른 수다.** 상시모돈은 번식 모돈만이고 "
            "배치·용량 계산이 쓰는 수이며, 총사육수는 자돈·육성·비육·웅돈까지 "
            "포함한다. 국가 스키마의 BRD_CO 는 총사육수 쪽이다.",
            "허가면적 기준 밀도와 방당 면적 기준 밀도는 **다른 질문**이라 "
            "따로 낸다 — 합치면 어느 쪽이 걸렸는지 모른다.",
            "무허가면적은 **적법화 대상일 수 있다**는 표시까지다. 위법이라고 "
            "판정하지 않는다 — 유예·특례는 이 프로그램이 모른다.",
            f"법정 면적의 정본은 {ld.SOURCE} 이고 `legal_density` 가 "
            "들고 있다. 사육방식이 갈리는 용도(종부대기돈·후보돈)는 방식을 "
            "받아야 값이 정해지고, 조문에 없는 방식은 지어내지 않는다.",
            "군사 전환의 신규·기존 농가 적용 시기는 별표에 없어 확인하지 "
            "못했다 — 시기를 단정해 말하지 않는다.",
        ],
    }


def _demo() -> dict:
    """합성 시연 — 검산 넷이 각각 걸리는 구성."""
    return {
        "name": "예시 농장", "n_sows": 300, "n_head_total": 3200,
        "barns": [
            {"name": "1동", "stage": "교배사", "rooms": 1, "per": 60,
             "housing": "stall", "area_m2": 90.0,
             "head": 60, "permit_area_m2": 90.0, "nonpermit_area_m2": 0.0},
            {"name": "2동", "stage": "임신사", "rooms": 2, "per": 90,
             "housing": "group", "area_m2": 130.0,
             "head": 170, "permit_area_m2": 200.0, "nonpermit_area_m2": 60.0},
            # 교배사는 사육방식이 값을 가른다 — 스톨 1.4 vs 군사 2.6
            {"name": "6동", "stage": "교배사", "rooms": 1, "per": 60,
             "housing": "group", "area_m2": 100.0,
             "head": 55, "permit_area_m2": 100.0, "nonpermit_area_m2": 0.0},
            # 사육수가 자리보다 많다 — 위험
            {"name": "5동", "stage": "자돈사", "rooms": 6, "per": 66,
             "housing": "group", "area_m2": 24.0,
             "head": 430, "permit_area_m2": 120.0, "nonpermit_area_m2": 0.0},
        ],
    }


def main(argv=None) -> int:
    import json
    setup = _demo() if not argv else json.load(open(argv[0], encoding="utf-8"))
    r = reconcile(setup)
    print("=" * 72)
    print("  등록 규모 검산 — 합성 시연 (**등급 합성** — 실농장 아님)"
          if not argv else "  등록 규모 검산")
    print("=" * 72)
    print(f"  상시모돈 {r['n_sows']:,}두 · 총사육수 {r['n_head_total']:,}두 "
          f"· 동별 합 {r['head_by_barn_sum']:,}두 · 자리 {r['slots_sum']:,}개")
    print(f"  허가면적 {r['permit_area_m2']}㎡ · 무허가면적 "
          f"{r['nonpermit_area_m2']}㎡")
    for d in r["density"]:
        mark = "⚠ 과밀" if d["overcrowded"] else "적정"
        print(f"  밀도({d['기준']}) {d['동']}/{d['stage']} "
              f"두당 {d['per_head_m2']}㎡ / 기준 {d['required_m2']}㎡ — {mark}"
              + (f" · 초과 {d['excess']:,}두" if d["excess"] else ""))
    for c in r["checks"]:
        who = f"{c['동']} " if c["동"] else ""
        print(f"  🔔 [{c['수준']}] {who}{c['내용']}")
    for n in r["notes"]:
        print(f"  ⚠ {n.replace('**', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
