"""법정 사육밀도 — 「축산법 시행령」[별표 1] 돼지 표를 **조문 그대로.**

    「축산법 시행령」[별표 1] 축산업의 허가 및 등록 요건
    (제14조제2항 및 제14조의2제2항 관련) · 개정 2025. 3. 25.
    3) 돼지 > 가) 성장단계별 마리당 가축사육시설 면적

이 파일이 **법정 면적의 유일한 정본**이다. 다른 모듈이 숫자를 따로 적으면
개정 때 한쪽만 낡는다 — `growth_flow.STAGES` 의 면적도 여기서 온다.

## 조문표 (단위: ㎡/마리)

    웅돈 6.0 · 임신돈 1.4 · 분만돈 3.9
    종부대기돈 1.4(스톨) / 2.6[군사(群飼)]
    후보돈 2.3(군사)
    새끼돼지 초기 0.2 · 후기 0.3 · 육성돈 0.45 · 비육돈 0.8

**표를 한 칸 밀려 읽는 오류가 흔하다.** 2차 정리본 중에 "임신돈 1.4/2.6,
후보돈 0.2" 로 적은 것이 있는데 틀렸다 — 스톨/군사가 갈리는 것은
**종부대기돈**이고, 0.2 는 새끼돼지 초기(20kg 미만) 칸이다. 국가법령정보
센터 원문 PDF 에서 셀 x좌표로 대조해 확정했다(종부대기돈 열 251–312,
후보돈 열 312–369).

## 조문이 정한 산정방법 — 숫자만큼 중요하다

1. **가)성장단계별과 나)경영형태별 중 선택 적용.** 둘을 섞지 않는다.
2. **새끼돼지는 젖 뗀 마릿수 기준** — 번식돈과 함께 있는 젖먹이는 세지
   않는다. 그래서 분만사의 포유자돈에는 면적 기준을 매기지 않는다.
3. 성장단계는 체중으로 가른다: 새끼돼지 초기 20kg 미만 · 후기 20~30kg ·
   육성돈 30~60kg · 비육돈 60kg 이상.

## 이 파일이 하지 않는 것

- **군사 전환 시기를 말하지 않는다.** 별표에는 부칙이 없어 신규·기존
  농가 적용 시점을 확인하지 못했다. 화면에 시기를 단정해 쓰면 안 된다.
- **스톨 값이 없는 항목에 값을 지어내지 않는다.** 후보돈은 조문에
  `2.3(군사)` 만 있고 스톨 값이 없다 — 없는 것은 없다고 답한다.
- 별표 1 은 **허가·등록 요건**이다. 이 프로그램의 판정은 "요건 대비
  면적이 모자란다"까지이고, 행정처분 여부를 말하지 않는다.

    python competition/src/legal_density.py      # 표와 출처 출력
"""
from __future__ import annotations

SOURCE = ("「축산법 시행령」[별표 1] 축산업의 허가 및 등록 요건"
          "(제14조제2항 및 제14조의2제2항 관련), 개정 2025. 3. 25. "
          "— 국가법령정보센터")
AMENDED = "2025-03-25"

# 조문의 성장단계 → 마리당 면적(㎡). 사육방식이 갈리는 항목은 dict 로 둔다.
# 값이 dict 인데 그 방식이 없으면 **판정하지 않는다** — 지어내지 않는다.
TABLE: dict = {
    "웅돈": 6.0,
    "임신돈": 1.4,
    "분만돈": 3.9,
    "종부대기돈": {"stall": 1.4, "group": 2.6},
    "후보돈": {"group": 2.3},
    "새끼돼지_초기": 0.2,        # 20kg 미만 (젖먹이 — 아래 비고 2 참고)
    "새끼돼지_후기": 0.3,        # 20~30kg (젖뗀 돼지)
    "육성돈": 0.45,              # 30~60kg
    "비육돈": 0.8,               # 60kg 이상
}
# 나) 경영 형태별 마리당 면적 — 가)와 **선택 적용**이라 섞지 않는다.
BY_BUSINESS = {"일관경영": 0.79, "번식경영-1": 2.42, "번식경영-2": 0.90,
               "비육경영-1": 0.62, "비육경영-2": 0.73}

# 우리 등록 어휘(`farm_registry.BARN_STAGES`) → 조문 성장단계.
# **이 대응은 조문 문구가 아니라 우리 해석이다.** 특히 교배사는 조문의
# 종부대기돈("임신·분만·이유를 거쳐 교배를 기다리는 암퇘지") 정의와 맞지만,
# 후보돈을 같은 방에 두는 농장이면 실제로는 섞여 있다.
BARN_TO_LAW = {
    "교배사": "종부대기돈",
    "임신사": "임신돈",
    "분만사": "분만돈",
    "후보사": "후보돈",
    "자돈사": "새끼돼지_후기",
    "육성사": "육성돈",
    "비육사": "비육돈",
}
INTERPRETED = {"교배사"}      # 조문에 같은 낱말이 없는 대응 — 화면이 밝힌다

LABEL = {"stall": "스톨", "group": "군사", "crate": "분만틀", "pen": "일반 돈방"}


def required_m2(law_stage: str, housing: str | None = None) -> dict:
    """조문 성장단계(+사육방식) → 마리당 필요 면적.

    돌려주는 것은 값 하나가 아니라 **왜 그 값인지**까지다. 값을 못 정하면
    `value=None` 과 이유를 준다 — 0 이나 기본값으로 때우지 않는다.
    """
    if law_stage not in TABLE:
        return {"value": None, "law_stage": law_stage,
                "why": f"조문에 '{law_stage}' 항목이 없다"}
    v = TABLE[law_stage]
    if not isinstance(v, dict):
        return {"value": float(v), "law_stage": law_stage, "housing": None,
                "source": SOURCE}
    # 사육방식이 갈리는 항목
    if housing is None:
        return {"value": None, "law_stage": law_stage, "options": dict(v),
                "why": (f"'{law_stage}' 은 사육방식마다 기준이 다르다"
                        f"({' · '.join(f'{LABEL.get(k, k)} {x}' for k, x in v.items())})"
                        " — 방식을 알아야 정한다")}
    if housing not in v:
        return {"value": None, "law_stage": law_stage, "housing": housing,
                "options": dict(v),
                "why": (f"조문에 '{law_stage}' 의 {LABEL.get(housing, housing)} "
                        "기준이 없다 — 없는 값을 지어내지 않는다")}
    return {"value": float(v[housing]), "law_stage": law_stage,
            "housing": housing, "source": SOURCE}


def for_barn(stage: str, housing: str | None = None) -> dict:
    """우리 등록 어휘(축사 용도) → 필요 면적. 대응이 해석이면 밝힌다."""
    law = BARN_TO_LAW.get(stage)
    if law is None:
        return {"value": None, "barn_stage": stage,
                "why": f"'{stage}' 에 대응하는 조문 성장단계를 정하지 않았다"}
    out = required_m2(law, housing)
    out["barn_stage"] = stage
    if stage in INTERPRETED:
        out["interpreted"] = (f"'{stage}' → 조문 '{law}' 대응은 우리 해석이다"
                              " — 조문에 같은 낱말이 없다")
    if stage == "분만사":
        out["note"] = ("조문 비고: 새끼돼지는 **젖 뗀 마릿수 기준**이라 "
                       "분만사의 포유자돈은 세지 않는다. 3.9 는 모돈 몫이다.")
    return out


def main() -> int:
    print("=" * 72)
    print("  법정 사육밀도 (돼지) — 조문 그대로")
    print("=" * 72)
    print(f"  {SOURCE}\n")
    for k, v in TABLE.items():
        if isinstance(v, dict):
            s = " / ".join(f"{x}({LABEL.get(h, h)})" for h, x in v.items())
        else:
            s = f"{v}"
        print(f"  {k:<14} {s} ㎡")
    print("\n  나) 경영 형태별 (가)와 **선택 적용**):")
    print("  " + " · ".join(f"{k} {v}" for k, v in BY_BUSINESS.items()))
    print("\n  우리 축사 용도 → 조문 대응:")
    for st in BARN_TO_LAW:
        for hs in (None, "stall", "group"):
            r = for_barn(st, hs)
            if r.get("value") is not None:
                tag = f"({LABEL[hs]})" if hs else ""
                mark = " ※해석" if r.get("interpreted") else ""
                print(f"    {st:<8}{tag:<6} {r['value']} ㎡{mark}")
        r0 = for_barn(st)
        if r0.get("value") is None and "options" in r0:
            print(f"    {st:<8}       — {r0['why']}")
    print("\n  ⚠ 군사 전환 적용 시기는 별표에 없다 — 부칙을 따로 확인할 것.")
    print("  ⚠ 별표 1 은 허가·등록 요건이다. 행정처분 여부를 말하지 않는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
