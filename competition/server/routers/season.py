"""여름 손실 — 발견 ③′ 를 **우리 규모로**.

## 이 축이 다른 축과 다른 점

전체 평균 하나로 말하면 안 된다. 67농장 실측에서 여름 교배분 분만율은
겨울보다 중앙 −2.97%p 떨어지는데, **농장마다 갈린다**(하위10% −4.4 ~
상위10% +13.0%p). 표본 오차를 걷어내도 관측 분산의 **41% 는 진짜 농장
차이**라서, 공통 처방이 아니라 **선별 처방**이다.

그리고 **연간 성적으로는 못 맞힌다**(PSY 와 ρ −0.149). 잘하는 농장도 여름은
피하지 못한다 — 그래서 "우리는 PSY 가 좋으니 괜찮겠지" 가 성립하지 않는다.

무너지는 경로는 사양이 아니라 **착상**이다. 여름에 이유두수·재귀율은 거의
그대로인데 임신사고 구성이 1차 재발 쪽으로 기운다. 겨냥할 시점이 그래서
교배 후 착상기다.

산식은 `farm_monthly_panel.to_money` 와 같다. 여기서 새로 만들면 등록
화면·CLI·API 가 같은 농장에 다른 금액을 말한다.
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException, Query

import barn_environment as be                                  # noqa: E402
import farm_monthly_panel as mp                                # noqa: E402

router = APIRouter(prefix="/api/season", tags=["season"])

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(os.path.dirname(HERE), "data", "farm_monthly_panel.json")


def _panel() -> dict:
    if not os.path.exists(PANEL):
        raise HTTPException(503, "farm_monthly_panel.json 이 없다 — "
                                 "python competition/src/farm_monthly_panel.py")
    return json.load(open(PANEL, encoding="utf-8"))


def compute(sows: int = 300, psy: float | None = None,
            summer: float | None = None, winter: float | None = None) -> dict:
    """라우터와 **정적 뷰 빌더가 같이 쓴다.**

    화면을 구울 때 여기 산식을 옮겨 적으면 서버로 본 금액과 파일로 본 금액이
    갈린다. 심사장에서 그 둘이 나란히 열릴 수 있으므로 한 함수로 둔다.
    """
    r = _panel()
    money, loss = r["money"], r["loss"]
    per_sow = float(money["per_sow_won"])
    share = mp.SEASON_SHARE
    scale = sows / max(1, int(money["ref_sows"]))

    # PSY 는 안 주면 실측 중앙을 쓰되 **가정이라고 밝힌다.** 조용히 넣으면
    # 그게 내 농장 값인 줄 안다.
    psy_used = float(psy) if psy is not None else 24.1
    psy_src = "입력값" if psy is not None else "466농장 중앙값 · 가정"

    out: dict = {
        "n_sows": sows, "n_farms": r["n_farms"],
        "psy_used": psy_used, "psy_source": psy_src,
        "per_sow_won": per_sow, "season_share": share,
        "overall": r["overall"],
        "loss": loss, "loss_shrunk": r["loss_shrunk"], "spread": r["spread"],
        "join": r["join"],
        "accidents": r["pathways"]["accidents"],
        "implantation_window": list(be.IMPLANTATION_WINDOW),
        "duplicates": r["duplicates"],
    }

    if summer is not None and winter is not None:
        gap = winter - summer
        d_psy = psy_used * share * (gap / max(1e-9, winter))
        out["mine"] = {
            "summer": summer, "winter": winter, "loss_pp": round(gap, 2),
            "d_psy": round(d_psy, 3),
            "won_year": round(d_psy * per_sow * sows),
            "worse_than_median": gap > loss["median"],
            "percentile_hint": _where(gap, loss),
        }
        out["given"] = True
    else:
        # 두 칸을 비웠으면 **우리 농장 값이 아니다.** 국내 분포를 우리 규모로
        # 환산한 범위일 뿐이고, 그걸 라벨로 밝힌다.
        #
        # ⚠️ 패널의 733만원(300두 중앙)과 **같은 값이 아니다.** 저건 농장마다
        # 자기 PSY·자기 겨울로 낸 금액들의 중앙값이고, 여기는 중앙 손실 하나를
        # 대표 PSY 에 적용한 시나리오다. 곱의 중앙값과 중앙값의 곱은 다르다.
        w = float(r["overall"]["winter"])
        def won(pp: float) -> int:
            return round(psy_used * share * (pp / w) * per_sow * sows)
        out["scenario"] = {
            "winter_used": w,
            "median": {"loss_pp": loss["median"], "won_year": won(loss["median"])},
            "p90": {"loss_pp": loss["p90"], "won_year": won(loss["p90"])},
            "p10": {"loss_pp": loss["p10"], "won_year": won(loss["p10"])},
            "note": ("두 칸을 비웠으므로 우리 농장 값이 아니다 — 국내 분포를 "
                     "우리 규모로 환산한 범위다. 패널의 농장별 금액 중앙값과는 "
                     "다른 수(곱의 중앙값 ≠ 중앙값의 곱)."),
        }
        out["given"] = False

    # 패널 실측 기준 금액(농장마다 자기 PSY·자기 겨울로 낸 것의 분위수)
    out["panel_won_ref"] = {k: round(v * scale)
                            for k, v in money["won_ref"].items()}
    out["caveats"] = [
        "겨울을 기준으로 잡았으므로 **손실 상한**이다 — 냉방 장비값을 뺀 "
        "순이익이 아니다.",
        f"연간 성적으로 계절 취약도를 맞힐 수 없다 (PSY 와 ρ "
        f"{r['join']['PSY']['rho']}). 잘하는 농장도 여름은 피하지 못한다.",
        f"관측 분산의 {r['spread']['true_share']:.0%} 만 진짜 농장 차이다. "
        f"나머지는 표본 오차이므로 개별 농장 값은 축소해서 읽어야 한다.",
        f"원자료 {r['duplicates']['rows_raw']:,}행 중 유일한 건 "
        f"{r['duplicates']['rows_dedup']:,}행뿐이었다(중복 "
        f"{r['duplicates']['dup_share']:.1%}). 지우고 낸 값이다.",
    ]
    return out


@router.get("", summary="여름 손실 — 분포 · 우리 규모 환산 · 겨냥 시점")
def season(sows: int = Query(300, ge=1, le=20000),
           psy: float | None = Query(None, ge=5, le=45),
           summer: float | None = Query(None, ge=20, le=100,
                                        description="여름(7·8·9월) 교배분 분만율 %"),
           winter: float | None = Query(None, ge=20, le=100,
                                        description="겨울(1·2·3월) 교배분 분만율 %")) -> dict:
    return compute(sows, psy, summer, winter)


def _where(gap: float, loss: dict) -> str:
    if gap <= loss["p10"]:
        return "하위 10% — 여름에 오히려 강하다"
    if gap <= loss["p25"]:
        return "하위 25% — 무던한 편"
    if gap <= loss["median"]:
        return "중앙 아래"
    if gap <= loss["p75"]:
        return "중앙 위 — 취약한 편"
    return "상위 25% — 취약. 착상기 냉방의 값이 가장 크다"
