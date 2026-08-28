"""영상 모델과 앱 사이의 **계약** — 모델은 갈아끼우는 부품이다.

행동 분류 모델을 만드는 중이고, 아직 없다. 그래서 이 모듈은 모델을 담지
않고 **모델이 꽂힐 자리의 모양**만 정한다. 계약을 먼저 동결해 두면 배선
전체를 지금 만들고 시험할 수 있고, 완성된 모델은 `predict()` 하나만 맞추면
들어온다.

## 이 파일이 하는 일 둘

  1. `BehaviorObs` · `BehaviorModel` — 모델이 내야 하는 것의 모양
  2. `targets()` — **오늘 어느 개체를 어느 헤드로 볼지**를 번식 달력이 정한다

둘째가 요점이다. 모델이 전 개체를 24시간 보는 구조가 아니라, **이미 아는
일정이 카메라를 겨냥한다.** 분만 예정일을 알면 분만징후를 그 며칠에만 찾으면
되고, 이유일을 알면 발정을 그 창에서만 찾으면 된다. 사전확률을 달력이 공짜로
주는 것이고, 오탐이 그만큼 줄어든다.

## 여기서 새 임계값을 만들지 않는다

겨냥 창은 전부 기존 모듈에서 온다:

    발정   이유 후 3일 ~ `estrus_early_warning.ANESTRUS_DAY`(21)
    재발   교배 후 `pregnancy_check.CHECKPOINTS[0]` = 18~24일
    분만   `farm_registry.stage_of()` 가 분만사로 판정한 개체 중 분만 전
    질병   달력 없음 — 상시. 유일하게 겨냥이 안 되는 헤드다

**분만 '임박' 을 여기서 판정하지 않는다.** 둥지짓기가 분만 몇 시간 전인지는
행동을 봐야 아는 것이고, 달력이 낼 수 있는 건 "이 개체가 분만사에 있고 아직
안 낳았다" 까지다. 임박은 모델의 몫이라 여기서 숫자를 지어내면 모델이 오기
전에 이미 틀린 답이 박힌다.

## 왜 확률 분포를 받는가

`probs` 는 argmax 한 라벨이 아니라 클래스별 확률이다. 이 프로젝트의 행동
10클래스 실측이 0.485 인데, 그 라벨을 확정값으로 넘기면 헤드가 절반쯤 틀린
입력 위에 서고 오류가 곱해진다. 확률로 넘기면 헤드가 불확실성을 안고 판단할
수 있다.

`activity_px` 는 모델과 **따로** 받는다. 활동 vs 휴식은 AUC 0.739 로 검증된
신호라, 모델이 실패해도 이건 살아 있어야 한다. 새 모델의 기준선이기도 하다.

    python competition/src/vision_contract.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol, runtime_checkable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import estrus_early_warning as ew                              # noqa: E402
import farm_registry as fr                                     # noqa: E402
import pregnancy_check as pc                                   # noqa: E402
import repro_calendar as rc                                    # noqa: E402

# 헤드 넷. **질병만 달력이 없다** — 나머지 셋은 일정이 시점을 정해 준다.
HEADS = ("estrus", "return", "farrowing", "disease")
HEAD_KR = {"estrus": "발정", "return": "재발", "farrowing": "분만징후",
           "disease": "질병"}

# 각 헤드가 행동 어휘에서 **필요로 하는 것**. 모델이 이걸 못 내면 그 헤드는
# 활동량만으로 돌거나 아예 못 돈다 — 등록 시점에 그 사실을 말해야 한다.
HEAD_NEEDS: dict[str, tuple[str, ...]] = {
    # 둥지짓기. `Parturition` 단독보다 이쪽이 현장 가치가 크다 — 낳은 뒤
    # 아는 것과 낳기 전 아는 것은 다른 일이다.
    "farrowing": ("Scrubbing", "Searching"),
    "disease": ("Coughing",),
    # 발정·재발은 활동량 원신호로도 돈다. 승가·기립이 있으면 보강된다.
    "estrus": (),
    "return": (),
}
HEAD_HELPS: dict[str, tuple[str, ...]] = {
    "estrus": ("Mounting", "Standing", "Walking"),
    "return": ("Mounting", "Standing", "Walking"),
    "farrowing": ("Lying", "Standing", "Parturition"),
    "disease": ("Eating", "Drinking", "Lying"),
}

# 어휘 밖 **독립 채널** — 모델이 분류하는 게 아니라 파이프라인이 따로 잰다.
# activity_px(추적기, AUC 0.739 실측)는 계약이 처음부터 갖고 있었고,
# resp(호흡수 — pig_behavior.respiration)가 합류로 들어왔다. 빈호흡은
# 행동 15종에 없어서 어휘로는 원리적으로 못 여는 질병 신호를, 분만 임박의
# 호흡 변화까지 생리 경로로 연다. 단 **합성 검증 11/11 뿐, 실제 돼지 영상
# 실증 0회**다 — 그 등급이 신고에 붙어 다닌다.
CHANNEL_OPENS: dict[str, tuple[str, ...]] = {
    "resp": ("disease", "farrowing"),
}
CHANNEL_GRADE: dict[str, str] = {
    "resp": "합성 — 합성 관통 11/11 · 실제 돼지 영상 실증 0회 (실증 대기)",
}

# 발정 겨냥 시작 — 이유 후 며칠부터 본다. 실측 WEI 중앙(5~7일)보다 앞서
# 시작해야 이른 발정을 놓치지 않는다.
ESTRUS_WATCH_FROM = 3
# 재발 창은 임신진단 3주 관문 그대로다. CCTV 를 붙이면 이 관문의 민감도가
# 0.70 → 0.92 로 오른다는 것이 이 프로젝트의 기존 측정이고, **겨냥의 값이
# 정확히 거기 있다.**
RETURN_FROM, RETURN_TO = pc.CHECKPOINTS[0][1], pc.CHECKPOINTS[0][2]


# --------------------------------------------------------------------------
# 1) 모델이 내야 하는 것
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BehaviorObs:
    """한 개체(또는 한 방)의 한 시간창 관측.

    `animal_id` 가 None 이면 **개체를 특정하지 못한 것**이다. 군사에서는
    트랙 ID 를 며칠씩 끌고 갈 수 없다(이 프로젝트 실측 ID 일관성 0.77).
    거짓 확신 대신 방 단위로 내려가고, 스톨은 자리가 곧 개체라 채워진다.
    """
    camera_id: str
    barn: str
    pen: str
    t0: str                       # ISO 로컬 시각
    t1: str
    track_id: int | None = None
    animal_id: str | None = None
    probs: dict[str, float] = field(default_factory=dict)
    activity_px: float = 0.0
    #: 호흡수(bpm). 모델 어휘가 아니라 별도 측정 채널이다 — 없으면 None.
    #: 판정하지 않는다: 정상 범위는 품종·일령·기온에 달려 고정 임계가 없다.
    resp_bpm: float | None = None
    model: str = ""

    def top(self) -> tuple[str, float] | None:
        """가장 높은 클래스. **집계에는 쓰지 말 것** — 분포가 정본이다."""
        return max(self.probs.items(), key=lambda x: x[1]) if self.probs else None


@runtime_checkable
class BehaviorModel(Protocol):
    """행동 분류 모델. 구현체는 셋을 예상한다.

        ReplayModel     라벨 캐시 재생 — 배선 시험용(모델 없이 지금 돈다)
        손피처 RF        기준선. 이 프로젝트 행동 10클래스 실측 0.485
        학습 중인 DL     완성되면 여기 꽂힌다

    **기준선이 명시적으로 존재하는 게 요점이다.** 이 프로젝트는 행동 시퀀스
    에서 1D-CNN(0.427)이 손피처 RF(0.485)에 진 것을 이미 실측했다. 새 모델은
    같은 검증(개체 분리)으로 그 값을 넘는 것이 등록 조건이고, 못 넘으면
    기준선이 그대로 돈다 — 시스템은 어느 쪽이든 멈추지 않는다.
    """
    version: str
    classes: tuple[str, ...]

    def predict(self, frames, tracks) -> list: ...


def head_support(model: "BehaviorModel", channels: tuple = ()) -> dict:
    """이 모델로 어느 헤드가 도는가. **등록 시점에 말해야 한다.**

    나중에 조용히 빈 결과를 내면 "경보가 안 뜬다" 와 "볼 수가 없다" 를
    구분할 수 없다.

    `channels` 는 어휘 밖 독립 채널("resp" 등)이다. 어휘 판정(`runs`)은
    그대로 두고 — 어휘로 막힌 것은 막혔다고 말해야 다음 학습 목록이
    남는다 — 채널 경로는 `channel_runs`/`channel_why` 로 **따로** 신고한다.
    등급도 같이 간다: 호흡은 합성 검증뿐이라 실증 전이다.
    """
    have = set(getattr(model, "classes", ()) or ())
    out = {}
    for h in HEADS:
        need = set(HEAD_NEEDS[h])
        miss = sorted(need - have)
        helps = sorted(set(HEAD_HELPS[h]) & have)
        if miss:
            why = f"어휘 {'·'.join(miss)} 가 없다"
        elif need:
            why = f"{'·'.join(sorted(need))} 로 돈다"
        elif helps:
            why = f"활동량 원신호 + {'·'.join(helps)} 로 보강"
        else:
            why = "활동량 원신호만으로 돈다"
        ch = [c for c in channels if h in CHANNEL_OPENS.get(c, ())]
        out[h] = {
            "head": h, "kr": HEAD_KR[h],
            "runs": not miss,
            "channel_runs": bool(ch),
            "channel_why": (" · ".join(f"{c} 채널({CHANNEL_GRADE[c]})"
                                       for c in ch) if ch else None),
            "missing": miss,
            "boosted_by": helps,
            "why": why,
        }
    return out


# --------------------------------------------------------------------------
# 2) 겨냥 — 오늘 누구를 어느 헤드로 볼 것인가
# --------------------------------------------------------------------------
def targets(records, on, disease_all: bool = True) -> dict:
    """개체 이력 + 오늘 → **헤드별 관찰 대상**.

    `farm_registry.stage_of()` 가 이미 "그날 어느 축사에 있는가" 를 판정하고
    유산·도태로 끊긴 기록을 걸러 낸다. 여기서 그걸 다시 짜지 않고 부른다.
    """
    t0 = rc._d(on)
    rows: list[dict] = []
    seen = 0
    for r in records:
        st, code, why = fr.stage_of(r, t0)
        if st is None:
            continue
        seen += 1
        aid = str(r.get("id") or "")
        for hit in _match(r, t0, st, code):
            rows.append({"animal_id": aid, "stage": st, **hit})
        if disease_all:
            # 질병만 달력이 없다. 전 개체 상시라 우선순위를 못 매긴다 —
            # 그 사실이 이 헤드의 성질이므로 숨기지 않는다.
            rows.append({"animal_id": aid, "stage": st, "head": "disease",
                         "kr": HEAD_KR["disease"], "day": None,
                         "window": None, "priority": 0.0,
                         "why": "달력 없음 — 상시 관찰"})

    by_head: dict = {}
    for x in rows:
        by_head.setdefault(x["head"], []).append(x)
    for h in by_head:
        by_head[h].sort(key=lambda x: (-x["priority"], x["animal_id"]))
    return {
        "on": t0, "n_placed": seen, "n_targets": len(rows),
        "heads": {h: {"kr": HEAD_KR[h], "n": len(by_head.get(h, [])),
                      "rows": by_head.get(h, [])} for h in HEADS},
        "note": ("겨냥은 번식 달력이 하고 판정은 모델이 한다. 이 표는 "
                 "**어느 개체를 볼지**까지이고, 분만 임박·발정 여부를 "
                 "여기서 말하지 않는다."),
    }


def _match(rec: dict, t0: date, stage: str, code: str) -> list:
    """개체 하나가 오늘 걸리는 헤드들. 창은 전부 기존 모듈에서 온다."""
    out = []

    def d(k):
        v = rec.get(k)
        if v is None or (isinstance(v, float) and v != v) or v == "":
            return None
        return rc._d(v)

    wea, svc, far = d("weaning_date"), d("service_date"), d("farrow_date")

    # 발정 — 이유했고 아직 교배 안 된 개체(공태)
    if code == "open" and wea:
        n = (t0 - wea).days
        if ESTRUS_WATCH_FROM <= n <= ew.ANESTRUS_DAY:
            # 우선순위는 실측 WEI 에 가까울수록 높다. 지나면 지연·무발정
            # 쪽이라 오히려 더 봐야 하므로 다시 오른다.
            wei = rc.expected_wei("sow")
            near = 1.0 - min(1.0, abs(n - wei) / max(1.0, ew.ANESTRUS_DAY - wei))
            late = 1.0 if n >= ew.DELAY_ALERT_DAY else 0.0
            out.append({"head": "estrus", "kr": HEAD_KR["estrus"], "day": n,
                        "window": [ESTRUS_WATCH_FROM, ew.ANESTRUS_DAY],
                        "priority": round(max(near, late * 0.9), 3),
                        "why": (f"이유 {n}일째 · 예상 WEI {wei:g}일"
                                + (" · 지연 경보 구간" if late else ""))})

    # 재발 — 임신진단 3주 관문. CCTV 가 민감도를 0.70 → 0.92 로 올리는 자리
    if svc and code in ("served", "pregnant", "returned"):
        n = (t0 - svc).days
        if RETURN_FROM <= n <= RETURN_TO:
            out.append({"head": "return", "kr": HEAD_KR["return"], "day": n,
                        "window": [RETURN_FROM, RETURN_TO], "priority": 1.0,
                        "why": (f"교배 {n}일째 · 3주 관문(재발돈의 "
                                f"{pc.CHECKPOINTS[0][4]:.0%} 를 여기서 잡는다)")})

    # 분만징후 — 분만사에 있고 아직 안 낳은 개체. **임박은 판정하지 않는다**
    if stage == "분만사" and code == "pre_farrow" and svc:
        exp = far or (svc + timedelta(days=fr.GEST))
        left = (exp - t0).days
        out.append({"head": "farrowing", "kr": HEAD_KR["farrowing"],
                    "day": left, "window": [-fr.PRE_FARROW, 0],
                    "priority": round(1.0 - min(1.0, max(0, left) / fr.PRE_FARROW), 3),
                    "why": (f"분만 예정 {left}일 " + ("전" if left >= 0 else "지남")
                            + " · 임박 여부는 모델이 판정한다")})
    return out


# --------------------------------------------------------------------------
# 3) 모델 없이 배선을 시험하는 스텁
# --------------------------------------------------------------------------
class ReplayModel:
    """**모델이 아니다.** 미리 정한 분포를 그대로 되돌려 준다.

    끝단(겨냥 → 관측 → 저장)이 실제로 관통하는지를 모델 없이 확인하려고
    둔다. 여기서 그럴듯한 확률을 지어내면 화면에 성능처럼 보이는 수가
    찍히므로, `version` 에 replay 라고 박아 응답마다 따라다니게 한다.
    """

    def __init__(self, classes: tuple[str, ...], fixture=None,
                 version: str = "replay-v0"):
        self.classes = tuple(classes)
        self.version = version
        self._fx = list(fixture or [])

    def predict(self, frames, tracks) -> list:
        out = []
        for i, (cam, barn, pen, t0, t1, probs, act) in enumerate(self._fx):
            bad = sorted(set(probs) - set(self.classes))
            if bad:
                raise ValueError(f"어휘에 없는 클래스: {bad}")
            out.append(BehaviorObs(camera_id=cam, barn=barn, pen=pen,
                                   t0=t0, t1=t1, track_id=i,
                                   probs=dict(probs), activity_px=float(act),
                                   model=self.version))
        return out


def main() -> int:
    import synth_farm as sf

    df = sf.generate(120, 1.0, "2025-01-01", 0, sf.Params())
    import tempfile
    fd, p = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        sf.to_herd_csv(df, p, "2025-01-01")
        recs, as_of = fr.herd_from_csv(p)
    finally:
        os.unlink(p)

    t = targets(recs, as_of)
    print("=" * 72)
    print(f"  영상 겨냥 — {t['on']} · 배치된 개체 {t['n_placed']}두")
    print("=" * 72)
    print("  겨냥은 번식 달력이 하고, 판정은 모델이 한다.\n")
    for h in HEADS:
        blk = t["heads"][h]
        cal = "달력 없음(상시)" if h == "disease" else "달력이 겨냥"
        print(f"  {blk['kr']:<6} {blk['n']:>4}두   {cal}")
        for r in blk["rows"][:3]:
            print(f"      {r['animal_id']:<8} {r['stage']:<5} {r['why']}")
    print()

    m = ReplayModel(("Lying", "Standing", "Scrubbing", "Searching"))
    print(f"  모델 {m.version} · 어휘 {len(m.classes)}종 — 어느 헤드가 도는가")
    for h, s in head_support(m).items():
        mark = "✅" if s["runs"] else "❌"
        print(f"    {mark} {s['kr']:<6} {s['why']}")
    print("\n  ⚠️ replay 는 모델이 아니다 — 배선을 시험하는 스텁이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
