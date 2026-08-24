"""행동 기준선 층 — 구성비의 **자기 기준선 편차**로 발정·분만·질병을 본다.

행동 분할 모델(`vision_pig_behavior`)은 사진 한 장의 스냅숏만 읽는다. 그런데
발정·분만·질병은 전부 **변화**에서 드러난다 — 불안정해지고, 먹기를 멈추고,
눕기만 한다. 이 층이 그 사이를 잇는다:

    fold() 구성비  →  자기 기준선(방/개체별)  →  헤드별 이탈 점수  →  경보

## 절대값을 쓰지 않는다

프레임 라벨은 AP 0.24~0.63 으로 잡음이 크고, 방마다 카메라 각도·돼지 수가
달라 "Eating 30% 는 정상인가" 에 공통 답이 없다. 그래서 **자기 기준선과의
편차**만 쓴다 — 71763 사양관리에서 직장온도 절대값(중앙 37.8℃, 정상 밖)을
버리고 평균 0 기준으로 갔던 것과 같은 판단이다. 산포도 표준편차가 아니라
**IQR 기반**으로 잰다 — 구성비 분포는 정규가 아니다.

## 문턱을 발명하지 않는다

"z 2.0 이상은 이상" 같은 상수를 박지 않는다. **과거 자기 이력에서 경보율이
목표 대역(기본 0.5~5%)에 들어오는 컷을 역산**한다 — 같은 프로젝트의 이상치
기준표가 쓴 방법 그대로다(3σ 일괄을 물렸더니 한 필드는 경보 0건, 한 필드는
10.2% 였다). 이력이 모자라면 컷을 만들지 않고 "기준선 미형성" 이라고
말한다 — 조용히 기본값으로 채우면 그게 근거 있는 경보인 줄 안다.

## 가중치는 배운 것이 아니다

헤드 점수의 부호는 문헌 징후에서 온다(발정=불안정·식욕저하, 분만 임박=절식·
탐색, 질병=섭식저하·무기력). **크기는 등가중**이다 — 라벨 데이터로 배우기
전까지 1:1:1 이고, 그 사실을 응답의 `weights` 가 들고 다닌다. 여기서
그럴듯한 가중치를 지어내면 배점처럼 읽힌다.

## 헤드를 가르는 것은 점수가 아니라 달력이다

발정과 분만 임박은 신호가 겹친다 — 둘 다 Eating↓ · Walking↑ 이다. 점수만
보면 발정 창이 분만 헤드도 울린다. 가르는 것은 `vision_contract.targets()`
다: 이유 후 3~21일의 공태돈에는 발정 헤드만, 분만 예정 D−7~0 의 분만틀
개체에는 분만 헤드만 열린다. 그래서 `assess(heads=...)` 로 **달력이 연
헤드만 계산**하는 것이 정상 경로이고, 전 헤드 계산은 시연·감사용이다.

## 판정이 아니라 의심이다

발정 판정은 이 프로젝트가 AUC 0.465 로 "이 데이터로는 안 된다" 를 실측한
과제다. 이 층의 출력은 **"이 방을 오늘 먼저 보라"** 까지이고, 그 사실이
등급(`계산`)과 함께 응답에 붙는다.

    python competition/src/behavior_baseline.py     # 합성 시연 (등급 합성)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 헤드별 부호 — 문헌 징후의 방향만 가져온다. 크기는 등가중(배운 값 아님).
#   발정      불안정(Walking↑) · 안정 감소(Resting↓) · 식욕 저하(Eating↓)
#   분만 임박  절식(Eating↓) · 둥지 탐색(Searching↑) · 불안정(Walking↑)
#   질병      섭식 저하(Eating↓) · 무기력(Resting↑)
HEAD_SIGNS = {
    "estrus": {"Walking": +1.0, "Resting": -1.0, "Eating": -1.0},
    "farrowing": {"Eating": -1.0, "Searching": +1.0, "Walking": +1.0},
    "disease": {"Eating": -1.0, "Resting": +1.0},
}
# 질병은 한 창 튀는 것으로 울리지 않는다 — 섭식은 급이 시각에 따라 출렁이므로
# 연속 창을 요구한다. estrus_early_warning 의 SUSTAIN 과 같은 생각이다.
SUSTAIN = {"estrus": 1, "farrowing": 1, "disease": 2}

# 보조 신호 — **있으면 등가중으로 합류하고, 없어도 헤드를 닫지 않는다.**
# 어휘(HEAD_SIGNS)와 달리 이건 모델 밖 채널이라 이력에 없는 것이 흠이
# 아니다. activity_px 는 이 프로젝트에서 가장 단단한 신호(AUC 0.739
# 실측)인데 그동안 기준선 층까지 오지 않고 버려지고 있었다. resp_bpm 은
# 합성 검증뿐이다(실증 대기). 부호는 문헌: 발정·분만 임박은 불안정으로
# 활동↑, 질병은 무기력으로 활동↓ · 빈호흡↑.
HEAD_EXTRA = {
    "estrus": {"activity_px": +1.0},
    "farrowing": {"activity_px": +1.0},
    "disease": {"activity_px": -1.0, "resp_bpm": +1.0},
}
# 채널 키 — 구성비와 **부재의 뜻이 다르다.** 구성비에서 빠진 클래스는
# "그 행동이 0"이지만(검출 안 됨), 채널이 빠진 창은 "안 쟀다"다.
# 0 으로 채우면 추적기가 꺼진 창이 z≈-7 로 계산돼 허위 질병 경보가
# 난다 — 리뷰가 실행으로 확인한 결함이라 여기 선을 긋는다.
CHANNEL_KEYS = ("activity_px", "resp_bpm")
# 창의 **분모**. 구성비와 나란히 실려 오지만 클래스가 아니라서 기준선을
# 만들 때도 편차를 낼 때도 빠진다. 밑줄로 시작하는 키는 전부 메타로 본다.
COUNT_KEY = "_n_used"

MIN_WINDOWS = 12          # 기준선을 만들 최소 창 수 — 미달이면 만들지 않는다
RATE_BAND = (0.005, 0.05)  # 목표 경보율 대역. 컷은 여기서 역산한다
_Z_GRID = np.arange(0.5, 6.01, 0.1)


def _robust(x: np.ndarray) -> tuple:
    """중앙값과 IQR 기반 산포. 구성비 분포는 정규가 아니라 σ 를 안 쓴다."""
    med = float(np.median(x))
    q1, q3 = np.percentile(x, [25, 75])
    rsd = float(max((q3 - q1) / 1.349, 1e-6))
    return med, rsd


@dataclass
class Baseline:
    """한 키(방 또는 개체)의 자기 기준선.

    `key` 는 방("3동/2방") 또는 분만틀 개체 — **어느 쪽인지는 호출자가
    정한다.** 군사는 방까지고(개체 특정 안 함), 분만틀은 자리가 곧 개체다.
    """
    key: str
    classes: tuple
    center: dict = field(default_factory=dict)   # {class: 중앙값}
    spread: dict = field(default_factory=dict)   # {class: IQR/1.349}
    cuts: dict = field(default_factory=dict)     # {head: 경보 컷 z}
    n_windows: int = 0
    n_typical: int | None = None   # 이력 창의 median 유효 검출 수 — 분모의 기준

    @property
    def formed(self) -> bool:
        return self.n_windows >= MIN_WINDOWS

    def deviation(self, probs: dict) -> dict:
        """구성비 하나 → 클래스별 이탈 z. 기준선 미형성이면 빈 dict."""
        if not self.formed:
            return {}
        out = {}
        for c in self.classes:
            if c in CHANNEL_KEYS and (c not in probs or c not in self.center):
                continue                    # 안 쟀으면 z 도 없다 — 0 이 아니다
            out[c] = (float(probs.get(c, 0.0)) - self.center[c]) / self.spread[c]
        return out

    def head_score(self, probs: dict, head: str) -> float | None:
        """헤드 하나의 이탈 점수 — 부호는 문헌, 크기는 등가중.

        기준선 어휘에 이 헤드의 신호 행동이 없으면 **None** 이다 — fold()
        는 관측된 행동만 내놓으므로, 이력에서 어휘를 추론해 만든 기준선은
        한 번도 안 보인 행동을 모른다. 그 행동의 편차를 0 으로 치고 점수를
        내면 근거 없는 경보가 된다. `head_support` 와 같은 원칙이다.
        """
        d = self.deviation(probs)
        if not d:
            return None
        signs = HEAD_SIGNS[head]
        if any(c not in d for c in signs):
            return None                              # 어휘 미비 — 못 잰다
        terms = dict(signs)
        for c, sgn in HEAD_EXTRA.get(head, {}).items():
            if c in d:                               # 보조는 있을 때만 합류
                terms[c] = sgn
        return round(sum(s * d[c] for c, s in terms.items()) / len(terms), 3)


def fit(history: list, key: str, classes: tuple | None = None) -> Baseline:
    """창 이력 → 기준선. `history` 는 `[{class: share, ...}, ...]`.

    **미달이면 기준선을 만들지 않는다.** `formed=False` 인 기준선은 편차도
    점수도 내지 않는다 — 이력 서너 개로 만든 중앙값은 기준선이 아니라
    우연이고, 그 위의 경보는 근거가 없다.
    """
    if classes is None:
        seen: set = set()
        for h in history:
            seen |= {k for k in h if not k.startswith("_")}
        classes = tuple(sorted(seen))
    ns = [int(h[COUNT_KEY]) for h in history if h.get(COUNT_KEY)]
    b = Baseline(key=key, classes=tuple(classes), n_windows=len(history),
                 n_typical=int(np.median(ns)) if ns else None)
    if not b.formed:
        return b
    for c in b.classes:
        if c in CHANNEL_KEYS:
            vals = [float(h[c]) for h in history if c in h]
            if len(vals) < MIN_WINDOWS:
                continue                    # 간헐 측정 — 이 채널 기준선 미형성
            x = np.array(vals)
        else:
            x = np.array([float(h.get(c, 0.0)) for h in history])
        b.center[c], b.spread[c] = _robust(x)
    # 경보 컷 — 자기 이력에서 경보율이 대역에 들어오는 z 를 찾는다.
    # 3σ 일괄이 필드마다 0건/10.2% 로 갈렸던 그 사고를 피하는 방법이다.
    for head, signs in HEAD_SIGNS.items():
        if any(c not in b.classes for c in signs):
            b.cuts[head] = None                      # 어휘 미비 — 경보 불가
            continue
        scores = np.array([b.head_score(h, head) for h in history], float)
        b.cuts[head] = _calibrate_cut(scores)
    return b


def _calibrate_cut(scores: np.ndarray, band: tuple = RATE_BAND) -> float | None:
    """이력 점수에서 경보율이 대역에 드는 컷. 못 찾으면 None(경보 불가).

    대역 안에 드는 z 가 여럿이면 **가장 보수적인(큰) 쪽**을 쓴다. 이력이
    상수라 산포가 없으면 어떤 컷도 대역을 못 만족한다 — 그때는 None 이고,
    None 이면 경보를 내지 않는다. 억지로 컷을 만들면 첫 잡음에 울린다.
    """
    n = len(scores)
    if n == 0:
        return None
    best = None
    for z in _Z_GRID:
        rate = float((scores >= z).mean())
        if band[0] <= rate <= band[1]:
            best = float(round(z, 1))
    return best


def measurable(baseline: Baseline, probs: dict, n_used: int | None,
               head: str) -> str | None:
    """이 창의 분모로 이 헤드를 **잴 수 있는가.** 못 재면 이유를 낸다.

    구성비는 유한한 검출에서 나온 비율이라 오차가 있다. 창이 평소
    (`n_typical`, 이력 창의 중앙값)보다 얇을 때 **늘어나는** 표본 분산이
    그 방 기준선 산포(IQR/1.349)를 넘으면, 그 창은 기준선과 대볼 해상도가
    없다 — 편차 z 는 그래도 숫자가 나오지만 그건 표본 잡음을 잰 것이다.

    **문턱을 지어내지 않았다** — 하한이 방마다 자기 산포·자기 평소 두께에서
    나온다. 검출 300개인 창과 3개인 창을 같은 무게로 넣던 것이 이 층의
    구멍이었고, 미측정 채널을 0 으로 채워 z≈−7.4 를 만들던 결함과 같은
    계열이다.
    """
    if n_used is None or baseline.n_typical is None:
        return None                     # 분모를 모르면 검사하지 않는다
    if n_used <= 0:
        return "유효 검출 0개 — 이 창은 구성비 자체가 없다"
    ntyp = baseline.n_typical
    if n_used >= ntyp:
        return None                     # 평소만큼 두꺼우면 물어볼 것이 없다
    for c in HEAD_SIGNS[head]:
        sd = baseline.spread.get(c)
        p = baseline.center.get(c)
        if sd is None or p is None:
            continue
        # **기준선의 p 에서 잰다.** 이 창의 p 로 재면 구성비가 옮겨 간 창
        # (=신호가 있는 창)일수록 SE 가 커져서, 잡으려던 것과 정반대로
        # 진짜 이탈을 먼저 막는다. 처음 짤 때 그렇게 짰다가 시연에서
        # 평시 두께(300개) 창까지 막히는 걸 보고 고쳤다.
        extra = max(p * (1.0 - p), 0.0) * (1.0 / n_used - 1.0 / ntyp)
        if extra > sd * sd:
            return (f"유효 검출 {n_used}개 — 평소 {ntyp}개보다 얇아서 못 "
                    f"잰다. {c} 에서 늘어난 표본 잡음 {math.sqrt(extra):.3f} "
                    f"가 이 방 기준선 산포 {sd:.3f} 를 넘는다")
    return None


def assess(baseline: Baseline, probs: dict, recent: list | None = None,
           heads: tuple | None = None, n_used: int | None = None) -> dict:
    """현재 창 하나를 기준선에 대본다.

    `recent` 는 직전 창들의 구성비(질병처럼 연속을 요구하는 헤드용).
    `heads` 는 **달력이 연 헤드만** 넘기는 자리다 — 발정과 분만은 신호가
    겹쳐서(둘 다 Eating↓·Walking↑) 점수로는 못 가른다. 생략하면 전 헤드를
    계산하지만 그건 시연·감사용이다.
    응답은 판정이 아니라 **의심**이고 그 한계를 스스로 말한다.
    """
    out: dict = {"key": baseline.key, "formed": baseline.formed,
                 "n_windows": baseline.n_windows, "heads": {},
                 "grade": "계산",
                 "weights": "등가중 — 부호만 문헌, 크기는 배운 값이 아니다",
                 "note": ("판정이 아니라 의심이다. '이 방을 오늘 먼저 보라' "
                          "까지이고, 발정 확정은 사람 관찰의 몫이다.")}
    if not baseline.formed:
        out["why"] = (f"기준선 미형성 — 창 {baseline.n_windows}개 < "
                      f"{MIN_WINDOWS}개. 경보를 내지 않는다")
        return out
    for head, signs in HEAD_SIGNS.items():
        if heads is not None and head not in heads:
            continue
        score = baseline.head_score(probs, head)
        unmeasurable = measurable(baseline, probs, n_used, head)
        if unmeasurable:
            score = None                # 못 재는 창은 점수를 내지 않는다
        cut = baseline.cuts.get(head)
        need = SUSTAIN[head]
        streak = 1 if (cut is not None and score is not None
                       and score >= cut) else 0
        if streak and need > 1 and recent:
            for h in reversed(recent[-(need - 1):]):
                s = baseline.head_score(h, head)
                if s is not None and s >= cut:
                    streak += 1
                else:
                    break
        out["heads"][head] = {
            "score": score, "cut": cut,
            "over": bool(cut is not None and score is not None
                         and score >= cut),
            "alert": bool(cut is not None and streak >= need),
            "sustain": need, "streak": streak,
            "n_used": n_used,
            "why": (None if (cut is not None and score is not None) else
                    (unmeasurable if unmeasurable else
                     "기준선 어휘에 이 헤드의 신호 행동이 없다 — 경보 불가"
                     if score is None else
                     "이력 산포가 없어 컷을 못 만든다 — 경보 불가")),
        }
    return out


def summarize(folded: list) -> dict:
    """`vision_pig_behavior.fold()` 출력들 → 키별 창 이력.

    키는 `(barn, pen)` — 방 단위다. 분만틀에서 개체 단위로 쓰려면 호출자가
    자리를 키에 넣는다(자리가 곧 개체이므로 추적이 필요 없다).
    """
    hist: dict = {}
    for r in folded:
        o = r["obs"] if isinstance(r, dict) else r
        key = f"{o.barn}/{o.pen}"
        entry = dict(o.probs)
        if isinstance(r, dict) and r.get("n_used") is not None:
            entry[COUNT_KEY] = int(r["n_used"])   # 분모 — 클래스가 아니다
        # 채널은 구성비(합 1)가 아니라 **따로 온 신호**라 키를 나란히 얹는다.
        # 편차(z)는 무차원이라 스케일이 달라도 섞인다. 0/None 은 "안 쟀다"다.
        if getattr(o, "activity_px", 0.0):
            entry["activity_px"] = float(o.activity_px)
        if getattr(o, "resp_bpm", None) is not None:
            entry["resp_bpm"] = float(o.resp_bpm)
        hist.setdefault(key, []).append(entry)
    return hist


def main() -> int:
    rng = np.random.default_rng(7)
    classes = ("Searching", "Resting", "Walking", "Eating")

    def window(rest, eat, walk):
        raw = {"Resting": rest, "Eating": eat, "Walking": walk,
               "Searching": max(0.0, 1 - rest - eat - walk)}
        tot = sum(raw.values())
        return {k: v / tot for k, v in raw.items()}

    # 평시 3주(하루 2창) — 잡음만. **창마다 유한한 검출에서 뽑는다.**
    # 예전 시연은 구성비를 바로 만들어서 표본 잡음이 0 이었고, 그러면
    # 기준선 산포가 실제보다 좁게 나와 어떤 창이 '못 잴 만큼 얇은지'를
    # 물어볼 수가 없다. 현장의 창은 늘 유한한 검출에서 나온다.
    N_TYPICAL = 300

    def sample(probs, n):
        c = rng.multinomial(n, [probs[k] for k in classes])
        out = {k: v / n for k, v in zip(classes, c)}
        out[COUNT_KEY] = n
        return out

    hist = [sample(window(0.60 + rng.normal(0, .03), 0.25 + rng.normal(0, .03),
                          0.10 + rng.normal(0, .02)), N_TYPICAL)
            for _ in range(42)]
    b = fit(hist, "3동/2방", classes)

    print("=" * 72)
    print("  행동 기준선 층 — 합성 시연 (**등급 합성** — 농장 성적이 아니다)")
    print("=" * 72)
    print(f"  기준선  {b.key} · 창 {b.n_windows}개 · "
          + " · ".join(f"{c} {b.center[c]:.2f}±{b.spread[c]:.3f}"
                       for c in classes))
    print("  경보 컷(자기 이력 경보율 0.5~5% 역산): "
          + " · ".join(f"{h} z≥{b.cuts[h]}" for h in HEAD_SIGNS))

    cases = [
        ("평시 창", window(0.61, 0.24, 0.10)),
        ("발정 의심(불안정+식욕저하)", window(0.42, 0.14, 0.32)),
        ("분만 임박 의심(절식+탐색)", window(0.48, 0.05, 0.22)),
        ("질병 의심(섭식↓·무기력) 1창째", window(0.78, 0.08, 0.05)),
    ]
    recent: list = []
    for name, w in cases:
        a = assess(b, w, recent=recent)
        marks = " · ".join(
            f"{h} {v['score']:+.2f}{'⚠' if v['alert'] else ('·연속대기' if v['over'] else '')}"
            for h, v in a["heads"].items())
        print(f"  {name:<24} {marks}")
        recent.append(w)
    # 질병은 연속 2창을 요구한다 — 같은 창이 한 번 더 오면 운다
    a = assess(b, cases[-1][1], recent=recent)
    d = a["heads"]["disease"]
    print(f"  질병 의심 2창째              disease {d['score']:+.2f}"
          f"{'⚠ 경보' if d['alert'] else ''}  (연속 {d['streak']}/{d['sustain']})")
    # 발정 창이 분만 헤드도 울렸다 — 신호가 겹치기 때문이고, 실전에서는
    # 달력이 연 헤드만 계산한다(공태돈=발정만, 분만틀 D−7~0=분만만)
    g = assess(b, cases[1][1], heads=("estrus",))
    print(f"\n  달력 게이팅: 같은 창을 발정 헤드만 열고 보면 → "
          f"{list(g['heads'])} 만 계산")
    print("  " + a["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
