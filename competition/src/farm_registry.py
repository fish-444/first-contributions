"""농장 구조 등록 + 돼지관리표 — 축사동 → 돈방/군사 → 개체의 3계층.

앱이 현장에서 쓰이려면 먼저 **농장이 어떻게 생겼는지** 알아야 한다. 개체 목록만
있으면 "3동 임신사 두 번째 돈방" 같은 현장 언어로 찾을 수 없고, 카메라가 잡은
위치를 개체에 연결할 수도 없다. 그래서 계층은 세 단계다:

    축사동(barn)  3동 임신사, 4동 분만사 …   용도(stage)와 수용능력을 가진다
      └ 돈방(pen)  군사 돈방 / 스톨 열        사육 방식이 여기서 갈린다
          └ 자리(slot)  스톨 번호 / 군사 내 개체

사육 방식이 계층에 박혀 있는 게 중요하다. **스톨은 자리가 곧 개체 ID** 라 추적이
필요 없지만 활동량 신호가 없고(stall_estrus 로 판정), **군사는 활동량을 쓸 수
있지만 개체 추적이 필요하다**(motion_tracker). 같은 카메라 영상이라도 어느
돈방이냐에 따라 다른 알고리즘을 써야 하므로, 등록 정보가 곧 분석 경로를 정한다.

  Farm.add_barn / add_pen / place      등록
  Farm.table()                          돼지관리표(축사동-돈방-개체)
  Farm.occupancy()                      동·돈방별 수용률
  Farm.locate(id) / Farm.at(barn, pen)  위치 ↔ 개체 조회
  Farm.analysis_route()                 돈방별 적용 알고리즘(사육 방식에 따라)

    python competition/src/farm_registry.py
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 축사 용도 — 번식 단계와 대응한다
BARN_STAGES = {
    "교배사": "이유~교배·임신 초기(스톨). 발정 확인과 교배가 일어나는 곳",
    "임신사": "임신 중기~후기. 군사 또는 스톨",
    "분만사": "분만 전 7일~이유(분만틀)",
    "후보사": "후보돈 순치·초교배 대기",
    # 뒷단은 growth_flow.STAGES 의 구간과 1:1 이다. 하나로 뭉쳐 두면 법정
    # 두당면적(0.30 / 0.45 / 0.80㎡)이 달라서 밀사 판정을 할 수 없다.
    "자돈사": "이유자돈 28~70일령 (법정 0.30㎡)",
    "육성사": "육성돈 70~105일령 (법정 0.45㎡)",
    "비육사": "비육돈 105일령~출하 (법정 0.80㎡). 육성을 겸하는 농장도 많다",
}

# 사육 방식 → 발정 판정 경로. 등록만 하면 분석 방법이 자동으로 정해진다.
HOUSING = {
    "stall": ("스톨(개별)", "stall_estrus", "자세·전환·부동자세",
              "자리가 곧 개체 ID — 추적 불필요, 활동량 신호 없음"),
    "group": ("군사(합사)", "motion_tracker + temporal_features", "활동량·시간 윈도우",
              "활동량을 쓸 수 있으나 개체 추적 필요"),
    "crate": ("분만틀", "-", "-", "번식 판정 대상 아님(분만·포유 관리)"),
    "pen": ("일반 돈방", "-", "-", "비번식(자돈·비육)"),
}


def _slot_key(s):
    """자리 번호 자연 정렬 키 — 숫자면 숫자로, 아니면 문자열로."""
    t = str(s)
    return (0, int(t), "") if t.isdigit() else (1, 0, t)


class Farm:
    """농장 구조 + 개체 배치."""

    def __init__(self, name: str = "농장"):
        self.name = name
        self.barns: dict = {}        # {barn_id: {stage, note}}
        self.pens: dict = {}         # {(barn_id, pen_id): {housing, capacity, ...}}
        self.slots: dict = {}        # {(barn_id, pen_id, slot): animal_id}
        self._where: dict = {}       # {animal_id: (barn_id, pen_id, slot)}

    # -- 등록 ---------------------------------------------------------------
    def add_barn(self, barn_id: str, stage: str, note: str = "") -> "Farm":
        if stage not in BARN_STAGES:
            raise ValueError(f"알 수 없는 축사 용도: {stage} (가능: {list(BARN_STAGES)})")
        self.barns[barn_id] = {"stage": stage, "note": note}
        return self

    def add_pen(self, barn_id: str, pen_id: str, housing: str,
                capacity: int, note: str = "") -> "Farm":
        if barn_id not in self.barns:
            raise KeyError(f"축사동 {barn_id} 이 등록되지 않았다")
        if housing not in HOUSING:
            raise ValueError(f"알 수 없는 사육 방식: {housing} (가능: {list(HOUSING)})")
        if capacity <= 0:
            raise ValueError("수용능력은 1 이상이어야 한다")
        self.pens[(barn_id, pen_id)] = {"housing": housing, "capacity": int(capacity),
                                        "note": note}
        return self

    def place(self, animal_id: str, barn_id: str, pen_id: str,
              slot: str | int | None = None) -> "Farm":
        """개체를 자리에 배치. 이미 다른 곳에 있으면 옮긴다(이중 배치 방지).

        스톨은 slot 이 필수다 — 자리가 곧 개체 ID 이므로 자리 없이 넣으면
        나중에 카메라 화면의 어느 칸인지 되짚을 수 없다.
        """
        key = (barn_id, pen_id)
        if key not in self.pens:
            raise KeyError(f"돈방 {barn_id}-{pen_id} 이 등록되지 않았다")
        pen = self.pens[key]
        if pen["housing"] == "stall" and slot is None:
            raise ValueError(f"{barn_id}-{pen_id} 은 스톨이다 — slot(자리 번호) 필수")
        if animal_id in self._where:                 # 기존 자리 비우기
            self.slots.pop(self._where[animal_id], None)
        if slot is None:                             # 군사: 자리 자동 부여
            slot = 1 + sum(1 for k in self.slots if k[0] == barn_id and k[1] == pen_id)
        sk = (barn_id, pen_id, str(slot))
        if sk in self.slots and self.slots[sk] != animal_id:
            raise ValueError(f"{barn_id}-{pen_id}-{slot} 에 이미 "
                             f"{self.slots[sk]} 이 있다")
        cur = sum(1 for k in self.slots if k[0] == barn_id and k[1] == pen_id)
        if cur >= pen["capacity"] and sk not in self.slots:
            raise ValueError(f"{barn_id}-{pen_id} 수용능력 {pen['capacity']} 초과")
        self.slots[sk] = animal_id
        self._where[animal_id] = sk
        return self

    def remove(self, animal_id: str) -> "Farm":
        """출하·도태·폐사 — 자리를 비운다."""
        sk = self._where.pop(animal_id, None)
        if sk:
            self.slots.pop(sk, None)
        return self

    # -- 조회 ---------------------------------------------------------------
    def locate(self, animal_id: str):
        """개체 → (축사동, 돈방, 자리). 없으면 None."""
        return self._where.get(animal_id)

    def at(self, barn_id: str, pen_id: str | None = None) -> list:
        """위치 → 개체 목록. pen_id 를 생략하면 그 동 전체."""
        hits = [(k, a) for k, a in self.slots.items()
                if k[0] == barn_id and (pen_id is None or k[1] == pen_id)]
        return [a for _, a in sorted(hits, key=lambda x: (x[0][1], _slot_key(x[0][2])))]

    def label(self, animal_id: str) -> str:
        """현장에서 부르는 위치 문자열 — '3동 임신사 A열 12번'."""
        sk = self._where.get(animal_id)
        if not sk:
            return "미배치"
        b, p, s = sk
        return f"{b} {self.barns[b]['stage']} {p} {s}번"

    # -- 표 -----------------------------------------------------------------
    def table(self, herd: pd.DataFrame | None = None) -> pd.DataFrame:
        """돼지관리표 — 축사동·돈방·자리·개체(+ 번식 상태).

        herd 를 주면(herd_board.build_herd 결과) 단계·산차·예정일이 붙는다.
        """
        rows = []
        for (b, p, s), aid in self.slots.items():
            pen = self.pens[(b, p)]
            h = HOUSING[pen["housing"]]
            rows.append({"barn": b, "stage": self.barns[b]["stage"],
                         "pen": p, "slot": s, "id": aid,
                         "housing": pen["housing"], "housing_kr": h[0]})
        df = pd.DataFrame(rows)
        if not len(df):
            return df
        if herd is not None and len(herd):
            df = df.merge(herd, on="id", how="left", suffixes=("", "_h"))
        # 자리 번호는 문자열이라 그냥 정렬하면 1,10,11,12,2 순이 된다.
        # 스톨 번호는 카메라 화면의 물리적 순서와 같아야 읽을 수 있다.
        df["_k"] = df["slot"].map(_slot_key)
        return (df.sort_values(["barn", "pen", "_k"]).drop(columns="_k")
                  .reset_index(drop=True))

    def occupancy(self) -> pd.DataFrame:
        """돈방별 수용률 — 과밀·공실을 한눈에."""
        rows = []
        for (b, p), pen in self.pens.items():
            n = sum(1 for k in self.slots if k[0] == b and k[1] == p)
            rows.append({"barn": b, "stage": self.barns[b]["stage"], "pen": p,
                         "housing": pen["housing"], "n": n,
                         "capacity": pen["capacity"],
                         "rate": round(n / pen["capacity"], 3),
                         "free": pen["capacity"] - n})
        return pd.DataFrame(rows).sort_values(["barn", "pen"]).reset_index(drop=True)

    def barn_summary(self, herd: pd.DataFrame | None = None) -> pd.DataFrame:
        """축사동별 두수 + (herd 가 있으면) 번식 단계 구성."""
        t = self.table(herd)
        if not len(t):
            return t
        g = t.groupby(["barn", "stage"], sort=False).agg(n=("id", "count"))
        if herd is not None and "stage_h" in t.columns:
            piv = t.pivot_table(index=["barn", "stage"], columns="stage_h",
                                values="id", aggfunc="count", fill_value=0)
            g = g.join(piv)
        return g.reset_index()

    def misplaced(self, herd: pd.DataFrame) -> pd.DataFrame:
        """축사 용도와 번식 단계가 어긋난 개체 — 이동 누락을 잡아낸다.

        분만 임박한 모돈이 교배사에 남아 있거나 이유한 모돈이 분만사를 차지하고
        있으면 그 자체가 사고다. 규칙은 단계별 허용 구간으로 두되, 경계는
        느슨하게 잡는다 — 교배사의 임신 초기와 분만사의 분만 직전은 정상이다.
        """
        allow = {
            "교배사": {"공태", "교배", "임신"},   # 임신은 초기 한정(아래에서 검사)
            "임신사": {"임신", "교배"},
            "분만사": {"포유", "임신"},           # 임신은 분만 임박 한정
            "후보사": {"후보", "공태"},
            "자돈사": set(), "육성사": set(), "비육사": set(),
        }
        t = self.table(herd)
        if not len(t) or "stage_h" not in t.columns:
            return pd.DataFrame()
        rows = []
        for r in t.itertuples(index=False):
            st, bs = r.stage_h, r.stage
            if not isinstance(st, str):
                continue
            why = None
            if st not in allow.get(bs, set()):
                why = f"{bs}에 {st} 개체 — 축사 용도와 불일치"
            elif bs == "교배사" and st == "임신" and (r.week or 0) > 5:
                why = f"임신 {int(r.week)}주인데 교배사에 있음 — 임신사 이동 누락"
            elif bs == "분만사" and st == "임신" and (r.d_day is not None
                                                 and r.d_day > 14):
                why = f"분만 {int(r.d_day)}일 남았는데 분만사 점유 — 자리 낭비"
            if why:
                rows.append({"id": r.id, "loc": f"{r.barn} {r.pen} {r.slot}번",
                             "barn_stage": bs, "repro_stage": st, "reason": why})
        return pd.DataFrame(rows)

    def analysis_route(self) -> pd.DataFrame:
        """돈방마다 어떤 발정 판정 경로를 쓸지 — 등록이 곧 분석 설계.

        번식 축사(교배사·임신사·후보사)만 발정 판정 대상이다. 분만사·자돈사에
        발정 알고리즘을 돌리는 것은 의미가 없으므로 여기서 걸러낸다.
        """
        repro = {"교배사", "임신사", "후보사"}
        rows = []
        for (b, p), pen in self.pens.items():
            st = self.barns[b]["stage"]
            kr, mod, sig, note = HOUSING[pen["housing"]]
            target = st in repro and pen["housing"] in ("stall", "group")
            rows.append({"barn": b, "stage": st, "pen": p, "housing_kr": kr,
                         "estrus_target": target,
                         "module": mod if target else "-",
                         "signal": sig if target else "-",
                         "note": note})
        return pd.DataFrame(rows).sort_values(["barn", "pen"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 번식주기 상수 — pigflow 기본값과 같은 값을 쓴다. 여기서 새로 적으면
# 같은 농장에 대해 두 모듈이 다른 배치를 만든다.
W2S, GEST, LACT = 7, 114, 24      # 이유~교배 · 임신 · 포유
CONFIRM = 28                      # 임신확인(재발정 확인)까지
PRE_FARROW = 7                    # 분만사 사전 이동
GILT_SHARE = 0.05                 # 후보사는 주기 밖 — 관행값


def stage_counts(n_sows: int) -> dict:
    """상시모돈 → 번식 축사별 두수. **비율은 유도되는 값이지 고르는 값이 아니다.**

    정상 상태에서

        단계별 두수 = 총두수 × (그 단계에 머무는 일수 ÷ 번식주기)

    이다. 처음에 25/55/15/5 로 눈대중해 적었다가 주기 일수로 검산하니 분만사가
    45두 vs 64두로 19두 어긋났고, pigflow 가 같은 농장에 대해 낸 분만틀 수와도
    맞지 않았다. 기본값 기준으로 교배사 24.1% · 임신사 54.5% · 분만사 21.4%.
    """
    n = max(1, int(n_sows))
    cycle = W2S + GEST + LACT
    seg = {"교배사": W2S + CONFIRM,
           "임신사": GEST - CONFIRM - PRE_FARROW,
           "분만사": PRE_FARROW + LACT}
    n_gilt = max(1, round(n * GILT_SHARE))
    body = n - n_gilt
    mate = max(1, round(body * seg["교배사"] / cycle))
    gest = max(1, round(body * seg["임신사"] / cycle))
    return {"교배사": mate, "임신사": gest,
            "분만사": max(1, body - mate - gest), "후보사": n_gilt}


RETURN_CHECK_DAYS = 21            # 재발은 교배 후 3주에 드러난다


def stage_of(rec: dict, on) -> tuple[str | None, str, str]:
    """**개체 하나가 그날 어느 축사에 있는가.** `stage_counts` 의 반대다.

    `stage_counts` 는 총두수를 주기 비율로 나눈 *유도값*이고, 이건 개체
    이력에서 **세는** 것이다. 그래서 실제 농장에서는 셋이 25/55/21 처럼
    깔끔하게 안 나온다 — 계절·사고·재발이 분포를 찌그러뜨린다. 그 찌그러짐이
    보이는 게 이 함수를 쓰는 이유다.

    **경계를 여기서 새로 정하면 안 된다.** `herd_board` 는 교배/임신을 21일에
    가르는데(재발 확인 시점) 축사 이동은 `CONFIRM` 28일에 일어난다. 둘을
    섞어 쓰면 유도값과 실측값의 차이에 **정의 차이**가 섞여 들어가, 데이터가
    말하는 것인지 경계를 옮긴 것인지 구분할 수 없게 된다. 그래서 여기서는
    `stage_counts` 가 쓰는 W2S·CONFIRM·PRE_FARROW·LACT 그대로 간다.

    돌려주는 것: `(축사, 사유코드, 근거)`. 축사가 None 이면 그날 어디에도 못
    놓은 것이고, 사유코드가 왜인지 말한다 — **조용히 빼면 두수가 맞지 않는데
    이유를 모른다.** 코드와 문장을 따로 두는 건 집계 때문이다. 문장에 일수가
    박혀 있어서 그걸로 묶으면 "분만 39일째" 3두, "분만 41일째" 5두 처럼
    한 두씩 흩어져 **몇 두가 왜 빠졌는지가 안 보인다.**
    """
    import repro_calendar as rc

    t0 = rc._d(on)

    def d(k):
        v = rec.get(k)
        if v is None or (isinstance(v, float) and v != v) or v == "":
            return None
        return rc._d(v)

    # parity 는 셋으로 갈린다: 값 / 빈 칸 / 쓰레기. 빈 칸은 pandas 가 NaN 으로
    # 주는데 **NaN 은 truthy 라** `or 0` 이 안 걸리고 int(nan) 이 터진다 —
    # 한 칸 때문에 전체 집계가 죽는다. 모르면 None 으로 두고 날짜로만 판정한다.
    parity = None
    v = rec.get("parity")
    if v is not None and v != "":
        try:
            f = float(v)
            if f == f:                     # NaN 걸러내기
                parity = int(f)
        except (TypeError, ValueError):
            parity = None                  # 쓰레기 값 — 산차를 모르는 것과 같다

    wea, svc, far = d("weaning_date"), d("service_date"), d("farrow_date")

    # 산차 0 = 후보돈, 단 **교배 기록이 없을 때만.** 실농장 내보내기는 산차를
    # 분만 횟수로 세므로, 초교배 후 첫 분만을 기다리는 후보돈도 산차 0 이다 —
    # 날짜보다 먼저 걸러 버리면 그 개체들이 전부 후보사로 잘못 집계된다.
    if parity is not None and parity <= 0 and svc is None:
        return "후보사", "gilt", "산차 0 — 초교배 전"
    # 같은 날짜면 **뒤 사건이 이긴다.** 이유와 교배가 같은 날인 기록은 흔한데
    # (이유 당일 발정), 문자열로 정렬하면 "wea" > "svc" 라 교배가 없던 일이
    # 되고 그 모돈은 영원히 공태로 남는다.
    order = {"wea": 0, "svc": 1, "far": 2}
    past = [(x, order[k], k) for x, k in
            ((wea, "wea"), (svc, "svc"), (far, "far"))
            if x is not None and x <= t0]
    if not past:
        return None, "before_record", "그날 이전 기록이 없다"
    last = max(past)[2]

    cycle = W2S + GEST + LACT              # 145일 — 한 번식주기

    if last == "wea":
        od = (t0 - wea).days
        # 이유했고 아직 교배 안 됐다 — 공태. 교배사에 있다. 공태가 길어지는
        # 것 자체는 실재하는 문제라(NPD 의 원천) 문턱으로 자르면 안 되지만,
        # **한 주기(145일)를 통째로 넘기고도 교배가 없는 기록**은 도태·폐사로
        # 끊긴 것이다 — 분만 뒤 이유가 없는 기록과 같은 취급을 한다.
        if od > cycle:
            return (None, "record_ends",
                    f"이유 {od}일째인데 교배 기록이 없다 — 기록이 여기서 끝난다")
        return "교배사", "open", f"이유 {od}일째 · 미교배"

    if last == "far":
        if (t0 - far).days < LACT:
            return "분만사", "lactating", f"분만 {(t0 - far).days}일째 · 포유 중"
        # 포유기간이 지났는데 이유 기록이 없다. **추정해서 넣지 않는다** —
        # 이유일은 다음 주기 전체의 기준점이라 틀리면 뒤가 다 밀린다.
        return (None, "record_ends",
                f"분만 {(t0 - far).days}일째인데 이유 기록이 없다 — "
                f"기록이 여기서 끝난다")

    # 마지막 사건이 교배
    gd = (t0 - svc).days
    if rec.get("outcome") == "재발" and gd >= RETURN_CHECK_DAYS:
        # 재발이 확인됐는데 재교배 기록이 없다. 임신사로 보내면 있지도 않은
        # 임신돈이 생긴다 — 재발돈은 교배사로 돌아온다.
        return "교배사", "returned", f"교배 {gd}일째 · 재발 확인 후 재교배 대기"
    if gd < CONFIRM:
        return "교배사", "served", f"교배 {gd}일째 · 임신확인 전"
    exp_far = far or (svc + timedelta(days=GEST))
    # 예정일이 임신기간 실측 폭(111~120일)을 넘겨 지났는데 분만 기록이 없다 —
    # 유산·도태로 기록이 끊긴 것이다. 안 자르면 그 개체가 **분만사에 영구
    # 배치**되고 근거에 "예정 -213일 전" 같은 있지도 않은 날짜가 찍힌다.
    if far is None and t0 > exp_far + timedelta(days=7):
        return (None, "record_ends",
                f"분만 예정 {(t0 - exp_far).days}일 지났는데 분만 기록이 "
                f"없다 — 기록이 여기서 끝난다")
    if t0 >= exp_far:
        return ("분만사", "pre_farrow",
                f"분만 예정 {(t0 - exp_far).days}일 경과 · 분만 대기")
    if t0 >= exp_far - timedelta(days=PRE_FARROW):
        return ("분만사", "pre_farrow",
                f"분만 예정 {(exp_far - t0).days}일 전 · 사전 이동")
    return "임신사", "pregnant", f"임신 {gd}일째"


def counts_from_herd(records, on) -> dict:
    """개체 이력 → 그날의 축사별 **실제** 두수.

    `stage_counts(n)` 은 총두수 하나에서 비율로 되푼 값이라 늘 매끈하다.
    이건 세는 것이라 안 매끈하고, **못 놓은 개체를 이유별로 남긴다.**
    """
    import repro_calendar as rc

    counts = {st: 0 for st in ("교배사", "임신사", "분만사", "후보사")}
    unplaced: dict = {}
    for r in records:
        st, code, _why = stage_of(r, on)
        if st is None:
            unplaced[code] = unplaced.get(code, 0) + 1
        else:
            counts[st] += 1
    n = sum(counts.values())
    return {
        "on": rc._d(on), "counts": counts, "n": n,
        "n_records": len(records) if hasattr(records, "__len__") else None,
        "unplaced": unplaced,
        "share": {k: (round(v / n, 3) if n else 0.0)
                  for k, v in counts.items()},
    }


def herd_from_csv(path: str) -> tuple[list, str | None]:
    """`synth_farm --csv` 가 낸 개체 이력을 읽는다. `(기록, 기준일)`.

    **이 함수는 합성 전용이 아니다.** 같은 열 이름(id·parity·weaning_date·
    service_date·farrow_date·outcome)이면 실농장 내보내기도 그대로 들어온다 —
    그날 ③단계의 등급이 `합성` 에서 `실측` 으로 바뀔 뿐이다.

    **기준일을 같이 돌려주는 게 요점이다.** 이 파일은 개체별로 그 날짜 이전의
    마지막 주기 한 줄이라 **하루의 스냅숏**이고, 다른 날짜로 읽으면 한 주기
    (145일)를 벗어난 개체가 통째로 빠진다. 기준일 없이 오늘 날짜로 읽었다가
    283두 중 175두만 잡히고 분만사 비중이 18%→37%로 부푼 적이 있다.
    """
    df = pd.read_csv(path)
    miss = {"id", "parity"} - set(df.columns)
    if miss:
        raise ValueError(f"열이 없다: {sorted(miss)} — "
                         f"있는 열 {list(df.columns)}")
    as_of = None
    if "as_of" in df.columns and len(df):
        vals = sorted({str(v) for v in df["as_of"].dropna()})
        if len(vals) > 1:
            raise ValueError(f"as_of 가 여러 날짜다 {vals[:3]} — "
                             f"스냅숏 파일은 하루여야 한다")
        as_of = vals[0] if vals else None
    # **빈 칸은 NaN 이 아니라 None 으로 낸다.** pandas 가 주는 NaN 은 JSON 에
    # 실리지 않아 API 로 그대로 넘기면 직렬화가 깨지고, `NaN or 0` 같은 흔한
    # 관용구도 NaN 이 truthy 라 조용히 틀린다. 비운 것은 비운 채로 둔다는
    # 이 프로젝트의 규칙과도 None 쪽이 맞다.
    return [{k: (None if v is None or (isinstance(v, float) and v != v) else v)
             for k, v in r.items()} for r in df.to_dict("records")], as_of


# 자리 번호가 있는 사육 방식 — 스톨·분만틀은 몇 번 자리인지가 관리 단위다
NUMBERED = ("stall", "crate")


def farm_from_setup(setup: dict, n_sows: int | None = None,
                    want: dict | None = None) -> tuple:
    """등록 화면 JSON → `Farm`. **방을 만들어 내지 않는다.**

    `demo_farm` 은 두수에 맞춰 방을 지어 내므로 늘 딱 들어맞는다 — 그래서
    "자리가 모자란다" 는 사실이 절대 안 보인다. 이 함수는 반대로 **등록한
    방만** 쓰고, 못 넣은 두수를 둘째 값으로 돌려준다. 조용히 넘기면 사용자는
    자기 농장이 그대로 반영된 줄 안다.

    `want` 를 주면 그 두수를 놓는다 — `counts_from_herd` 로 **개체 이력에서
    센 값**을 넣는 자리다. 안 주면 `stage_counts` 로 되푼 유도값을 쓴다.
    둘의 차이가 이 프로그램에서 유도와 실측이 갈리는 마지막 지점이라, 부르는
    쪽이 어느 것인지 반드시 표시해야 한다. 뒷단(자돈·육성·비육)은 개체 단위로
    관리하지 않으므로 여기서 놓지 않는다 — 그건 돈군흐름 쪽이다.
    """
    barns = setup.get("barns") or []
    n = int(n_sows or setup.get("n_sows") or 0)
    want = dict(want) if want else stage_counts(n)
    f = Farm(setup.get("name") or "내 농장")

    by_stage: dict = {}
    for b in barns:
        st = b.get("stage")
        if st not in want:                 # 뒷단은 개체 배치 대상이 아니다
            continue
        bid = str(b.get("name") or st)
        housing = b.get("housing") or "group"
        rooms = max(1, int(b.get("rooms") or 1))
        per = max(1, int(b.get("per") or 1))
        f.add_barn(bid, st, f"등록 {rooms}방 × {per}자리")
        pens = []
        for i in range(1, rooms + 1):
            pen = f"{i}방"
            f.add_pen(bid, pen, housing, per)
            pens.append(pen)
        by_stage.setdefault(st, []).append((bid, pens, per, housing))

    idx, notes = 0, []
    for st, need in want.items():
        got = by_stage.get(st)
        if not got:
            notes.append((st, need, 0, "등록된 동이 없다"))
            continue
        done = 0
        for bid, pens, per, housing in got:
            for pen in pens:
                for slot in range(1, per + 1):
                    if done >= need:
                        break
                    f.place(f"{2000 + idx}", bid, pen,
                            slot if housing in NUMBERED else None)
                    idx += 1
                    done += 1
        if done < need:
            notes.append((st, need, done, f"자리 {need - done}두분 부족"))
    return f, notes


def demo_farm(n_sows: int = 68, want: dict | None = None) -> Farm:
    """전형적인 일관농장 구조 — 교배사(스톨)·임신사(군사)·분만사·후보사.

    **자리 수를 두수에 맞춰 만든다.** 처음엔 배치가 68두로 고정돼 있어서
    n_sows 를 300 으로 줘도 68두만 배치되고 나머지는 조용히 사라졌다.
    실제 농장 도면이 들어오면 이 함수 대신 Farm 을 직접 구성한다.

    **구성 비율을 눈대중으로 정하면 안 된다.** 처음에 25/55/15/5 로 적었다가
    주기 일수로 검산하니 분만사가 45두 vs 64두로 19두 어긋났고, pigflow 가
    같은 농장에 대해 계산한 분만틀 60틀과도 맞지 않았다. 정상 상태에서

        단계별 두수 = 총두수 × (그 단계에 머무는 일수 ÷ 번식주기)

    이므로 비율은 유도되는 값이지 고르는 값이 아니다. 기본값 기준:

        교배사  이유~임신확인      7 + 28 =  35일 → 24.1%
        임신사  임신확인~분만7일전 114-28-7 = 79일 → 54.5%
        분만사  분만7일전~이유      7 + 24 =  31일 → 21.4%

    후보사는 주기 밖(전입 대기)이라 별도로 얹는다.
    """
    f = Farm("시연농장")
    (f.add_barn("1동", "교배사", "이유 후 발정 확인 + 교배")
      .add_barn("2동", "임신사", "임신 확인 후 군사 전환")
      .add_barn("3동", "분만사", "분만 7일 전 이동")
      .add_barn("4동", "후보사", "순치 중인 후보돈"))

    # 비율 유도는 stage_counts 하나로 모았다 — 등록 농장(farm_from_setup)과
    # 시연 농장이 다른 비율을 쓰면 같은 두수가 다르게 배치된다.
    #
    # `want` 를 주면 그걸 놓는다. 개체 이력을 셌는데 여기서 유도값을 쓰면
    # **한 화면에 두 값이 찍힌다** — 머리글은 65두, 바로 아래 표는 75두.
    want = dict(want) if want else stage_counts(n_sows)
    n_mate, n_gest = want["교배사"], want["임신사"]
    n_farrow, n_gilt = want["분만사"], want["후보사"]

    # 교배사: 스톨 열당 12자리
    STALL_PER_COL = 12
    cols = [f"{chr(ord('A') + i)}열" for i in
            range(max(1, -(-n_mate // STALL_PER_COL)))]
    for c in cols:
        f.add_pen("1동", c, "stall", STALL_PER_COL)
    # 임신사: 군사 방당 10두
    GROUP_SIZE = 10
    gpens = [f"{i}방" for i in range(1, max(1, -(-n_gest // GROUP_SIZE)) + 1)]
    for g in gpens:
        f.add_pen("2동", g, "group", GROUP_SIZE)
    # 분만사: 분만틀 방당 12틀
    CRATE_PER_ROOM = 12
    frooms = [f"분만{i}실" for i in
              range(1, max(1, -(-n_farrow // CRATE_PER_ROOM)) + 1)]
    for r in frooms:
        f.add_pen("3동", r, "crate", CRATE_PER_ROOM)
    f.add_pen("4동", "순치방", "group", max(1, n_gilt))

    idx = 0

    def place_slots(barn, pens, per, want):
        """번호 자리가 있는 돈방(스톨·분만틀)에 채운다."""
        nonlocal idx
        done = 0
        for pen in pens:
            for s in range(1, per + 1):
                if done >= want:
                    return
                f.place(f"{2000 + idx}", barn, pen, s)
                idx += 1
                done += 1

    def place_group(barn, pens, per, want):
        """자리 번호가 없는 군사방."""
        nonlocal idx
        done = 0
        for pen in pens:
            for _ in range(per):
                if done >= want:
                    return
                f.place(f"{2000 + idx}", barn, pen)
                idx += 1
                done += 1

    place_slots("1동", cols, STALL_PER_COL, n_mate)
    place_group("2동", gpens, GROUP_SIZE, n_gest)
    place_slots("3동", frooms, CRATE_PER_ROOM, n_farrow)
    place_group("4동", ["순치방"], max(1, n_gilt), n_gilt)
    return f


def main() -> int:
    f = demo_farm()
    print(f"=== {f.name} 구조 등록 ===")
    for b, meta in f.barns.items():
        pens = [(p, v) for (bb, p), v in f.pens.items() if bb == b]
        cap = sum(v["capacity"] for _, v in pens)
        print(f"  {b} {meta['stage']:<5} 돈방 {len(pens)}개 · 수용 {cap}두 — {meta['note']}")
        for p, v in pens:
            print(f"     └ {p:<6} {HOUSING[v['housing']][0]:<8} 수용 {v['capacity']:>3}두")

    print("\n=== 돼지관리표 (축사동-돈방-개체) ===")
    t = f.table()
    print(f"  총 {len(t)}두")
    print(f"  {'축사동':<5} {'용도':<5} {'돈방':<6} {'자리':>4} {'개체':>6} {'사육방식':<8}")
    for r in t.head(8).itertuples(index=False):
        print(f"  {r.barn:<5} {r.stage:<5} {r.pen:<6} {r.slot:>4} {r.id:>6} "
              f"{r.housing_kr:<8}")
    print(f"  … 이하 {max(0, len(t) - 8)}행")

    print("\n=== 수용률 ===")
    for r in f.occupancy().itertuples(index=False):
        bar = "█" * int(r.rate * 20)
        print(f"  {r.barn:<5} {r.pen:<6} {r.n:>3}/{r.capacity:<3} "
              f"{r.rate:>6.0%} {bar:<20} 여유 {r.free}두")

    print("\n=== 위치 조회 ===")
    print(f"  2003 → {f.label('2003')}")
    print(f"  2030 → {f.label('2030')}")
    print(f"  1동 A열 개체: {', '.join(f.at('1동', 'A열')[:6])} …")

    print("\n=== 번식 상태 결합 (herd_board) ===")
    import herd_board as hb
    ids = sorted(f._where)
    recs = hb.generate_demo(n=len(ids) + 40, today="2026-08-10")[:len(ids)]
    for r, i in zip(recs, ids):
        r["id"] = i
    herd = hb.build_herd(recs, today="2026-08-10")
    print(f.barn_summary(herd).to_string(index=False))

    mp = f.misplaced(herd)
    print(f"\n=== 배치 오류 {len(mp)}두 (축사 용도 ↔ 번식 단계 불일치) ===")
    for r in mp.head(6).itertuples(index=False):
        print(f"  {r.id} {r.loc:<16} {r.reason}")
    print("  ※ 합성 데이터라 배치가 무작위다. 실제 농장에서는 이 목록이 곧"
          "\n    '이동 누락' 알림이 된다 — 분만 임박한 모돈이 교배사에 남아 있는 식.")

    print("\n=== 돈방별 발정 판정 경로 (등록이 곧 분석 설계) ===")
    for r in f.analysis_route().itertuples(index=False):
        mark = "○" if r.estrus_target else "·"
        print(f"  {mark} {r.barn:<4} {r.pen:<6} {r.housing_kr:<8} "
              f"{r.module:<32} {r.note}")

    print("\n※ 사육 방식이 계층에 박혀 있어야 한다. 스톨은 자리가 곧 개체 ID 라 추적이"
          "\n  필요 없는 대신 활동량 신호가 없고, 군사는 그 반대다. 같은 카메라라도"
          "\n  돈방에 따라 다른 알고리즘을 써야 하므로 등록 정보가 분석 경로를 정한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
