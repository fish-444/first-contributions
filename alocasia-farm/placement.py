"""배치 최적화 — 어느 화분에 어느 식물을 두면 잘 자라나

문제: 화분 자리는 고정이고 식물은 옮길 수 있다. 어떤 식물을 어느 자리에 두면
전체가 가장 잘 자라는가. 답이 자명하지 않은 이유는 **식물끼리 서로 영향을 주기**
때문이다 — 대품을 밝은 자리에 두면 그 그늘에 옆 소품이 들어가고, 잎이 빽빽한
개체를 팬 바람이 안 닿는 구석에 두면 무름병이 온다.

세 가지를 계산한다:

  1. 조도    조명에서 오는 빛. 거리²반비례 × 입사각 cos.
  2. 그늘    키 큰 이웃의 잎우산이 덮는 만큼 깎인다 (원-원 겹침 넓이).
  3. 통풍    팬 바람. 거리로 약해지고, 바람 위쪽(upwind) 잎에 막힌다.

좌표는 전부 실좌표 cm (`_cm`). 선반 60 x 40 cm 의 중앙이 원점이고,
y 는 높이다. 외부 의존성 없이 순수 파이썬으로만 계산한다.
"""

import functools
import math
import os
from typing import Dict, List

# --------------------------------------------------------------------------- 환경
# 기본값은 60x40cm 선반 위 40cm 높이에 LED 바 두 줄, 뒤쪽에 서큘레이터 하나.
# 실제 케이지에 맞춰 POST /api/environment 로 바꿀 수 있다.
DEFAULT_LIGHTS = [{"x": -15.0, "y": 40.0, "z": 0.0, "power": 1.0},
                  {"x": 15.0, "y": 40.0, "z": 0.0, "power": 1.0}]
# 팬은 위치와 '부는 방향'이 함께 있어야 한다. dx/dz 는 바람이 가는 쪽.
DEFAULT_FANS = [{"x": 0.0, "y": 25.0, "z": -28.0, "dx": 0.0, "dz": 1.0, "power": 1.0}]

# 잎이 모인 높이(cm). 빛이 실제로 닿아 광합성하는 지점.
CANOPY_Y_CM = float(os.environ.get("CANOPY_Y_CM", "18"))
# 바람이 거리에 따라 약해지는 정도 — 이만큼 멀어지면 절반이 된다
AIR_HALF_CM = float(os.environ.get("AIR_HALF_CM", "35"))
# 점수에서 빛과 통풍의 비중
LIGHT_WEIGHT = float(os.environ.get("LIGHT_WEIGHT", "0.6"))

# 크기 등급별 기본 몸집 — 잎을 실측(leaf_max_cm)했으면 그쪽이 우선이다.
# 잎우산 반지름 ≈ 잎 긴 변 길이 (잎자루가 사방으로 뻗는 알로카시아 기준).
GRADE_SHAPE = {"소품": (7.0, 14.0), "중품": (13.0, 28.0), "대품": (20.0, 45.0)}
FALLBACK_SHAPE = (10.0, 20.0)          # 미검출 등


def plant_shape(plant: dict) -> tuple:
    """식물 → (잎우산 반지름 cm, 키 cm). 실측이 있으면 실측을 쓴다."""
    radius_cm, height_cm = GRADE_SHAPE.get(plant.get("size_class"), FALLBACK_SHAPE)
    measured = plant.get("leaf_max_cm")
    if measured:
        radius_cm = float(measured)
        height_cm = radius_cm * 2.2       # 알로카시아는 잎보다 키가 크다
    return radius_cm, height_cm


# --------------------------------------------------------------------------- 빛
def illuminance(x_cm: float, z_cm: float, lights: List[dict] = None,
                canopy_y_cm: float = None) -> float:
    """한 지점이 조명들로부터 받는 빛의 양 (상대값).

    거리²에 반비례하고, 위에서 내리꽂을수록(입사각 cos) 잘 받는다.
    옆에서 스치는 빛은 잎을 못 데운다.
    """
    lights = DEFAULT_LIGHTS if lights is None else lights
    canopy_y_cm = CANOPY_Y_CM if canopy_y_cm is None else canopy_y_cm
    total = 0.0
    for lamp in lights:
        dx = lamp["x"] - x_cm
        dy = lamp["y"] - canopy_y_cm
        dz = lamp["z"] - z_cm
        d2 = dx * dx + dy * dy + dz * dz
        if d2 <= 0:
            continue
        cos = dy / math.sqrt(d2)          # 잎이 위를 보고 있다고 본다
        if cos > 0:
            total += lamp.get("power", 1.0) * cos / d2
    return total


# 최적화는 같은 (반지름, 반지름, 거리) 조합을 수천 번 다시 묻는다 — 자리도 몸집도
# 값의 가짓수가 적어서 캐시가 거의 다 맞는다.
@functools.lru_cache(maxsize=200_000)
def _circle_overlap(r1_cm: float, r2_cm: float, d_cm: float) -> float:
    """두 원이 겹치는 넓이. 그늘이 잎우산을 얼마나 덮는지 재는 데 쓴다."""
    if d_cm >= r1_cm + r2_cm:
        return 0.0
    if d_cm <= abs(r1_cm - r2_cm):
        return math.pi * min(r1_cm, r2_cm) ** 2
    a1 = math.acos(max(-1.0, min(1.0, (d_cm * d_cm + r1_cm ** 2 - r2_cm ** 2) / (2 * d_cm * r1_cm))))
    a2 = math.acos(max(-1.0, min(1.0, (d_cm * d_cm + r2_cm ** 2 - r1_cm ** 2) / (2 * d_cm * r2_cm))))
    return (r1_cm ** 2 * (a1 - math.sin(2 * a1) / 2)
            + r2_cm ** 2 * (a2 - math.sin(2 * a2) / 2))


def shade_factor(here: dict, neighbours: List[dict]) -> float:
    """이웃의 그늘에 덮인 비율 0~1. 나보다 키 큰 이웃만 그늘을 만든다.

    같은 키끼리는 서로 가리지 못한다 — 옆으로 겹칠 뿐 위를 덮지 않는다.
    """
    r_cm, h_cm = here["r_cm"], here["h_cm"]
    mine = math.pi * r_cm * r_cm
    if mine <= 0:
        return 0.0
    covered = 0.0
    for other in neighbours:
        if other is here or other["h_cm"] <= h_cm:
            continue
        d_cm = math.dist((here["x_cm"], here["z_cm"]), (other["x_cm"], other["z_cm"]))
        overlap = _circle_overlap(r_cm, other["r_cm"], d_cm)
        # 키 차이가 클수록 더 확실히 덮는다 (바로 위를 지나간다)
        lead = min(1.0, (other["h_cm"] - h_cm) / 20.0)
        covered += overlap * lead
    return min(1.0, covered / mine)


# --------------------------------------------------------------------------- 바람
def airflow(x_cm: float, z_cm: float, fans: List[dict] = None,
            blockers: List[dict] = None) -> float:
    """한 지점의 바람 세기 0~1 (가장 센 팬 기준).

    팬 정면일수록 세고, 멀수록 약해지고, 바람이 오는 쪽에 잎이 있으면 막힌다.
    잎 사이 습기가 안 빠지면 무름병이 오므로 빽빽한 개체일수록 중요하다.
    """
    fans = DEFAULT_FANS if fans is None else fans
    blockers = blockers or []
    best = 0.0
    for fan in fans:
        to_x, to_z = x_cm - fan["x"], z_cm - fan["z"]
        d_cm = math.hypot(to_x, to_z)
        if d_cm <= 0.01:
            best = max(best, fan.get("power", 1.0))
            continue
        wind = math.hypot(fan.get("dx", 0.0), fan.get("dz", 1.0))
        if wind <= 0:
            continue
        # 팬이 부는 방향과 얼마나 일치하나 (뒤쪽은 바람이 안 온다)
        cos = (to_x * fan.get("dx", 0.0) + to_z * fan.get("dz", 1.0)) / (d_cm * wind)
        if cos <= 0:
            continue
        strength = fan.get("power", 1.0) * cos * (0.5 ** (d_cm / AIR_HALF_CM))

        # 바람 위쪽(팬과 나 사이)에 있는 잎우산이 바람을 깎는다
        ux, uz = to_x / d_cm, to_z / d_cm
        blocked = 0.0
        for b in blockers:
            bx, bz = b["x_cm"] - fan["x"], b["z_cm"] - fan["z"]
            along = bx * ux + bz * uz               # 팬→나 축 위의 위치
            if not (0 < along < d_cm):
                continue                            # 내 뒤이거나 팬 뒤쪽
            off = abs(bx * (-uz) + bz * ux)         # 축에서 옆으로 벗어난 거리
            if off < b["r_cm"]:
                blocked += (1 - off / b["r_cm"]) * min(1.0, b["r_cm"] / 20.0)
        best = max(best, strength * max(0.0, 1 - min(0.85, blocked)))
    return best


# --------------------------------------------------------------------------- 점수
def _need_light(plant: dict, r_cm: float) -> float:
    """잎이 넓을수록 빛을 많이 써야 한다 (잎면적에 대략 비례)."""
    return max(0.35, min(1.0, (r_cm / 20.0) ** 1.5))


def _need_air(plant: dict) -> float:
    """잎이 서로 겹쳐 습기가 갇힌 개체일수록 바람이 절실하다."""
    density = (plant.get("overlap_density") or 0) / 100.0
    return max(0.25, min(1.0, 0.25 + density))


def geometry(spots: List[dict], lights=None, fans=None) -> dict:
    """자리에만 딸린 값을 미리 잰다 — 식물을 바꿔 놓아도 안 변하는 것들.

    최적화는 배치를 수천 번 채점하는데, 조도·자리 간 거리·팬 축 투영은 자리가
    안 움직이는 한 그대로다. 매번 다시 재면 50개 배치에 7초가 걸린다.
    """
    fans = DEFAULT_FANS if fans is None else fans
    n = len(spots)
    raw = [illuminance(s["x_cm"], s["z_cm"], lights) for s in spots]
    brightest = max(raw, default=0.0) or 1.0
    lit = [v / brightest for v in raw]                 # 0~1, 제일 밝은 자리가 1

    dist = [[math.dist((spots[i]["x_cm"], spots[i]["z_cm"]),
                       (spots[j]["x_cm"], spots[j]["z_cm"]))
             for j in range(n)] for i in range(n)]

    # 팬마다: 막힘 없을 때 바람 세기와, 가로막을 수 있는 자리의 축 이탈 거리
    wind = []
    for fan in fans:
        strength, offs = [], []
        wlen = math.hypot(fan.get("dx", 0.0), fan.get("dz", 1.0))
        for s in spots:
            to_x, to_z = s["x_cm"] - fan["x"], s["z_cm"] - fan["z"]
            d_cm = math.hypot(to_x, to_z)
            if d_cm <= 0.01 or wlen <= 0:
                strength.append(fan.get("power", 1.0) if wlen > 0 else 0.0)
                offs.append({})
                continue
            cos = (to_x * fan.get("dx", 0.0) + to_z * fan.get("dz", 1.0)) / (d_cm * wlen)
            if cos <= 0:
                strength.append(0.0); offs.append({})
                continue
            strength.append(fan.get("power", 1.0) * cos * (0.5 ** (d_cm / AIR_HALF_CM)))
            ux, uz = to_x / d_cm, to_z / d_cm
            here = {}
            for j, b in enumerate(spots):
                bx, bz = b["x_cm"] - fan["x"], b["z_cm"] - fan["z"]
                along = bx * ux + bz * uz
                if 0 < along < d_cm:
                    here[j] = abs(bx * (-uz) + bz * ux)
            offs.append(here)
        wind.append({"strength": strength, "offs": offs})
    return {"n": n, "lit": lit, "dist": dist, "wind": wind}


def _measure(geo: dict, radii: List[float], heights: List[float]) -> tuple:
    """미리 잰 자리값 + 지금 놓인 식물 몸집 → (받은 빛, 바람) 목록."""
    n = geo["n"]
    got_light, got_air = [], []
    for i in range(n):
        r_cm, h_cm, mine = radii[i], heights[i], math.pi * radii[i] ** 2
        covered = 0.0
        if mine > 0:
            for j in range(n):
                if j == i or heights[j] <= h_cm:
                    continue
                overlap = _circle_overlap(r_cm, radii[j], geo["dist"][i][j])
                if overlap:
                    covered += overlap * min(1.0, (heights[j] - h_cm) / 20.0)
        shade = min(1.0, covered / mine) if mine > 0 else 0.0
        got_light.append((geo["lit"][i] * (1 - shade), shade))

        best = 0.0
        for fan in geo["wind"]:
            strength = fan["strength"][i]
            if strength <= 0:
                continue
            blocked = 0.0
            for j, off in fan["offs"][i].items():
                if off < radii[j]:
                    blocked += (1 - off / radii[j]) * min(1.0, radii[j] / 20.0)
            best = max(best, strength * max(0.0, 1 - min(0.85, blocked)))
        got_air.append(best)
    return got_light, got_air


def score_layout(spots: List[dict], lights=None, fans=None, geo: dict = None) -> dict:
    """배치 하나를 채점. spots = [{x_cm, z_cm, r_cm, h_cm, plant}] .

    빛 점수는 '받은 빛 / 필요한 빛'을 1에서 자른 값이다. 넘치게 받아도 더 좋아지지
    않는다 — 알로카시아는 직사광에 잎이 타므로 과잉은 이득이 아니다.
    """
    if not spots:
        return {"score": 0.0, "spots": []}
    geo = geo or geometry(spots, lights, fans)
    radii = [s["r_cm"] for s in spots]
    heights = [s["h_cm"] for s in spots]
    lit, air = _measure(geo, radii, heights)

    out, totals = [], []
    for i, s in enumerate(spots):
        got_light, shade = lit[i]
        wind = air[i]
        need_l = _need_light(s["plant"], s["r_cm"])
        need_a = _need_air(s["plant"])
        light_ok = min(1.0, got_light / need_l) if need_l else 1.0
        air_ok = min(1.0, wind / need_a) if need_a else 1.0
        total = LIGHT_WEIGHT * light_ok + (1 - LIGHT_WEIGHT) * air_ok
        totals.append(total)
        out.append({"slot": s["slot"], "plant_id": s["plant"].get("id"),
                    "name": s["plant"].get("name"),
                    "light": round(got_light * 100), "shade": round(shade * 100),
                    "air": round(wind * 100), "score": round(total * 100),
                    "need_light": round(need_l * 100), "need_air": round(need_a * 100)})
    return {"score": round(sum(totals) / len(totals) * 100, 1), "spots": out}


def build_spots(plants: List[dict], pot_xz_cm) -> List[dict]:
    """등록된 식물 → 채점용 자리 목록. pot_xz_cm(slot) 이 실제 좌표를 준다."""
    spots = []
    for p in plants:
        xz = pot_xz_cm(p.get("pos"))
        if xz is None:
            continue
        r_cm, h_cm = plant_shape(p)
        spots.append({"slot": p.get("pos"), "x_cm": xz[0], "z_cm": xz[1],
                      "r_cm": r_cm, "h_cm": h_cm, "plant": p})
    return spots


def optimize(spots: List[dict], lights=None, fans=None, rounds: int = 40) -> dict:
    """식물끼리 자리를 바꿔 가며 전체 점수를 올린다.

    자리는 못 옮기고 식물만 옮길 수 있으니, 답은 '식물의 순열'이다. 그늘이
    이웃에 걸려 있어서 자리마다 따로 최적을 고를 수 없다(이차 배정 문제).
    개체 수가 수십 개라 둘씩 바꿔 보는 언덕오르기로 충분하다.

    돌려주는 건 '무엇과 무엇을 바꾸라'는 목록이다 — 화분은 사람이 직접 옮긴다.
    """
    if len(spots) < 2:
        graded = score_layout(spots, lights, fans)
        return {"before": graded["score"], "after": graded["score"],
                "gain": 0.0, "moves": [], "cycles": [], "layout": graded["spots"]}

    n = len(spots)
    geo = geometry(spots, lights, fans)
    order = list(range(n))                   # order[i] = i번 자리에 놓인 식물의 원래 번호
    plants = [s["plant"] for s in spots]
    shapes = [(s["r_cm"], s["h_cm"]) for s in spots]
    # 필요한 빛/바람은 식물에 딸린 값이라 자리를 옮겨도 그대로다
    needs = [(_need_light(plants[k], shapes[k][0]), _need_air(plants[k])) for k in range(n)]

    def rate(order):
        radii = [shapes[order[i]][0] for i in range(n)]
        heights = [shapes[order[i]][1] for i in range(n)]
        lit, air = _measure(geo, radii, heights)
        total = 0.0
        for i in range(n):
            need_l, need_a = needs[order[i]]
            total += (LIGHT_WEIGHT * min(1.0, lit[i][0] / need_l)
                      + (1 - LIGHT_WEIGHT) * min(1.0, air[i] / need_a))
        return round(total / n * 100, 1)

    def laid_out(order):
        return [{**spots[i], "plant": plants[order[i]],
                 "r_cm": shapes[order[i]][0], "h_cm": shapes[order[i]][1]}
                for i in range(n)]

    before = rate(order)
    best = before
    for _ in range(rounds):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                order[i], order[j] = order[j], order[i]
                got = rate(order)
                if got > best + 1e-9:
                    best, improved = got, True
                else:
                    order[i], order[j] = order[j], order[i]
        if not improved:
            break

    # order[i] = i번 자리에 놓인 식물의 원래 번호 → dest[p] = p번 식물이 갈 자리
    dest = [0] * n
    for i, src in enumerate(order):
        dest[src] = i

    moves = [{"plant_id": plants[p].get("id"), "name": plants[p].get("name"),
              "from": spots[p]["slot"], "to": spots[dest[p]]["slot"]}
             for p in range(n) if dest[p] != p]

    # 둘씩 맞바꾸는 걸로 안 끝나는 경우가 있다 — A→B, B→C, C→A 처럼 돌아가는 고리다.
    # 고리째 보여 줘야 사람이 순서대로 옮길 수 있다.
    cycles, seen = [], set()
    for p in range(n):
        if p in seen or dest[p] == p:
            continue
        ring, cur = [], p
        while cur not in seen:
            seen.add(cur)
            ring.append(spots[cur]["slot"])
            cur = dest[cur]
        if len(ring) > 1:
            cycles.append(ring)

    return {"before": before, "after": best, "gain": round(best - before, 1),
            "moves": moves, "cycles": cycles,
            "layout": score_layout(laid_out(order), lights, fans, geo)["spots"]}


def heatmap(w_cm: float, d_cm: float, cols: int, rows: int,
            lights=None, fans=None, blockers=None) -> dict:
    """선반을 격자로 훑어 빛·바람 세기를 재 온다. 3D 바닥에 깔아 보여 준다."""
    light_grid, air_grid = [], []
    for r in range(rows):
        z_cm = -d_cm / 2 + d_cm * (r + 0.5) / rows
        lrow, arow = [], []
        for c in range(cols):
            x_cm = -w_cm / 2 + w_cm * (c + 0.5) / cols
            lrow.append(illuminance(x_cm, z_cm, lights))
            arow.append(airflow(x_cm, z_cm, fans, blockers))
        light_grid.append(lrow); air_grid.append(arow)
    top = max((v for row in light_grid for v in row), default=1.0) or 1.0
    return {"cols": cols, "rows": rows,
            "light": [[round(v / top, 3) for v in row] for row in light_grid],
            "air": [[round(v, 3) for v in row] for row in air_grid]}
