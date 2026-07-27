"""배치 최적화 스모크 테스트 (네트워크 불필요)

실행:  python3 test_placement.py

'어느 화분에 어느 식물을 두면 잘 자라나'를 계산하는 부분. 물리가 그럴듯한지
(조명에 가까우면 밝다, 큰 이웃이 그늘을 만든다, 팬 뒤쪽은 바람이 없다)와
자리를 바꿔서 점수가 실제로 올라가는지를 확인한다.
"""

import asyncio
import json

from fastapi import HTTPException

import os
os.environ["FARM_DB"] = ""      # 테스트는 파일에 저장하지 않는다

import main
import placement


def _plant(pid, pos, grade="중품", leaf_cm=None, density=0):
    return {"id": pid, "name": f"식물 {pos}", "pos": pos, "size_class": grade,
            "leaf_max_cm": leaf_cm, "overlap_density": density,
            "leaf_count": 5, "shoot_count": 1, "mature_count": 3, "old_count": 1}


def _spot(slot, x_cm, z_cm, plant=None, r_cm=13.0, h_cm=28.0):
    return {"slot": slot, "x_cm": x_cm, "z_cm": z_cm, "r_cm": r_cm, "h_cm": h_cm,
            "plant": plant or _plant(slot, slot)}


def _reset():
    main.POTS.clear(); main.PLANTS.clear(); main.FEATS.clear()
    main.LEAVES.clear(); main.LEAF_FIXES.clear(); main.ENVIRONMENT.clear()


# ── 빛 ───────────────────────────────────────────────────────────────────
def test_under_the_lamp_is_brighter_than_the_corner():
    lamp = [{"x": 0.0, "y": 40.0, "z": 0.0, "power": 1.0}]
    middle = placement.illuminance(0, 0, lamp)
    corner = placement.illuminance(-28, -18, lamp)
    assert middle > corner * 1.5, (middle, corner)


def test_light_falls_off_with_distance_squared():
    """거리가 2배면 밝기는 대략 1/4 이하 (입사각까지 더 나빠진다)."""
    lamp = [{"x": 0.0, "y": 20.0, "z": 0.0, "power": 1.0}]
    near = placement.illuminance(0, 0, lamp, canopy_y_cm=0)      # 20cm 아래
    far = placement.illuminance(0, 0, lamp, canopy_y_cm=-20)     # 40cm 아래
    assert far < near / 4 * 1.05, (near, far)


def test_two_lamps_beat_one():
    one = [{"x": 0.0, "y": 40.0, "z": 0.0}]
    two = one + [{"x": 10.0, "y": 40.0, "z": 0.0}]
    assert placement.illuminance(5, 0, two) > placement.illuminance(5, 0, one)


# ── 그늘 ─────────────────────────────────────────────────────────────────
def test_a_big_neighbour_casts_shade():
    small = _spot("C3", 0, 0, r_cm=8, h_cm=15)
    big = _spot("C4", 6, 0, r_cm=20, h_cm=45)
    assert placement.shade_factor(small, [small, big]) > 0.3
    # 반대로 큰 쪽은 작은 이웃 때문에 어두워지지 않는다
    assert placement.shade_factor(big, [small, big]) == 0.0


def test_same_height_plants_do_not_shade_each_other():
    """옆으로 겹칠 뿐 위를 덮지는 않는다."""
    a = _spot("C3", 0, 0, r_cm=15, h_cm=30)
    b = _spot("C4", 5, 0, r_cm=15, h_cm=30)
    assert placement.shade_factor(a, [a, b]) == 0.0
    assert placement.shade_factor(b, [a, b]) == 0.0


def test_a_far_away_giant_casts_no_shade():
    small = _spot("A1", -28, -18, r_cm=8, h_cm=15)
    big = _spot("E10", 28, 18, r_cm=20, h_cm=45)
    assert placement.shade_factor(small, [small, big]) == 0.0


def test_shade_never_exceeds_everything():
    small = _spot("C3", 0, 0, r_cm=6, h_cm=12)
    crowd = [small] + [_spot(f"C{i}", 0, 0, r_cm=25, h_cm=50) for i in range(4)]
    assert placement.shade_factor(small, crowd) == 1.0


# ── 바람 ─────────────────────────────────────────────────────────────────
def test_behind_the_fan_there_is_no_wind():
    fan = [{"x": 0.0, "y": 25.0, "z": -28.0, "dx": 0.0, "dz": 1.0}]
    downwind = placement.airflow(0, 0, fan)
    upwind = placement.airflow(0, -35, fan)      # 팬 뒤쪽
    assert downwind > 0 and upwind == 0.0, (downwind, upwind)


def test_wind_weakens_with_distance():
    fan = [{"x": 0.0, "y": 25.0, "z": -28.0, "dx": 0.0, "dz": 1.0}]
    assert placement.airflow(0, -10, fan) > placement.airflow(0, 18, fan)


def test_a_big_plant_upwind_blocks_the_wind():
    """바람 위쪽에 대품이 서 있으면 뒤쪽은 습기가 안 빠진다."""
    fan = [{"x": 0.0, "y": 25.0, "z": -28.0, "dx": 0.0, "dz": 1.0}]
    blocker = [_spot("C5", 0, -5, r_cm=22, h_cm=48)]
    clear = placement.airflow(0, 10, fan)
    behind = placement.airflow(0, 10, fan, blocker)
    assert behind < clear * 0.8, (clear, behind)
    # 축에서 옆으로 비켜날수록 덜 막힌다
    def loss(x_cm):
        return 1 - placement.airflow(x_cm, 10, fan, blocker) / placement.airflow(x_cm, 10, fan)
    assert loss(0) > loss(14) > loss(28), [round(loss(v), 3) for v in (0, 14, 28)]
    # 잎우산 밖으로 완전히 벗어난 길은 그대로다
    narrow = [_spot("C5", -20, -5, r_cm=8, h_cm=20)]
    assert placement.airflow(20, 10, fan, narrow) == placement.airflow(20, 10, fan)


# ── 채점 ─────────────────────────────────────────────────────────────────
def test_score_reports_every_spot():
    spots = [_spot("C3", -10, 0), _spot("C7", 10, 0)]
    got = placement.score_layout(spots)
    assert 0 <= got["score"] <= 100
    assert {s["slot"] for s in got["spots"]} == {"C3", "C7"}
    for s in got["spots"]:
        for key in ("light", "shade", "air", "score", "need_light", "need_air"):
            assert key in s, (key, s)


def test_a_dense_plant_needs_more_air():
    airy = placement._need_air(_plant("p1", "C3", density=0))
    packed = placement._need_air(_plant("p2", "C4", density=90))
    assert packed > airy


def test_a_big_plant_needs_more_light():
    assert placement._need_light({}, 20.0) > placement._need_light({}, 7.0)


def test_measured_leaf_beats_the_grade():
    """실측 잎 길이가 있으면 등급 기본값 대신 그걸 쓴다."""
    r_graded, _ = placement.plant_shape(_plant("p1", "C3", "소품"))
    r_measured, _ = placement.plant_shape(_plant("p2", "C4", "소품", leaf_cm=24))
    assert r_measured > r_graded * 2, (r_graded, r_measured)


def test_empty_farm_scores_zero_without_crashing():
    assert placement.score_layout([])["score"] == 0.0


# ── 최적화 ───────────────────────────────────────────────────────────────
def test_swapping_a_shaded_small_plant_helps():
    """어두운 구석의 대품과 밝은 자리의 소품을 바꾸면 점수가 올라야 한다."""
    lamp = [{"x": -25.0, "y": 40.0, "z": 0.0}]
    big = _plant("big", "C2", "대품", leaf_cm=22)
    small = _plant("small", "C9", "소품", leaf_cm=6)
    spots = [_spot("C2", 25, 0, small, r_cm=6, h_cm=13),    # 밝은 자리에 소품
             _spot("C9", -25, 0, big, r_cm=22, h_cm=48)]    # 어두운 자리에 대품
    res = placement.optimize(spots, lamp)
    assert res["after"] >= res["before"], res
    if res["moves"]:
        assert res["gain"] > 0, res


def test_every_plant_gets_exactly_one_new_home():
    """제안이 '한 자리에 둘'이 되면 사람이 실행할 수 없다."""
    lamp = [{"x": -25.0, "y": 40.0, "z": 0.0}]
    spots = [_spot("C1", -25, 0, _plant("a", "C1", "소품", leaf_cm=6), 6, 13),
             _spot("C5", 0, 0, _plant("b", "C5", "중품", leaf_cm=13), 13, 28),
             _spot("C9", 25, 0, _plant("c", "C9", "대품", leaf_cm=22), 22, 48)]
    res = placement.optimize(spots, lamp)
    slots = [s["slot"] for s in spots]
    froms = [m["from"] for m in res["moves"]]
    tos = [m["to"] for m in res["moves"]]
    assert len(set(froms)) == len(froms), froms
    assert len(set(tos)) == len(tos), tos
    assert set(froms) == set(tos), (froms, tos)      # 자리를 서로 채워 준다
    for mv in res["moves"]:
        assert mv["from"] in slots and mv["to"] in slots and mv["from"] != mv["to"], mv


def test_a_three_way_rotation_is_reported_as_a_ring():
    """둘씩 맞바꿔서 안 끝나는 배치는 '고리'로 알려 줘야 한다."""
    lamp = [{"x": -28.0, "y": 40.0, "z": -18.0}]
    fan = [{"x": 28.0, "y": 25.0, "z": 18.0, "dx": -1.0, "dz": -1.0}]
    grades = ["대품", "소품", "중품", "대품", "소품", "중품"]
    spots = [_spot(f"C{i+1}", -25 + i * 10, (i % 2) * 12 - 6,
                   _plant(f"p{i}", f"C{i+1}", g, density=i * 15),
                   *placement.GRADE_SHAPE[g])
             for i, g in enumerate(grades)]
    res = placement.optimize(spots, lamp, fan)
    moved = {m["from"] for m in res["moves"]}
    # 고리에 든 자리를 다 합치면 움직이는 자리 전체와 같아야 한다
    assert {s for ring in res["cycles"] for s in ring} == moved, (res["cycles"], moved)
    for ring in res["cycles"]:
        assert len(ring) == len(set(ring)) >= 2, ring


def test_one_plant_has_nothing_to_swap():
    res = placement.optimize([_spot("C3", 0, 0)])
    assert res["moves"] == [] and res["cycles"] == []
    assert res["before"] == res["after"]


def test_optimize_never_makes_it_worse():
    lamp = [{"x": -20.0, "y": 40.0, "z": 10.0}]
    grades = ["소품", "중품", "대품", "중품", "소품"]
    spots = [_spot(f"C{i+1}", -24 + i * 12, 0, _plant(f"p{i}", f"C{i+1}", g))
             for i, g in enumerate(grades)]
    res = placement.optimize(spots, lamp)
    assert res["after"] >= res["before"] - 1e-9, res


# ── 히트맵 ───────────────────────────────────────────────────────────────
def test_heatmap_covers_the_shelf():
    hm = placement.heatmap(60, 40, 10, 7)
    assert hm["cols"] == 10 and hm["rows"] == 7
    assert len(hm["light"]) == 7 and len(hm["light"][0]) == 10
    assert max(v for row in hm["light"] for v in row) == 1.0   # 제일 밝은 곳이 1
    assert all(0 <= v <= 1 for row in hm["air"] for v in row)


# ── 엔드포인트 ───────────────────────────────────────────────────────────
def test_environment_defaults_then_takes_real_positions():
    _reset()
    assert main.get_environment()["custom"] is False
    asyncio.run(main.set_environment(
        lights=json.dumps([{"x": 0, "y": 55, "z": 0, "power": 2}]), fans=None))
    env = main.get_environment()
    assert env["custom"] is True and env["lights"][0]["y"] == 55
    assert env["fans"] == placement.DEFAULT_FANS      # 안 준 쪽은 기본값 유지
    _reset()


def test_bad_environment_is_refused():
    _reset()
    for bad in ("{oops", json.dumps({"x": 1}), json.dumps([{"y": 40}])):
        try:
            asyncio.run(main.set_environment(lights=bad, fans=None))
        except HTTPException as e:
            assert e.status_code == 400
        else:
            raise AssertionError(f"거부해야 한다: {bad}")
    _reset()


def _farm(grades):
    """화분을 찍고 그 자리에 식물을 앉힌다."""
    _reset()
    pts = [[0.1 + 0.16 * i, 0.5] for i in range(len(grades))]
    asyncio.run(main.set_pots(points=json.dumps(pts), points_px=None, corners=None))
    for i, (pot, grade) in enumerate(zip(main.POTS, grades)):
        xz = main._pot_xz_cm(pot["slot"])
        main.PLANTS[f"p{i}"] = {**_plant(f"p{i}", pot["slot"], grade), "x": xz[0], "z": xz[1]}


def test_placement_endpoint_grades_the_current_layout():
    _farm(["대품", "소품", "중품"])
    got = main.get_placement()
    assert 0 <= got["score"] <= 100
    assert len(got["spots"]) == 3
    assert len(got["worst"]) == 3 and got["worst"][0]["score"] <= got["worst"][-1]["score"]
    _reset()


def test_optimize_endpoint_suggests_without_moving_anything():
    """제안만 한다. 화분은 사람이 옮기기 전까지 기록이 안 바뀌어야 한다."""
    _farm(["소품", "중품", "대품"])
    before = {pid: p["pos"] for pid, p in main.PLANTS.items()}
    got = main.optimize_placement()
    assert "moves" in got and "cycles" in got, got
    after = {pid: p["pos"] for pid, p in main.PLANTS.items()}
    assert before == after, (before, after)
    _reset()


def _swap_moves(a, b):
    return json.dumps([{"plant_id": a["id"], "from": a["pos"], "to": b["pos"]},
                       {"plant_id": b["id"], "from": b["pos"], "to": a["pos"]}])


def test_apply_swaps_the_records_and_the_coordinates():
    _farm(["소품", "대품"])
    a, b = (main.PLANTS["p0"], main.PLANTS["p1"])
    slot_a, slot_b = a["pos"], b["pos"]
    x_a = a["x"]
    res = asyncio.run(main.apply_placement(moves=_swap_moves(a, b)))
    assert res["applied"] == 2
    assert a["pos"] == slot_b and b["pos"] == slot_a
    assert a["x"] != x_a, "좌표가 자리를 따라가야 한다"
    _reset()


def test_apply_walks_a_three_way_ring():
    """A→B, B→C, C→A. 둘씩 바꾸는 게 아니라 고리째 돌아간다."""
    _farm(["소품", "중품", "대품"])
    a, b, c = main.PLANTS["p0"], main.PLANTS["p1"], main.PLANTS["p2"]
    sa, sb, sc = a["pos"], b["pos"], c["pos"]
    res = asyncio.run(main.apply_placement(moves=json.dumps([
        {"plant_id": a["id"], "from": sa, "to": sb},
        {"plant_id": b["id"], "from": sb, "to": sc},
        {"plant_id": c["id"], "from": sc, "to": sa}])))
    assert res["applied"] == 3
    assert (a["pos"], b["pos"], c["pos"]) == (sb, sc, sa)
    assert len({a["pos"], b["pos"], c["pos"]}) == 3, "두 식물이 한 자리에 앉으면 안 된다"
    _reset()


def test_apply_refuses_a_plan_that_stacks_two_plants():
    _farm(["소품", "중품", "대품"])
    a, b, c = main.PLANTS["p0"], main.PLANTS["p1"], main.PLANTS["p2"]
    try:
        asyncio.run(main.apply_placement(moves=json.dumps([
            {"plant_id": a["id"], "from": a["pos"], "to": c["pos"]},
            {"plant_id": b["id"], "from": b["pos"], "to": c["pos"]}])))
    except HTTPException as e:
        assert e.status_code == 400
    else:
        raise AssertionError("한 자리에 둘은 거부해야 한다")
    _reset()


def test_apply_refuses_moving_onto_a_plant_that_stays_put():
    _farm(["소품", "중품", "대품"])
    a, c = main.PLANTS["p0"], main.PLANTS["p2"]
    try:
        asyncio.run(main.apply_placement(moves=json.dumps([
            {"plant_id": a["id"], "from": a["pos"], "to": c["pos"]}])))
    except HTTPException as e:
        assert e.status_code == 400
    else:
        raise AssertionError("가만히 있는 식물 자리로는 못 간다")
    _reset()


def test_apply_moves_the_leaf_records_too():
    _farm(["소품", "대품"])
    a, b = main.PLANTS["p0"], main.PLANTS["p1"]
    main.LEAVES["lf_1"] = {"leaf_id": "lf_1", "plant_id": "p0", "pot_slot": a["pos"],
                           "stage": "mature", "centroid_uv": [0.1, 0.5],
                           "ambiguous": False, "manual": False,
                           "assign": {"nearest": a["pos"], "second": b["pos"], "margin": 0.4}}
    main.LEAF_FIXES.append({"u": 0.1, "v": 0.5, "pot_slot": a["pos"]})
    slot_a, slot_b = a["pos"], b["pos"]
    asyncio.run(main.apply_placement(moves=_swap_moves(a, b)))
    assert main.LEAVES["lf_1"]["pot_slot"] == slot_b
    assert main.LEAF_FIXES == [], "식물이 옮겨졌으니 좌표에 묶인 보정은 무효다"
    _reset()


def test_apply_ignores_slots_with_no_plant():
    _farm(["소품", "대품"])
    res = asyncio.run(main.apply_placement(
        moves=json.dumps([{"plant_id": "nope", "from": "E10", "to": "E9"}])))
    assert res["applied"] == 0
    _reset()


def test_apply_refuses_junk():
    _reset()
    for bad in ("{oops", json.dumps({"from": "A1"})):
        try:
            asyncio.run(main.apply_placement(moves=bad))
        except HTTPException as e:
            assert e.status_code == 400
        else:
            raise AssertionError(f"거부해야 한다: {bad}")
    _reset()


def test_heatmap_endpoint_clamps_silly_sizes():
    _reset()
    assert main.get_heatmap(cols=999, rows=1, occupied=None)["cols"] == 60
    assert main.get_heatmap(cols=1, rows=999, occupied=None)["rows"] == 40
    _reset()


def test_heatmap_can_include_the_plants_standing_there():
    _farm(["대품", "대품", "대품"])
    plain = main.get_heatmap(cols=10, rows=7, occupied=None)
    blocked = main.get_heatmap(cols=10, rows=7, occupied="1")
    total = lambda hm: sum(v for row in hm["air"] for v in row)
    assert total(blocked) < total(plain), (total(plain), total(blocked))
    _reset()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✔ {t.__name__}")
    print(f"\n{len(tests)}개 통과")
