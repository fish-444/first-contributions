"""잎 → 식물 그룹화 스모크 테스트 (네트워크 불필요)

실행:  python3 test_grouping.py

탑뷰 사진 한 장에 여러 화분이 들어올 때, 어느 잎이 어느 식물인지 묶는 로직을 검증한다.
"""

from main import (analyze_metrics, analyze_top, group_by_pots_and_shape, group_leaves,
                  leaf_features, shape_similarity, _label_shape_groups)


def leaves_only(boxes):
    """화분 박스를 빼고 잎만 — '화분이 없을 때'를 흉내 내 비교용으로 쓴다."""
    return [b for b in boxes if b["cls"] != "pot"]


def box(cx, cy, w, h, cls="leaf"):
    return {"cls": cls, "conf": 0.9,
            "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2,
            "area": w * h}


def feat(aspect=1.0, size=100.0, h=60.0, s=180.0, v=120.0, color=True):
    return {"aspect": aspect, "size": size, "h": h, "s": s, "v": v, "has_color": color}


def test_two_separated_plants_split_by_distance():
    """멀리 떨어진 두 무리는 두 식물로 갈라져야 한다."""
    leaves = [box(100, 100, 60, 60), box(140, 130, 60, 60),      # 왼쪽 위
              box(900, 900, 60, 60), box(940, 930, 60, 60)]      # 오른쪽 아래
    groups = group_leaves(leaves)
    assert len(groups) == 2, [len(g) for g in groups]
    assert sorted(len(g) for g in groups) == [2, 2]


def test_touching_leaves_merge():
    """맞닿은 잎은 같은 식물로 묶인다."""
    leaves = [box(100, 100, 60, 60), box(130, 100, 60, 60), box(115, 130, 60, 60)]
    assert len(group_leaves(leaves)) == 1


def test_single_leaf_is_its_own_plant():
    assert len(group_leaves([box(50, 50, 30, 30)])) == 1


def test_no_leaves():
    assert group_leaves([]) == []


def test_pot_grouping_beats_distance_when_canopies_touch():
    """캐노피가 겹쳐도 화분이 있으면 정확히 갈린다 — 화분 방식을 쓰는 이유."""
    # 두 화분이 가깝고, 안쪽 잎끼리 실제로 겹쳐 있는 상황
    pots = [box(100, 100, 80, 80, cls="pot"), box(220, 100, 80, 80, cls="pot")]
    leaves = [box(100, 100, 70, 70), box(155, 100, 70, 70),     # 1번 화분 (오른쪽으로 뻗음)
              box(170, 100, 70, 70), box(220, 100, 70, 70)]     # 2번 화분 (왼쪽으로 뻗음)

    # 화분을 무시하고 잎만 보면 하나로 뭉친다
    assert len(group_leaves(leaves)) == 1

    # 화분을 쓰면 둘로 갈린다
    groups = group_leaves(pots + leaves)
    assert len(groups) == 2, [len(g) for g in groups]
    assert sorted(len(g) for g in groups) == [2, 2]


def test_group_leaves_prefers_pots_automatically():
    """group_leaves 는 화분이 잡히면 알아서 화분 기준으로 간다."""
    boxes = [box(100, 100, 80, 80, cls="pot"), box(220, 100, 80, 80, cls="pot"),
             box(90, 100, 70, 70), box(110, 110, 70, 70),
             box(215, 100, 70, 70), box(235, 110, 70, 70)]
    assert len(group_leaves(boxes)) == 2


def test_group_leaves_falls_back_to_distance_without_pots():
    boxes = [box(100, 100, 60, 60), box(900, 900, 60, 60)]
    assert len(group_leaves(boxes)) == 2


def test_pots_never_counted_as_leaves():
    """화분 박스가 잎 개수에 섞이면 안 된다."""
    boxes = [box(100, 100, 80, 80, cls="pot"), box(100, 100, 60, 60, cls="mature leaf")]
    groups = group_leaves(boxes)
    assert len(groups) == 1 and len(groups[0]) == 1
    assert analyze_metrics(groups[0], 10000)["leaf_count"] == 1


def test_leaf_outside_any_pot_joins_nearest():
    """화분 밖으로 뻗은 잎도 가장 가까운 화분에 붙는다 (알로카시아는 잎이 멀리 뻗는다)."""
    pots = [box(100, 100, 40, 40, cls="pot"), box(500, 100, 40, 40, cls="pot")]
    leaves = [box(160, 100, 60, 60)]        # 어느 화분 안에도 없음, 1번에 더 가까움
    groups = group_leaves(pots + leaves)
    assert len(groups) == 1 and len(groups[0]) == 1


def test_ref_area_keeps_leaf_size_meaningful_in_farm_photo():
    """농장 전체 사진에서도 대/중/소엽이 뭉개지면 안 된다."""
    farm_area = 4000 * 3000
    big_leaf = [box(500, 500, 700, 700)]     # 큰 잎이지만 전체 사진 대비로는 4%

    # 사진 전체를 기준으로 재면 무조건 소엽으로 뭉개진다
    assert analyze_top(big_leaf, farm_area)["top_leaf_size"] == "소엽"

    # 식물 1개 몫(10개체 가정)을 기준으로 재면 제대로 큰 잎으로 잡힌다
    per_plant = farm_area / 10
    assert analyze_top(big_leaf, farm_area, ref_area=per_plant)["top_leaf_size"] == "대엽"


def test_each_group_gets_its_own_stage_counts():
    """무리별로 단계 집계가 따로 나와야 3D 색이 개체별로 달라진다."""
    boxes = [box(100, 100, 60, 60, cls="old leaf"), box(140, 100, 60, 60, cls="old leaf"),
             box(900, 900, 60, 60, cls="shoot"), box(940, 900, 60, 60, cls="mature leaf")]
    groups = sorted(group_leaves(boxes), key=lambda g: g[0]["x1"])
    a = analyze_metrics(groups[0], 10 ** 6)
    b = analyze_metrics(groups[1], 10 ** 6)
    assert (a["old_count"], a["shoot_count"]) == (2, 0), a
    assert (b["shoot_count"], b["mature_count"]) == (1, 1), b


def test_shape_similarity_basics():
    """같은 잎끼리는 1에 가깝고, 모양·색이 다르면 뚝 떨어진다."""
    a = feat(aspect=1.2, size=100, h=60, s=180, v=120)
    assert shape_similarity(a, a) > 0.99

    long_narrow = feat(aspect=3.0, size=100, h=60, s=180, v=120)
    assert shape_similarity(a, long_narrow) < 0.4, shape_similarity(a, long_narrow)

    much_bigger = feat(aspect=1.2, size=400, h=60, s=180, v=120)
    assert shape_similarity(a, much_bigger) < 0.4

    different_colour = feat(aspect=1.2, size=100, h=110, s=180, v=120)
    assert shape_similarity(a, different_colour) < 0.4


def test_shape_similarity_ignores_missing_colour():
    """색 정보가 없으면 모양·크기만으로 판단하고, 그래도 동작해야 한다."""
    a = feat(aspect=1.2, size=100, color=False)
    b = feat(aspect=1.25, size=105, color=False)
    assert shape_similarity(a, b) > 0.9


def test_leaf_features_survive_degenerate_boxes():
    """폭이 0인 박스 같은 쓰레기 입력에도 죽지 않는다."""
    bad = {"cls": "leaf", "x1": 10, "y1": 10, "x2": 10, "y2": 50, "area": 0}
    f = leaf_features(None, bad)
    assert f["aspect"] == 1.0 and f["has_color"] is False


def test_shape_groups_label_varieties():
    """생김새가 닮은 개체끼리 같은 딱지를, 다른 개체는 다른 딱지를 받는다."""
    items = [{"name": "둥근잎1"}, {"name": "둥근잎2"}, {"name": "길쭉잎"}]
    feats = [feat(aspect=1.2, size=100), feat(aspect=1.25, size=104),
             feat(aspect=3.2, size=100)]
    _label_shape_groups(items, feats)
    assert items[0]["shape_group"] == items[1]["shape_group"]
    assert items[2]["shape_group"] != items[0]["shape_group"]
    # 큰 무리가 A 를 받는다
    assert items[0]["shape_group"] == "A" and items[0]["shape_group_size"] == 2


def test_shape_helps_a_stray_leaf_find_its_own_pot():
    """화분 밖으로 뻗은 잎: 거리만 보면 옆 화분, 생김새를 보면 제 화분.

    A(화분) + B(거리) 를 합치고 생김새로 교정하는 부분의 핵심.
    """
    # 화분 박스는 x 70~130 과 370~430
    pots = [box(100, 100, 60, 60, "pot"), box(400, 100, 60, 60, "pot")]
    leaves = [box(100, 100, 40, 120),      # 1번 화분: 길쭉한 잎
              box(400, 100, 90, 90)]       # 2번 화분: 둥근 잎
    feats = [feat(aspect=3.0), feat(aspect=1.0)]

    # 어느 화분 박스에도 안 걸치는 떠돌이 잎. 2번이 4배 가깝지만(60 vs 240) 생김새는 길쭉 = 1번 쪽
    stray = box(340, 100, 40, 120)
    groups = group_by_pots_and_shape(leaves + [stray], pots, feats + [feat(aspect=3.0)])
    assert len(groups) == 2, [len(g) for g in groups]

    owner = next(g for g in groups if any(b is stray for b in g))
    assert any(b is leaves[0] for b in owner), "생김새가 같은 1번 화분에 붙어야 한다"
    assert not any(b is leaves[1] for b in owner), "가깝다는 이유로 2번에 붙으면 안 된다"


def test_shape_gate_stops_unlike_leaves_merging():
    """가까워도 생김새가 전혀 다르면 (화분 없을 때) 같은 식물로 묶지 않는다."""
    round_leaf = box(100, 100, 90, 90)
    long_leaf = box(150, 100, 30, 150)
    groups = group_leaves([round_leaf, long_leaf])
    assert len(groups) == 2, [len(g) for g in groups]


def _farm_boxes():
    """실제 농장 탑뷰와 비슷한 배치: 화분 3개가 한 줄, 잎은 사방으로 뻗어 이웃과 겹침."""
    boxes = []
    for px in (400, 1000, 1600):
        boxes.append(box(px, 900, 260, 260, "pot"))
        for dx, dy, cls in [(-260, -120, "leaf"), (250, -100, "leaf"), (0, -300, "shoot"),
                            (-150, 180, "old leaf"), (170, 200, "leaf")]:
            boxes.append(box(px + dx, 900 + dy, 300, 300, cls))
    return boxes


def test_radiating_canopy_defeats_distance_grouping():
    """알로카시아 탑뷰의 핵심 성질: '이웃 식물의 잎'이 '같은 식물의 잎'보다 가깝다.

    잎이 잎자루로 사방으로 뻗기 때문에, 거리 임계값을 어떻게 잡아도
    같은 식물끼리 묶으면 이웃까지 딸려 오고, 이웃을 떼면 자기 잎도 떨어져 나간다.
    → 화분 클래스가 '있으면 좋은 것'이 아니라 '필요한 것'인 이유.
    """
    import math
    same_plant = math.dist((400 - 260, 900 - 120), (400 + 250, 900 - 100))
    neighbours = math.dist((400 + 250, 900 - 100), (1000 - 260, 900 - 120))
    assert neighbours < same_plant / 3, (neighbours, same_plant)

    boxes = _farm_boxes()
    leaves = leaves_only(boxes)
    assert len(group_leaves(leaves)) != 3            # 잎만으로는 3개가 안 나온다
    assert len(group_leaves(boxes)) == 3             # 화분을 쓰면 정확히 3개


def test_scan_endpoint_registers_one_plant_per_pot():
    """농장 사진 1장 → 화분 수만큼 개체가 자리와 함께 등록된다."""
    import asyncio, io
    from PIL import Image
    import main

    class _Upload:                                   # UploadFile 흉내 (content_type + read)
        content_type = "image/jpeg"

        def __init__(self, raw):
            self._raw = raw

        async def read(self):
            return self._raw

    buf = io.BytesIO()
    Image.new("RGB", (2000, 1500), (20, 80, 30)).save(buf, format="JPEG")

    boxes = _farm_boxes()
    orig_detect, orig_plants = main.detect_boxes, dict(main.PLANTS)
    main.detect_boxes = lambda image, model_id: (boxes, float(2000 * 1500))
    main.PLANTS.clear()
    try:
        res = asyncio.run(main.scan_farm(file=_Upload(buf.getvalue()), replace=None, mode=None))
        assert res["count"] == 3, res
        assert res["grouped_by"] == "pot", res

        positions = [p["pos"] for p in res["plants"]]
        assert len(set(positions)) == 3, positions          # 자리가 겹치면 안 된다
        for p in res["plants"]:
            assert p["leaf_count"] == 5, p
            assert p["thumb"].startswith("data:image/jpeg;base64,")

        # 같은 사진을 다시 스캔해도 개체가 늘어나지 않는다 (이름·방향 유지하며 갱신)
        main.PLANTS[list(main.PLANTS)[0]]["name"] = "내가 지은 이름"
        again = asyncio.run(main.scan_farm(file=_Upload(buf.getvalue()), replace=None, mode=None))
        assert again["count"] == 3 and len(main.PLANTS) == 3, (again["count"], len(main.PLANTS))
        assert "내가 지은 이름" in [p["name"] for p in main.PLANTS.values()]
    finally:
        main.detect_boxes = orig_detect
        main.PLANTS.clear()
        main.PLANTS.update(orig_plants)


def test_scan_labels_varieties_across_pots():
    """서로 떨어진 화분이라도 생김새가 같으면 같은 모양 그룹을 받는다."""
    import asyncio, io
    from PIL import Image
    import main

    class _Upload:
        content_type = "image/jpeg"

        def __init__(self, raw):
            self._raw = raw

        async def read(self):
            return self._raw

    # 대각선으로 마주 보는 화분끼리 같은 품종 (위치가 아니라 생김새로 묶이는지 확인)
    plan = [(300, 400, "long"), (900, 400, "round"),
            (300, 1000, "round"), (900, 1000, "long")]
    boxes = []
    for px, py, kind in plan:
        boxes.append(box(px, py, 200, 200, "pot"))
        for dx, dy in [(-180, -90), (180, -90), (0, -200)]:
            w, h = (80, 240) if kind == "long" else (190, 190)
            boxes.append(box(px + dx, py + dy, w, h))

    buf = io.BytesIO()
    Image.new("RGB", (1200, 1400), (20, 80, 30)).save(buf, format="JPEG")

    orig_detect, orig_plants, orig_feats = main.detect_boxes, dict(main.PLANTS), dict(main.FEATS)
    main.detect_boxes = lambda image, model_id: (boxes, float(1200 * 1400))
    main.PLANTS.clear(); main.FEATS.clear()
    try:
        res = asyncio.run(main.scan_farm(file=_Upload(buf.getvalue()), replace=None, mode=None))
        assert res["count"] == 4, res["count"]
        assert res["shape_groups"] == 2, res["shape_groups"]

        # 같은 품종끼리는 같은 딱지, 다른 품종끼리는 다른 딱지 → 2개씩 두 무리
        groups = {}
        for p in res["plants"]:
            groups.setdefault(p["shape_group"], []).append(p["pos"])
        assert sorted(len(v) for v in groups.values()) == [2, 2], groups
        for p in res["plants"]:
            assert p["shape_group_size"] == 2, p
    finally:
        main.detect_boxes = orig_detect
        main.PLANTS.clear(); main.PLANTS.update(orig_plants)
        main.FEATS.clear(); main.FEATS.update(orig_feats)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✔ {t.__name__}")
    print(f"\n{len(tests)}개 통과")
