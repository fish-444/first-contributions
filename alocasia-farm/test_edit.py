"""직접 수정 + 스캔 모드 스모크 테스트 (네트워크 불필요)

실행:  python3 test_edit.py

사진이 뭉개지거나 잎이 가려지면 탐지가 틀린다. 손으로 고칠 수 있는지,
그리고 '잎은 유지하고 새 잎만 기록'하는 스캔이 값을 안 깎는지 확인한다.
"""

import asyncio
import io

from PIL import Image
from fastapi import HTTPException

import os
os.environ["FARM_DB"] = ""      # 테스트는 파일에 저장하지 않는다

import main


def box(cx, cy, w, h, cls="leaf"):
    return {"cls": cls, "conf": 0.9,
            "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2,
            "area": w * h}


class _Upload:
    content_type = "image/jpeg"

    def __init__(self, raw):
        self._raw = raw

    async def read(self):
        return self._raw


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (1200, 800), (25, 90, 35)).save(buf, format="JPEG")
    return buf.getvalue()


def _reset():
    main.POTS.clear(); main.PLANTS.clear(); main.FEATS.clear()


def _plant(**kw):
    """등록된 식물 하나를 직접 만들어 둔다."""
    _reset()
    p = {"id": "t1", "name": "테스트", "pos": "C5", "x": 0, "z": 0, "rot": 0,
         "size_class": "중품", "leaf_count": 6, "shoot_count": 1,
         "mature_count": 4, "old_count": 1, "top_leaf_size": "중엽",
         "top_leaf_pct": 12.0, "overlap_count": 0, "overlap_density": 0,
         "updated": "2026-01-01 00:00:00"}
    p.update(kw)
    main.PLANTS["t1"] = p
    return p


def _patch(**kw):
    args = {"name": None, "rot": None, "size_class": None,
            "shoot_count": None, "mature_count": None, "old_count": None}
    args.update(kw)
    return asyncio.run(main.update_plant(pid="t1", **args))


# ── 직접 수정 ────────────────────────────────────────────────────────────
def test_size_class_can_be_set_by_hand():
    _plant()
    assert _patch(size_class="대품")["size_class"] == "대품"
    _reset()


def test_bad_size_class_rejected():
    _plant()
    try:
        _patch(size_class="특대품")
    except HTTPException as e:
        assert e.status_code == 400
    else:
        raise AssertionError("없는 등급은 거부해야 한다")
    _reset()


def test_leaf_counts_editable_and_total_follows():
    """잎을 더하거나 빼면 총 개수가 따라와야 한다."""
    _plant()
    p = _patch(shoot_count=3, mature_count=5, old_count=2)
    assert (p["shoot_count"], p["mature_count"], p["old_count"]) == (3, 5, 2)
    assert p["leaf_count"] == 10, p["leaf_count"]
    _reset()


def test_stage_move_is_just_two_edits():
    """성엽 → 노엽 이동: 한쪽 -1, 다른 쪽 +1. 총 개수는 그대로."""
    _plant(mature_count=4, old_count=1, leaf_count=6)
    p = _patch(mature_count=3, old_count=2)
    assert (p["mature_count"], p["old_count"]) == (3, 2)
    assert p["leaf_count"] == 6, p["leaf_count"]
    _reset()


def test_negative_leaf_count_rejected():
    _plant()
    try:
        _patch(old_count=-1)
    except HTTPException as e:
        assert e.status_code == 400
    else:
        raise AssertionError("음수는 거부해야 한다")
    _reset()


def test_hand_edit_is_marked():
    _plant()
    assert "manual" not in main.PLANTS["t1"]
    assert _patch(old_count=2)["manual"] is True
    _reset()


def test_renaming_alone_is_not_a_hand_edit():
    """이름만 바꾼 건 탐지값을 고친 게 아니다."""
    _plant()
    p = _patch(name="프라이덱")
    assert p["name"] == "프라이덱" and "manual" not in p
    _reset()


# ── 스캔 모드 ────────────────────────────────────────────────────────────
def _scan(mode, boxes):
    orig = main.detect_boxes
    main.detect_boxes = lambda im, det=None: (boxes, float(1200 * 800))
    try:
        return asyncio.run(main.scan_farm(file=_Upload(_jpeg()), replace=None, mode=mode))
    finally:
        main.detect_boxes = orig


def test_mode_names_validated():
    _reset()
    try:
        main._scan_mode("이상한모드", None)
    except HTTPException as e:
        assert e.status_code == 400
    else:
        raise AssertionError("모르는 모드는 거부해야 한다")
    assert main._scan_mode(None, None) == "update"
    assert main._scan_mode(None, "1") == "replace"     # 예전 방식 호환
    assert main._scan_mode("keep", None) == "keep"
    _reset()


def test_keep_mode_does_not_lose_hidden_leaves():
    """가려져서 덜 잡힌 사진이 기존 잎 수를 깎으면 안 된다."""
    _reset()
    main.PLANTS.clear()
    asyncio.run(main.set_pots(points='[[0.5,0.5]]', points_px=None, corners=None))
    _scan("update", [box(600, 400, 90, 90, "mature leaf") for _ in range(1)]
          + [box(500, 400, 90, 90, "mature leaf"), box(700, 400, 90, 90, "mature leaf")])
    before = list(main.PLANTS.values())[0]["leaf_count"]
    assert before == 3, before

    # 다음 사진엔 1장만 잡혔다 (나머지는 가려짐)
    res = _scan("keep", [box(600, 400, 90, 90, "mature leaf")])
    after = list(main.PLANTS.values())[0]
    assert after["leaf_count"] == 3, f"잎이 깎였다: {after['leaf_count']}"
    assert res["new_leaves"] == 0
    _reset()


def test_keep_mode_records_new_leaves():
    """잎이 늘어난 만큼만 새 잎으로 기록하고 기록을 남긴다."""
    _reset()
    asyncio.run(main.set_pots(points='[[0.5,0.5]]', points_px=None, corners=None))
    _scan("update", [box(600, 400, 90, 90, "mature leaf"),
                     box(500, 400, 90, 90, "mature leaf")])
    assert list(main.PLANTS.values())[0]["leaf_count"] == 2

    res = _scan("keep", [box(600, 400, 90, 90, "mature leaf"),
                         box(500, 400, 90, 90, "mature leaf"),
                         box(700, 400, 90, 90, "shoot"),
                         box(650, 330, 90, 90, "shoot")])
    p = list(main.PLANTS.values())[0]
    assert p["leaf_count"] == 4, p["leaf_count"]
    assert res["new_leaves"] == 2, res["new_leaves"]
    assert p["leaf_log"][-1]["added"] == 2, p["leaf_log"]
    assert p["shoot_count"] == 2, p          # 단계 분포는 새 탐지를 따라감
    _reset()


def test_update_mode_still_overwrites():
    """기본 모드는 예전 그대로 — 탐지값으로 덮어쓴다."""
    _reset()
    asyncio.run(main.set_pots(points='[[0.5,0.5]]', points_px=None, corners=None))
    _scan("update", [box(600, 400, 90, 90, "mature leaf"),
                     box(500, 400, 90, 90, "mature leaf"),
                     box(700, 400, 90, 90, "mature leaf")])
    _scan("update", [box(600, 400, 90, 90, "mature leaf")])
    assert list(main.PLANTS.values())[0]["leaf_count"] == 1
    _reset()


def test_keep_mode_protects_hand_edits():
    """손으로 고친 값도 keep 모드에선 깎이지 않는다."""
    _reset()
    asyncio.run(main.set_pots(points='[[0.5,0.5]]', points_px=None, corners=None))
    _scan("update", [box(600, 400, 90, 90, "mature leaf")])
    pid = list(main.PLANTS)[0]
    args = {"name": None, "rot": None, "size_class": None, "shoot_count": None,
            "mature_count": 7, "old_count": None}
    asyncio.run(main.update_plant(pid=pid, **args))
    assert main.PLANTS[pid]["leaf_count"] == 7

    _scan("keep", [box(600, 400, 90, 90, "mature leaf")])   # 여전히 1장만 잡힘
    assert main.PLANTS[pid]["leaf_count"] == 7, main.PLANTS[pid]["leaf_count"]
    _reset()


def test_leaf_log_is_capped():
    """기록이 무한정 쌓이지 않는다."""
    old = {"leaf_count": 0, "shoot_count": 0, "mature_count": 0, "old_count": 0,
           "leaf_log": [{"at": "x", "added": 1, "total": 1}] * 40}
    merged, added = main._merge_keep(old, {"leaf_count": 1, "shoot_count": 1,
                                           "mature_count": 0, "old_count": 0})
    assert added == 1
    log = old["leaf_log"]; log.append({"at": "y", "added": 1, "total": 1})
    del log[:-30]
    assert len(log) == 30


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✔ {t.__name__}")
    print(f"\n{len(tests)}개 통과")
