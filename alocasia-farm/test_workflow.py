"""워크플로 응답 파싱 스모크 테스트 (네트워크 불필요)

실행:  python3 test_workflow.py

로보플로우 워크플로는 출력 블록 이름이 워크플로마다 달라서 응답 모양이 제각각이다.
어떤 모양으로 와도 잎 박스를 뽑아낼 수 있는지 확인한다.
"""

import os
os.environ["FARM_DB"] = ""      # 테스트는 파일에 저장하지 않는다

from main import analyze_metrics, analyze_top
from providers.roboflow_workflow import (extract_boxes as _extract_workflow_boxes,
                                         image_area_px as _workflow_image_area,
                                         workflow_urls)


def _pred(x, y, w, h, cls, conf=0.9):
    return {"x": x, "y": y, "width": w, "height": h, "class": cls, "confidence": conf}


def test_flat_shape():
    """가장 흔한 모양: outputs[0].predictions.predictions"""
    payload = {"outputs": [{"predictions": {"predictions": [
        _pred(100, 80, 60, 50, "old leaf"),
        _pred(300, 200, 40, 40, "shoot"),
    ]}}]}
    boxes = _extract_workflow_boxes(payload)
    assert len(boxes) == 2, boxes
    assert {b["cls"] for b in boxes} == {"old leaf", "shoot"}


def test_named_output_block():
    """출력 블록 이름이 커스텀인 경우 (find-old-leaf-and-others 같은)"""
    payload = {"outputs": [{
        "old_leaf_detections": {"predictions": [_pred(50, 50, 20, 20, "old leaf")]},
        "other_detections": {"predictions": [_pred(90, 90, 30, 30, "mature leaf")]},
        "some_label": "무시되어야 함",
        "count": 2,
    }]}
    boxes = _extract_workflow_boxes(payload)
    assert len(boxes) == 2, boxes
    assert {b["cls"] for b in boxes} == {"old leaf", "mature leaf"}


def test_deduplicates_repeated_boxes():
    """같은 박스가 여러 출력 블록에 중복으로 실려 와도 한 번만 센다."""
    same = _pred(120, 120, 40, 40, "mature leaf")
    payload = {"outputs": [
        {"a": {"predictions": [same]}},
        {"b": {"predictions": [dict(same)]}},
    ]}
    assert len(_extract_workflow_boxes(payload)) == 1


def test_confidence_filter():
    """CONFIDENCE(기본 25%) 미만은 버린다."""
    payload = {"outputs": [{"p": {"predictions": [
        _pred(10, 10, 10, 10, "shoot", conf=0.05),   # 5% → 탈락
        _pred(50, 50, 10, 10, "shoot", conf=0.80),   # 80% → 통과
    ]}}]}
    boxes = _extract_workflow_boxes(payload)
    assert len(boxes) == 1 and boxes[0]["conf"] == 0.80, boxes


def test_malformed_entries_are_skipped():
    """좌표가 깨진 항목이 섞여 있어도 죽지 않는다."""
    payload = {"outputs": [{"p": {"predictions": [
        {"x": 10, "width": 10, "class": "shoot"},          # y/height 없음
        {"x": "?", "y": 1, "width": 1, "height": 1},        # 숫자 아님
        _pred(50, 50, 10, 10, "mature leaf"),
    ]}}]}
    assert len(_extract_workflow_boxes(payload)) == 1


def test_empty_response():
    assert _extract_workflow_boxes({"outputs": []}) == []
    assert _extract_workflow_boxes({}) == []


def test_boxes_feed_the_3d_and_modal_metrics():
    """뽑아낸 박스가 3D(모델1)·모달(모델2) 지표로 그대로 흘러가는지."""
    payload = {"outputs": [{"p": {"predictions": [
        _pred(200, 60, 300, 260, "old leaf"),      # 가장 위쪽 → 맨 위 잎
        _pred(200, 400, 120, 110, "mature leaf"),
        _pred(210, 410, 120, 110, "mature leaf"),  # 위와 크게 겹침
        _pred(500, 500, 40, 40, "shoot"),
    ]}}]}
    boxes = _extract_workflow_boxes(payload)
    img_area = 800 * 800

    top = analyze_top(boxes, img_area)
    assert top["top_leaf_size"] in ("소엽", "중엽", "대엽")

    m = analyze_metrics(boxes, img_area)
    assert (m["shoot_count"], m["mature_count"], m["old_count"]) == (1, 2, 1), m
    assert m["leaf_count"] == 4
    assert m["overlap_count"] == 2 and m["overlap_density"] == 50, m


def test_image_area_prefers_workflow_reported_size():
    """워크플로가 리사이즈했다면 응답이 알려 주는 크기를 써야 한다."""
    payload = {"outputs": [{"p": {"image": {"width": 640, "height": 640},
                                  "predictions": [_pred(10, 10, 10, 10, "shoot")]}}]}
    assert _workflow_image_area(payload, fallback=100.0) == 640 * 640


def test_image_area_falls_back_to_original():
    payload = {"outputs": [{"p": {"predictions": [_pred(10, 10, 10, 10, "shoot")]}}]}
    assert _workflow_image_area(payload, fallback=1234.0) == 1234.0


def test_image_area_ignores_garbage_dims():
    payload = {"outputs": [{"p": {"image": {"width": 0, "height": "?"}}}]}
    assert _workflow_image_area(payload, fallback=999.0) == 999.0


def test_top_leaf_pct_never_exceeds_100():
    """좌표계가 어긋나도 119% 같은 값이 UI 로 새어 나가면 안 된다."""
    boxes = _extract_workflow_boxes(
        {"outputs": [{"p": {"predictions": [_pred(200, 60, 300, 260, "old leaf")]}}]})
    top = analyze_top(boxes, img_area=256 * 256)      # 박스가 이미지보다 큰 상황
    assert top["top_leaf_pct"] == 100.0, top


def test_url_candidates():
    urls = workflow_urls("s-workspace-br86f", "find-old-leaf-and-others")
    assert len(urls) >= 1
    assert all(u.startswith("http") for u in urls), urls


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✔ {t.__name__}")
    print(f"\n{len(tests)}개 통과")
