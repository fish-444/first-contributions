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


# ── API 키 다듬기 ────────────────────────────────────────────────────────
# 401 의 대부분은 키가 죽어서가 아니라 따옴표·공백이 값에 섞여서다.
#
# 키를 다듬는 쪽(clean_api_key)과 사람에게 알리는 쪽(api_key_warnings)은 일부러
# 갈라 놓았다. 한 함수가 (키, 경고) 를 함께 돌려주면 경고를 찍는 코드가 키를
# 찍는 코드와 구별되지 않는다 — 코드 스캐닝이 그 줄을 비밀값 평문 로깅으로
# 잡았고, 그 지적이 맞았다.
from providers import api_key_warnings, clean_api_key


def test_batch_style_quotes_are_stripped_from_the_key():
    """배치 파일에서 set X="k" 라고 쓰면 따옴표까지 값이 된다 — 걷어내고 알려 준다."""
    assert clean_api_key('"abcdsecret"') == "abcdsecret"
    warns = api_key_warnings('"abcdsecret"')
    assert len(warns) == 1 and "따옴표" in warns[0], warns


def test_surrounding_whitespace_is_stripped():
    assert clean_api_key("  abcdsecret\n") == "abcdsecret"
    warns = api_key_warnings("  abcdsecret\n")
    assert warns and "공백" in warns[0], warns


def test_publishable_key_is_flagged():
    """rf_ 로 시작하는 공개키는 워크플로에서 막힌다 — 미리 알려 준다."""
    assert clean_api_key("rf_workspaceid_example000000000") == "rf_workspaceid_example000000000"
    warns = api_key_warnings("rf_workspaceid_example000000000")
    assert any("공개" in w for w in warns), warns


def test_a_clean_key_produces_no_noise():
    assert clean_api_key("abcdsecret") == "abcdsecret"
    assert api_key_warnings("abcdsecret") == []


def test_missing_key_is_not_an_error():
    """키를 아예 안 넣은 건 데모 모드로 가는 정상 경로다 — 경고를 뿌리지 않는다."""
    assert clean_api_key("") == "" and api_key_warnings("") == []
    assert clean_api_key(None) == "" and api_key_warnings(None) == []


def test_no_warning_ever_carries_the_key_itself():
    """경고 문구는 사람에게 보여 주는 것이자 로그로 남는 것이다 — 키가 섞이면 안 된다.

    키를 담아 두면 select() 의 `print(f"[키 경고] {w}")` 가 곧바로 비밀값을
    콘솔에 남긴다. 실제로 코드 스캐닝(py/clear-text-logging-sensitive-data)이
    그 줄을 잡았다. 문구가 늘어날 때 다시 섞이지 않게 여기서 못박아 둔다.
    """
    secret = "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"
    for raw in (f'"{secret}"', f"  {secret}\n", f"rf_{secret}", f"{secret[:8]} {secret[8:]}"):
        for w in api_key_warnings(raw):
            assert secret not in w, w
            assert secret[:4] not in w, w          # 앞자리만 흘리는 것도 안 된다


def test_the_startup_line_does_not_print_any_of_the_key(capsys, monkeypatch):
    """'앱이 어떤 키를 들고 있나' 를 알려 주되, 키 글자는 한 자도 찍지 않는다."""
    secret = "wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY"
    monkeypatch.setenv("ROBOFLOW_API_KEY", secret)
    for name in ("ROBOFLOW_WORKSPACE", "ROBOFLOW_WORKFLOW_ID", "ROBOFLOW_WORKFLOW_URL"):
        monkeypatch.delenv(name, raising=False)

    import providers
    providers.select()

    out = capsys.readouterr().out
    assert f"({len(secret)}자)" in out, out       # 길이는 알려 준다 (따옴표·잘림 진단)
    assert secret[:4] not in out, out             # 앞자리는 안 된다
    assert secret not in out, out


# ── 설정 파일 읽기 ────────────────────────────────────────────────────────
# start.bat 이 call 로 불러 주는 것에만 기대면, uvicorn 을 직접 띄웠을 때
# 설정이 통째로 무시돼 조용히 데모 모드로 떨어진다.
import os
import tempfile
from main import _load_env_file


def _with_env_file(text, env=None, name="farm_env.bat"):
    """임시 폴더에 설정 파일을 두고 _load_env_file 을 돌려 본다."""
    before = dict(os.environ)
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(text)
            os.chdir(d)
            for k in ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "FARM_PORT"):
                os.environ.pop(k, None)
            os.environ.update(env or {})
            _load_env_file()
            return {k: os.environ.get(k) for k in
                    ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "FARM_PORT")}
    finally:
        os.chdir(cwd)
        os.environ.clear()
        os.environ.update(before)


def test_the_app_reads_farm_env_itself():
    """uvicorn 을 직접 띄워도 키가 잡혀야 한다 — 실행 방법에 따라 달라지면 안 된다."""
    got = _with_env_file("@echo off\n"
                         "set ROBOFLOW_API_KEY=cuxIsecret\n"
                         "set ROBOFLOW_WORKSPACE=s-workspace-br86f\n")
    assert got["ROBOFLOW_API_KEY"] == "cuxIsecret", got
    assert got["ROBOFLOW_WORKSPACE"] == "s-workspace-br86f", got


def test_comments_are_skipped():
    got = _with_env_file("@echo off\n"
                         "rem set ROBOFLOW_API_KEY=주석이라_무시\n"
                         ":: set ROBOFLOW_WORKSPACE=이것도\n"
                         "set ROBOFLOW_API_KEY=cuxIreal\n")
    assert got["ROBOFLOW_API_KEY"] == "cuxIreal", got
    assert got["ROBOFLOW_WORKSPACE"] is None, got


def test_an_already_set_variable_wins():
    """파워셸에서 직접 지정했거나 start.bat 이 먼저 넣어 준 값이 우선이어야 한다."""
    got = _with_env_file("set ROBOFLOW_API_KEY=fromFile\n",
                         env={"ROBOFLOW_API_KEY": "fromShell"})
    assert got["ROBOFLOW_API_KEY"] == "fromShell", got


def test_shell_style_env_file_also_works():
    """맥·리눅스의 farm_env.sh 는 export 형식이다."""
    got = _with_env_file("#!/bin/sh\nexport ROBOFLOW_API_KEY=cuxIsh\nFARM_PORT=8123\n",
                         name="farm_env.sh")
    assert got["ROBOFLOW_API_KEY"] == "cuxIsh", got
    assert got["FARM_PORT"] == "8123", got


def test_no_env_file_is_not_an_error():
    before = dict(os.environ)
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            assert _load_env_file() is None      # 조용히 지나가기만 하면 된다
    finally:
        os.chdir(cwd); os.environ.clear(); os.environ.update(before)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✔ {t.__name__}")
    print(f"\n{len(tests)}개 통과")
