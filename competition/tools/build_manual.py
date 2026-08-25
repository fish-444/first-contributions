"""전체 파이썬 모듈 설명서를 **코드에서 생성**한다 → `docs/MANUAL.md`.

109개를 손으로 적으면 다음 커밋에 낡는다. 그래서 목록도 설명도 파일에서
뽑는다: 모듈 요약은 **각 파일 docstring 첫 문장**이고, 실행 예시는
docstring 안의 `python competition/...` 줄이며, 규모 수치는
`check_docs.actual_counts()` 를 그대로 부른다.

묶음(GROUPS)만 사람이 정한다 — 파일 이름으로는 "이게 어느 축의 일인가"를
알 수 없기 때문이다. 대신 **어느 묶음에도 안 든 파일은 '미분류'로 크게
찍는다.** 새 모듈이 조용히 설명서 밖에 남는 것을 막는 자리다.

    python competition/tools/build_manual.py
"""
from __future__ import annotations

import ast
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(HERE)
OUT = os.path.join(COMP, "docs", "MANUAL.md")
sys.path.insert(0, HERE)

import check_docs  # noqa: E402

#: 묶음 정의 — (제목, 설명, 이름 목록 또는 접두사 규칙).
#  순서대로 먼저 맞는 묶음에 넣는다. 접두사는 `*` 로 끝낸다.
GROUPS: list = [
    ("기둥 1 — 발정·교배 관리", "농장이 오늘 쓰는 번식 판단.", [
        "breeding_timing", "repro_calendar", "pregnancy_check", "herd_board",
        "breeding_ledger", "work_log", "mating_plan", "estrus_calendar",
        "estrus_early_warning", "estrus_onset", "estrus_link",
        "repro_cause_attribution", "stall_estrus", "feeding_monitor",
        "pipeline_gilt", "vision_contract",
    ]),
    ("기둥 2 — 돈사 운영·행정", "지은 돈사에서 무엇이 병목이고 몇 두를 받는가.", [
        "batch_flow", "barn_watch", "growth_flow", "farm_registry",
        "farm_scale", "legal_density", "perf_formula", "improve_path",
        "run_farm", "synth_farm", "table_export",
    ]),
    ("환경·생체 — 지침 층 + 자기 기준선 층", "위험(지침 위반)과 주의(평소와 다름)를 겹으로 본다.", [
        "barn_env_control", "barn_environment", "env_scale",
        "bio_baseline_71763", "behavior_baseline", "behavior_vocab",
        "behavior_head_train", "vision_pig_behavior",
    ]),
    ("근거층 — 국내 실측 분석", "466행·179농장에서 나온 격차·계절·하락.", [
        "korean_farm_stats", "farm_gap", "farm_panel", "farm_monthly",
        "farm_monthly_panel", "psy_priority", "farm_economics",
        "path_predict", "estrus_label_audit", "eda",
    ]),
    ("영상·모델 — 학습과 평가", "탐지·자세·행동·추적. 되는 범위를 재는 코드까지.", [
        "model_*", "train*", "posture_*", "view_align", "iou_tracker",
        "motion_tracker", "box_merge", "polygon_shape_eval",
        "pose_vs_behavior_eval", "analyze_*", "finetune_polygon",
        "tracking_sweep", "temporal_features", "ml_core",
        "validate_estrus_reference", "estrus_contrast_eval",
    ]),
    ("데이터 — 파서·수집", "원자료를 표로 바꾸는 층. 여기서 감사도 한다.", [
        "parse_*", "fetch_*", "aihub*", "generate_data",
    ]),
    ("화면 — 대시보드 빌더", "정적 HTML 을 만든다. 계산은 위 모듈을 부른다.", [
        "build_*",
    ]),
]


def summary(path: str) -> tuple:
    """파일 → (첫 문장, 실행 예시들, 실행 가능 여부)."""
    src = open(path, encoding="utf-8").read()
    try:
        doc = ast.get_docstring(ast.parse(src)) or ""
    except SyntaxError:
        doc = ""
    first = ""
    for line in doc.splitlines():
        if line.strip():
            first = line.strip()
            break
    # 마침표까지만 — 첫 문장이 요약이다
    first = re.split(r"(?<=[.다])\s", first)[0].strip()
    cmds = re.findall(r"^\s{4}(python competition/[^\n]+)$", doc, re.M)
    return first, cmds, "__main__" in src


def group_of(name: str) -> str | None:
    for title, _desc, pats in GROUPS:
        for p in pats:
            if (name == p if not p.endswith("*")
                    else name.startswith(p[:-1])):
                return title
    return None


def build() -> str:
    a = check_docs.actual_counts()
    files = sorted(glob.glob(os.path.join(COMP, "src", "*.py")))
    tools = sorted(glob.glob(os.path.join(COMP, "tools", "*.py")))

    buckets: dict = {t: [] for t, _d, _p in GROUPS}
    unfiled: list = []
    for f in files:
        name = os.path.basename(f)[:-3]
        row = (name,) + summary(f)
        g = group_of(name)
        (buckets[g] if g else unfiled).append(row)

    L = ["# 전체 모듈 설명서 — **코드에서 생성된다**", "",
         "> `python competition/tools/build_manual.py` 로 다시 만든다.",
         "> 각 줄의 설명은 그 파일 docstring 의 첫 문장이고, 실행 예시도",
         "> docstring 에서 뽑는다 — **손으로 고치지 말 것.** 설명을 바꾸려면",
         "> 모듈의 docstring 을 고친다.", "",
         f"**규모**: 도메인·모델 모듈 {a['modules']}개 · 빌더 {a['builders']}개 "
         f"(src {a['src_total']}개) · 대시보드 뷰 {a['views']}개 · "
         f"테스트 {a['tests']}개", "",
         "## 먼저 읽을 것 셋", "",
         "1. `docs/CAPABILITIES.md` — 되는 것과 **안 되는 것(§E)**",
         "2. `docs/REPORT_INPUT.md` — 제안서를 쓸 때 인용할 사실 전부",
         "3. `docs/AIHUB_71763.md` · `docs/PREREGISTRATION.md` — 데이터 감사와 "
         "사전등록 규약", "",
         "## 이 저장소의 규율 다섯", "",
         "1. **계산을 재구현하지 않는다.** 서버도 화면도 모듈을 부른다.",
         "2. **상수를 화면에 박지 않는다.** 정본 모듈에서 주입한다.",
         "3. **문턱을 발명하지 않는다.** 자기 이력의 경보율 대역에서 역산한다.",
         "4. **근거 있는 것만 신고한다.** 못 내는 것은 못 낸다고 답한다.",
         "5. **등급을 섞지 않는다.** 실측/계산/유도/합성/지침.", ""]

    for title, desc, _p in GROUPS:
        rows = buckets[title]
        if not rows:
            continue
        L += [f"## {title}", "", desc, "",
              "| 모듈 | 하는 일 | 실행 |", "|---|---|---|"]
        for name, first, cmds, runnable in rows:
            run = "`직접 실행`" if runnable else "라이브러리"
            L.append(f"| `{name}` | {first or '—'} | {run} |")
        L.append("")
        cmd_rows = [(n, c) for n, _f, cs, _r in rows for c in cs]
        if cmd_rows:
            L += ["<details><summary>실행 예시</summary>", "", "```"]
            L += [c for _n, c in cmd_rows]
            L += ["```", "", "</details>", ""]

    if unfiled:
        L += ["## ⚠ 미분류 — **묶음에 안 든 모듈**", "",
              "새 모듈이 설명서 밖에 조용히 남는 것을 막으려고 크게 찍는다. "
              "`tools/build_manual.py` 의 `GROUPS` 에 넣을 것.", "",
              "| 모듈 | 하는 일 |", "|---|---|"]
        for name, first, _c, _r in unfiled:
            L.append(f"| `{name}` | {first or '—'} |")
        L.append("")

    L += ["## 도구 (`tools/`)", "", "| 도구 | 하는 일 |", "|---|---|"]
    for f in tools:
        name = os.path.basename(f)[:-3]
        if name.startswith("_"):
            continue
        L.append(f"| `{name}` | {summary(f)[0] or '—'} |")
    L += ["",
          "## 검증", "",
          "```",
          "python competition/tests/smoke_test.py     # 전체 테스트",
          "python competition/tools/check_docs.py     # 문서 수치 대조",
          "```", ""]
    return "\n".join(L)


def main() -> int:
    text = build()
    open(OUT, "w", encoding="utf-8").write(text)
    n_unfiled = text.count("## ⚠ 미분류")
    print(f"설명서 생성: {OUT} ({len(text) // 1024}KB)")
    if n_unfiled:
        print("  ⚠ 미분류 모듈이 있다 — GROUPS 에 넣을 것")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
