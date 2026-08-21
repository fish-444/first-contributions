"""CLI 검출 JSONL → 시간창 → 행동 기준선 — **로컬 실증용 브리지.**

`pig_behavior.cli` 가 뱉은 검출 JSONL 을 창으로 묶어 `vision_pig_behavior.fold()`
→ `behavior_baseline.fit()/assess()` 에 그대로 통과시킨다. 계산은 전부 src
정본이 하고, 이 파일은 **배선만** 한다 — 여기에 문턱·가중치·판정식을 더하지
말 것. 로컬 세션이 스니펫을 새로 짜다 재구현하는 사고를 막으려고 커밋해 둔다.

    python competition/tools/baseline_from_dets.py data/cctv/dets/ID.jsonl \
        --barn A --pen 1 --per 60 [--heads estrus] [--out summary.json]

- `--per` 는 창 하나의 프레임 수다. 30초 1장 추출이면 60 = 30분 창.
- `--heads` 는 달력이 연 헤드만 계산하는 자리다(발정 라벨 창 평가는
  `--heads estrus`). 생략하면 전 헤드 — 시연·감사용.
- 출력은 집계 요약뿐이다(방 키·창 수·기준선·창별 점수). 프레임 경로 같은
  원자료 식별자는 싣지 않는다 — 요약만이 커밋 후보이고 그것도 사용자 확인
  후다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import behavior_baseline as bb          # noqa: E402
import vision_pig_behavior as vpb       # noqa: E402
from pig_behavior.predictor import Detection  # noqa: E402


def load_windows(jsonl_path: str, per: int) -> list:
    """JSONL 프레임들 → `fold()` 입력 창 목록. 꼬리 자투리 창은 버린다 —
    창마다 프레임 수가 다르면 구성비의 잡음 수준이 달라진다."""
    frames = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            dets = [Detection(d["label"], d["score"], tuple(d["bbox"]))
                    for d in r["detections"]]
            frames.append((r["image"], dets))
    return [frames[i:i + per] for i in range(0, len(frames) - per + 1, per)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", help="pig_behavior.cli --out 이 만든 JSONL")
    ap.add_argument("--barn", required=True, help="동 익명키 (예: A)")
    ap.add_argument("--pen", required=True, help="방 익명키 (예: 1)")
    ap.add_argument("--per", type=int, default=60,
                    help="창 하나의 프레임 수 (30초 1장이면 60 = 30분)")
    ap.add_argument("--camera", default="cam", help="카메라 익명키")
    ap.add_argument("--heads", nargs="*", default=None,
                    help="달력이 연 헤드만 (예: estrus). 생략 = 전 헤드")
    ap.add_argument("--out", help="요약 JSON 경로 (생략하면 표준출력)")
    args = ap.parse_args(argv)

    windows = load_windows(args.jsonl, args.per)
    model = f"pig-behavior-0.1.0/{os.path.basename(args.jsonl)}"
    folded = [vpb.fold(w, args.camera, args.barn, args.pen, model=model)[0]
              for w in windows]
    key = f"{args.barn}/{args.pen}"
    hist = bb.summarize(folded).get(key, [])
    # 어휘는 계약 4종을 명시한다 — 이력에서 추론하면 한 번도 안 보인 행동을
    # 몰라서 헤드가 어휘 미비로 닫힌다(미관측은 구성비 0 인 정상 이력이다)
    b = bb.fit(hist, key, classes=vpb.CONTRACT_CLASSES)

    heads = tuple(args.heads) if args.heads else None
    rows = []
    for i, probs in enumerate(hist):
        a = bb.assess(b, probs, recent=hist[:i], heads=heads)
        rows.append({"window": i, "heads": {
            h: {k: v[k] for k in ("score", "cut", "over", "alert", "streak")}
            for h, v in a["heads"].items()}})

    summary = {
        "key": key, "n_frames": len(windows) * args.per if windows else 0,
        "per_window": args.per, "n_windows": b.n_windows,
        "formed": b.formed, "center": b.center, "spread": b.spread,
        "cuts": b.cuts, "heads": list(heads) if heads else "전 헤드(감사용)",
        "grade": "계산",
        "note": ("컷은 이 이력 자체에서 역산했다(자기 보정). 라벨 검증은 "
                 "within-room contrast 가 성립할 때만 의미가 있다 — 같은 "
                 "방에 발정 라벨 창과 비발정 창이 둘 다 있어야 한다."),
        "windows": rows,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"요약 → {args.out}  (창 {b.n_windows}개 · 형성 {b.formed})",
              file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
