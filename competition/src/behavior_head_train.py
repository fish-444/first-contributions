"""행동 헤드 가중치 학습 — **등가중을 이길 때만** 교체 후보가 된다.

지금 `behavior_baseline` 의 헤드 점수는 등가중이다(부호만 문헌). 유튜브
CCTV 라벨(방·구간·사람관측)이 오면 이 모듈이 가중치를 학습하고, 같은
기준선·같은 분할에서 등가중과 겨룬다. 규약은 라벨을 열기 전에
`docs/PREREGISTRATION.md` 등록 2 로 못박았다 — 여기 상수·판정은 그 등록의
구현이지 재량이 아니다.

    분할     방 단위 leave-one-room-out (창 단위 분할은 성적을 부풀린다)
    특징     자기 기준선 편차 z 4종만 — 표본이 적은데 특징을 늘리면
             그럴듯한 과적합이 나온다
    이길 대상 같은 프로토콜의 등가중 헤드 점수 AUC
    채택     학습이 이기고 + 계수 부호가 문헌과 충돌하지 않을 때만 후보.
             지면 등가중 유지 — 그대로 보고한다
    성립     방 ≥ 3 · 대비(양쪽 라벨) 방 ≥ 2 · 양성 창 ≥ 10. 미달이면
             학습하지 않는다

라벨 CSV 헤더(고정): ``room,video,start_sec,end_sec,label``

    # 로컬 실행 (dets 는 pig_behavior.cli 출력)
    python competition/src/behavior_head_train.py \
        --dets-dir data/cctv/dets --labels data/cctv/labels/labels.csv \
        --per 60 --sec-per-frame 30 --head estrus \
        --positive 발정 --negative 비발정 --out data/cctv/summary/head_estrus.json

    python competition/src/behavior_head_train.py     # 합성 시연 (등급 합성)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import behavior_baseline as bb              # noqa: E402
import vision_pig_behavior as vpb           # noqa: E402
from pig_behavior.predictor import Detection  # noqa: E402

MIN_ROOMS = 3        # LORO 가 성립할 최소 방 수
MIN_CONTRAST = 2     # 발정·비발정이 둘 다 있는 방의 최소 수
MIN_POS = 10         # 최소 양성 창 수
LABEL_HEADER = ("room", "video", "start_sec", "end_sec", "label")


def load_windows(jsonl_path: str, per: int) -> list:
    """검출 JSONL → `fold()` 입력 창 목록. 꼬리 자투리 창은 버린다."""
    frames = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            dets = [Detection(d["label"], d["score"], tuple(d["bbox"]))
                    for d in r["detections"]]
            frames.append((r["image"], dets))
    return [frames[i:i + per] for i in range(0, len(frames) - per + 1, per)]


def load_labels(path: str) -> list:
    """라벨 CSV → 행 목록. 헤더가 규약과 다르면 **바로 죽는다** — 열을
    짐작해서 맞으면 그게 더 위험하다(71471 카메라 라벨의 교훈)."""
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        got = tuple((rd.fieldnames or []))
        if got != LABEL_HEADER:
            raise SystemExit(f"라벨 헤더가 규약과 다르다: {got} ≠ {LABEL_HEADER}")
        return [{"room": r["room"], "video": r["video"],
                 "t0": float(r["start_sec"]), "t1": float(r["end_sec"]),
                 "label": r["label"].strip()} for r in rd]


def label_windows(n: int, per: int, sec_per_frame: float, spans: list,
                  positive: str, negative: str) -> list:
    """창마다 겹침이 가장 큰 라벨. 양쪽 다 안 겹치면 None(비라벨 — 기준선
    이력에는 남고 학습 표본에서는 빠진다)."""
    out = []
    wlen = per * sec_per_frame
    for i in range(n):
        w0, w1 = i * wlen, (i + 1) * wlen
        best, best_ov = None, 0.0
        for s in spans:
            if s["label"] not in (positive, negative):
                continue
            ov = min(w1, s["t1"]) - max(w0, s["t0"])
            if ov > best_ov:
                best, best_ov = s["label"], ov
        out.append(best)
    return out


def build_dataset(dets_dir: str, labels: list, per: int, sec_per_frame: float,
                  positive: str, negative: str) -> dict:
    """방별 {probs(전 창 — 기준선 이력), y(라벨 창만, 창 인덱스 포함)}."""
    by_video: dict = {}
    for r in labels:
        by_video.setdefault(r["video"], []).append(r)
    rooms: dict = {}
    for video, spans in sorted(by_video.items()):
        room = {s["room"] for s in spans}
        if len(room) != 1:
            raise SystemExit(f"영상 {video} 가 방 여러 개에 걸려 있다: {room}")
        room = room.pop()
        path = os.path.join(dets_dir, f"{video}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"검출 파일이 없다: {path}")
        wins = load_windows(path, per)
        probs = [vpb.fold(w, "cam", room, "-", model="head-train")[0]["obs"].probs
                 for w in wins]
        ys = label_windows(len(wins), per, sec_per_frame, spans,
                           positive, negative)
        d = rooms.setdefault(room, {"probs": [], "labeled": []})
        base = len(d["probs"])
        d["probs"] += probs
        d["labeled"] += [(base + i, 1 if y == positive else 0)
                         for i, y in enumerate(ys) if y is not None]
    return rooms


def audit(rooms: dict) -> dict:
    """성립 조건 검사 — 학습 가부를 **학습 전에** 말한다."""
    per_room = {}
    for room, d in rooms.items():
        ys = [y for _, y in d["labeled"]]
        per_room[room] = {"windows": len(d["probs"]), "labeled": len(ys),
                          "pos": sum(ys), "neg": len(ys) - sum(ys),
                          "contrast": bool(sum(ys) and len(ys) - sum(ys))}
    n_pos = sum(v["pos"] for v in per_room.values())
    n_contrast = sum(1 for v in per_room.values() if v["contrast"])
    why = []
    if len(rooms) < MIN_ROOMS:
        why.append(f"방 {len(rooms)} < {MIN_ROOMS}")
    if n_contrast < MIN_CONTRAST:
        why.append(f"대비(양쪽 라벨) 방 {n_contrast} < {MIN_CONTRAST} — "
                   "within-room contrast 미성립")
    if n_pos < MIN_POS:
        why.append(f"양성 창 {n_pos} < {MIN_POS}")
    return {"rooms": per_room, "n_pos": n_pos, "n_contrast": n_contrast,
            "ok": not why, "why": why}


def _features(rooms: dict, head: str) -> dict:
    """방별 기준선 형성 → 라벨 창의 (z 특징, y, 등가중 점수)."""
    classes = vpb.CONTRACT_CLASSES
    out = {}
    for room, d in rooms.items():
        b = bb.fit(d["probs"], room, classes=classes)
        if not b.formed:
            continue                      # 기준선 미형성 방은 학습에 못 쓴다
        X, y, eq = [], [], []
        for i, yi in d["labeled"]:
            z = b.deviation(d["probs"][i])
            X.append([z[c] for c in classes])
            y.append(yi)
            eq.append(b.head_score(d["probs"][i], head))
        out[room] = {"X": np.array(X), "y": np.array(y), "eq": np.array(eq)}
    return out


def train(rooms: dict, head: str) -> dict:
    """LORO 로 학습 AUC vs 등가중 AUC. 판정은 사전 등록 2 의 규칙."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    a = audit(rooms)
    report = {"head": head, "audit": a, "protocol": "LORO(방 단위) · z 4종",
              "preregistered": "docs/PREREGISTRATION.md 등록 2"}
    if not a["ok"]:
        report["verdict"] = "학습 불가 — " + " · ".join(a["why"])
        return report
    feats = _features(rooms, head)
    dropped = sorted(set(rooms) - set(feats))
    if dropped:
        report["dropped_rooms"] = {r: "기준선 미형성" for r in dropped}
    names = sorted(feats)
    classes = vpb.CONTRACT_CLASSES

    def _clf():
        return LogisticRegression(class_weight="balanced", max_iter=1000)

    pool_y, pool_learn, pool_eq, within = [], [], [], {}
    for held in names:
        rest = [r for r in names if r != held]
        Xtr = np.vstack([feats[r]["X"] for r in rest])
        ytr = np.concatenate([feats[r]["y"] for r in rest])
        if len(set(ytr)) < 2:
            report["verdict"] = f"학습 불가 — {held} 제외 시 훈련 라벨이 한쪽뿐"
            return report
        clf = _clf().fit(Xtr, ytr)
        p = clf.predict_proba(feats[held]["X"])[:, 1]
        pool_y += list(feats[held]["y"])
        pool_learn += list(p)
        pool_eq += list(feats[held]["eq"])
        if len(set(feats[held]["y"])) == 2:
            within[held] = {
                "learned": round(float(roc_auc_score(feats[held]["y"], p)), 3),
                "equal": round(float(roc_auc_score(feats[held]["y"],
                                                   feats[held]["eq"])), 3)}
    auc_l = round(float(roc_auc_score(pool_y, pool_learn)), 3)
    auc_e = round(float(roc_auc_score(pool_y, pool_eq)), 3)

    # 보고용 가중치는 전 데이터 적합 — 부호를 문헌 부호와 대조한다
    Xa = np.vstack([feats[r]["X"] for r in names])
    ya = np.concatenate([feats[r]["y"] for r in names])
    coef = _clf().fit(Xa, ya).coef_[0]
    weights = {c: round(float(w), 3) for c, w in zip(classes, coef)}
    signs = bb.HEAD_SIGNS[head]
    conflicts = [c for c in signs if c in weights
                 and weights[c] * signs[c] < 0]

    beats = auc_l > auc_e
    if beats and not conflicts:
        verdict = "채택 후보 — 등가중을 이겼고 부호가 문헌과 일치. 배선은 사용자 확인 후"
    elif beats:
        verdict = f"보류 — 이겼으나 부호 충돌 {conflicts}. 원인 규명 전 교체 금지"
    else:
        verdict = "등가중 유지 — 학습이 이기지 못했다. 그대로 보고한다"
    report.update({"auc_learned_loro": auc_l, "auc_equal": auc_e,
                   "within_room": within, "weights": weights,
                   "sign_conflicts": conflicts, "verdict": verdict})
    return report


# -- 합성 시연 ---------------------------------------------------------------
def _synth(tmp: str, rng, rooms=4, wins=30, pos_from=20) -> str:
    """방 rooms 개 × 창 wins 개 분량의 dets/labels 합성. 뒤쪽 창을 발정으로."""
    os.makedirs(os.path.join(tmp, "dets"), exist_ok=True)
    rows = []
    for k in range(rooms):
        room, video = f"R{k}", f"v{k}"
        with open(os.path.join(tmp, "dets", f"{video}.jsonl"), "w",
                  encoding="utf-8") as f:
            for i in range(wins * 60):
                estrus = i // 60 >= pos_from
                w = [55, 25, 10, 6] if not estrus else [35, 12, 34, 15]
                dets = []
                for _ in range(8):
                    lab = ["Resting", "Eating", "Walking", "Searching"][
                        rng.choice(4, p=np.array(w) / sum(w))]
                    dets.append({"label": lab,
                                 "score": round(float(rng.uniform(.4, .9)), 4),
                                 "bbox": [0, 0, 50, 50], "reliable": True})
                f.write(json.dumps({"image": f"{i:06d}.jpg",
                                    "detections": dets}) + "\n")
        for i in range(wins):
            rows.append((room, video, i * 1800, (i + 1) * 1800,
                         "발정" if i >= pos_from else "비발정"))
    with open(os.path.join(tmp, "labels.csv"), "w", encoding="utf-8",
              newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(LABEL_HEADER)
        wcsv.writerows(rows)
    return tmp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="행동 헤드 가중치 학습")
    ap.add_argument("--dets-dir")
    ap.add_argument("--labels")
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--sec-per-frame", type=float, default=30.0)
    ap.add_argument("--head", default="estrus", choices=sorted(bb.HEAD_SIGNS))
    ap.add_argument("--positive", default="발정")
    ap.add_argument("--negative", default="비발정")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.dets_dir and args.labels:
        rows = load_labels(args.labels)
        rooms = build_dataset(args.dets_dir, rows, args.per,
                              args.sec_per_frame, args.positive, args.negative)
        report = train(rooms, args.head)
        report["grade"] = "실측(외부 영상·사람관측 라벨 — 실농장 실증 아님)"
    else:
        import tempfile
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            _synth(tmp, rng)
            rows = load_labels(os.path.join(tmp, "labels.csv"))
            rooms = build_dataset(os.path.join(tmp, "dets"), rows,
                                  60, 30.0, "발정", "비발정")
            report = train(rooms, "estrus")
        report["grade"] = "합성 — 파이프라인 관통 확인일 뿐 성능이 아니다"
        print("=" * 72)
        print("  헤드 가중치 학습 — 합성 시연 (**등급 합성**)")
        print("=" * 72)
        print(f"  방 {len(report['audit']['rooms'])}개 · "
              f"양성 창 {report['audit']['n_pos']} · "
              f"대비 방 {report['audit']['n_contrast']}")
        print(f"  LORO AUC  학습 {report['auc_learned_loro']} vs "
              f"등가중 {report['auc_equal']}")
        print(f"  가중치 {report['weights']} · 부호 충돌 {report['sign_conflicts']}")
        print(f"  판정: {report['verdict']}")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"보고 → {args.out}")
    elif args.dets_dir:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
