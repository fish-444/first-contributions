"""71763 라벨 36만개 → 프레임·클립 CSV **한 번만** 만드는 배치 파서.

parse_aihub.parse_71763() 을 그대로 쓰면 한 프로세스가 파일을 하나씩 열어
초당 48건, 36만 건에 약 2시간이 걸린다. 병목은 파싱 로직이 아니라 파일당
open/close 왕복이라 CPU 를 나눠 주면 그대로 줄어든다 — 실측 19분(6배).

**샤드 체크포인트가 이 모듈의 존재 이유다.** 단일 프로세스판은 프레임을 전부
메모리에 모은 뒤 마지막에 한 번 저장해서, 중간에 끊기면 몇 시간치가 통째로
날아갔다(실제로 61% 지점에서 한 번 날렸다). 여기서는 샤드마다 즉시 저장하고
이미 있는 샤드는 건너뛰므로, 재실행하면 남은 것부터 이어서 한다.

샤드는 경로 정렬 뒤 `paths[i::NSHARD]` 로 나눈다. 연속 구간으로 자르지 않는
이유는 클립(=폴더)마다 프레임 수가 달라 구간 분할이면 워커별 부하가 기울기
때문이다. 어차피 클립 접기는 전체 프레임을 모은 뒤에 하므로 순서는 무관하다.

실행:
    python competition/src/parse_71763_batch.py <라벨디렉터리> [--out 디렉터리]
                                                [--procs 12] [--shards 48]
    python competition/src/parse_71763_batch.py <라벨디렉터리> --clean

되읽기(재파싱 불필요):
    python competition/src/bio_baseline_71763.py --clips <출력>/clips_71763.csv
    python competition/src/env_anomaly.py --clips <출력>/clips_71763.csv
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_aihub  # noqa: E402

#: 기본 워커 수. 코어를 다 쓰면 대화형 세션이 멎어서 넉넉히 남긴다.
DEFAULT_PROCS = 12
#: 샤드 수. 워커 수의 배수라야 꼬리에서 놀지 않는다.
DEFAULT_SHARDS = 48
FRAMES_CSV = "frames_71763.csv"
CLIPS_CSV = "clips_71763.csv"
SHARD_DIR = "shards"


def iter_label_paths(label_dir: str) -> list[str]:
    """라벨 JSON 전체 경로. 정렬해야 샤드 분할이 재실행마다 같아진다."""
    out = []
    for root, _dirs, files in os.walk(label_dir):
        for f in files:
            if f.lower().endswith(".json"):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def _parse_shard(arg):
    """워커 1개 몫. 이미 저장된 샤드면 파일을 열지도 않는다."""
    idx, label_dir, paths, shard_dir = arg
    dst = os.path.join(shard_dir, "shard_%03d.csv" % idx)
    if os.path.exists(dst):
        return idx, -1
    # parse_71763() 은 디렉터리를 훑는다. 이 워커 몫만 훑도록 바꿔 끼운다.
    parse_aihub._iter_json = lambda _d, _p=paths: iter(_p)
    df = parse_aihub.parse_71763(label_dir)
    # tmp → replace. 중간에 죽어도 반쪽 샤드를 완성본으로 오인하지 않는다.
    df.to_csv(dst + ".tmp", index=False, encoding="utf-8-sig")
    os.replace(dst + ".tmp", dst)
    return idx, len(df)


def parse_frames(label_dir: str, out_dir: str, procs: int = DEFAULT_PROCS,
                 shards: int = DEFAULT_SHARDS,
                 verbose: bool = True) -> pd.DataFrame:
    """라벨 디렉터리 → 프레임 테이블. 샤드를 남기므로 중단·재개가 된다."""
    shard_dir = os.path.join(out_dir, SHARD_DIR)
    os.makedirs(shard_dir, exist_ok=True)
    t0 = time.time()

    paths = iter_label_paths(label_dir)
    if not paths:
        raise SystemExit("라벨 JSON 이 없다: %s" % label_dir)

    # 재개 안전망의 구멍 하나를 막는다: 샤드 분할은 `paths[i::shards]` 라
    # **파일 수나 샤드 수가 지난 실행과 다르면 같은 번호의 샤드가 다른
    # 부분집합**이 된다. 그때 기존 샤드를 "완료"로 믿고 건너뛰면 중복과
    # 누락이 조용히 섞인 CSV 가 나온다. 분할 지문을 남겨 두고, 어긋나면
    # 이어 하지 않고 멈춘다 — 틀린 재개보다 처음부터가 싸다.
    manifest = os.path.join(shard_dir, "manifest.json")
    fp = {"n_paths": len(paths), "shards": shards,
          "first": paths[0], "last": paths[-1]}
    if os.path.exists(manifest):
        prev = json.load(open(manifest, encoding="utf-8"))
        if prev != fp:
            raise SystemExit(
                "샤드가 다른 분할로 만들어져 있다 — 이어 하면 중복·누락이 "
                "섞인다.\n  이전: %s\n  지금: %s\n  %s 를 지우고 다시 "
                "실행할 것" % (prev, fp, shard_dir))
    else:
        json.dump(fp, open(manifest, "w", encoding="utf-8"))
    if verbose:
        print("파일 %d개 · 목록 %.0fs" % (len(paths), time.time() - t0),
              flush=True)

    jobs = [(i, label_dir, paths[i::shards], shard_dir) for i in range(shards)]
    done = 0
    with mp.Pool(procs) as pool:
        for idx, n in pool.imap_unordered(_parse_shard, jobs):
            done += 1
            if verbose:
                print("  샤드 %02d %s · %d/%d · %.0fs"
                      % (idx, "건너뜀" if n < 0 else "%d행" % n,
                         done, shards, time.time() - t0), flush=True)

    parts = []
    for i in range(shards):
        f = os.path.join(shard_dir, "shard_%03d.csv" % i)
        try:
            parts.append(pd.read_csv(f, encoding="utf-8-sig",
                                     low_memory=False))
        except pd.errors.EmptyDataError:
            pass       # 전 파일이 못 읽힌 샤드 — 빈 것도 완료다
    frames = pd.concat(parts, ignore_index=True)
    if verbose:
        print("프레임 %d행 · %.0fs" % (len(frames), time.time() - t0),
              flush=True)
    return frames


def build(label_dir: str, out_dir: str, procs: int = DEFAULT_PROCS,
          shards: int = DEFAULT_SHARDS) -> tuple[str, str]:
    """프레임·클립 CSV 를 만들고 두 경로를 돌려준다."""
    os.makedirs(out_dir, exist_ok=True)
    frames = parse_frames(label_dir, out_dir, procs, shards)
    frames_csv = os.path.join(out_dir, FRAMES_CSV)
    frames.to_csv(frames_csv, index=False, encoding="utf-8-sig")

    clips = parse_aihub.aggregate_71763_clips(frames)
    clips_csv = os.path.join(out_dir, CLIPS_CSV)
    clips.to_csv(clips_csv, index=False, encoding="utf-8-sig")
    print("클립 %d행" % len(clips))

    # 접고 나서 무엇이 비었는지 보이지 않으면 다음 단계에서 조용히 틀린다.
    for c in ("temp_c", "humidity_pct", "ventilation", "date", "chamber",
              "modality", "breath_rate", "evaporation"):
        if c in clips.columns:
            print("  %-14s 값있음 %d / %d"
                  % (c, int(clips[c].notna().sum()), len(clips)))
    print("→ %s\n→ %s" % (frames_csv, clips_csv))
    return frames_csv, clips_csv


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    def _opt(name, default=None):
        if name in args:
            i = args.index(name)
            v = args[i + 1] if i + 1 < len(args) else None
            del args[i:i + 2]
            return v
        return default

    clean = "--clean" in args
    if clean:
        args.remove("--clean")
    out_dir = _opt("--out")
    procs = int(_opt("--procs", DEFAULT_PROCS))
    shards = int(_opt("--shards", DEFAULT_SHARDS))

    label_dir = args[0] if args else None
    if not (label_dir and os.path.isdir(label_dir)):
        print(__doc__.split("실행:")[1].strip())
        return 1
    out_dir = out_dir or os.path.dirname(os.path.abspath(label_dir))

    frames_csv, clips_csv = build(label_dir, out_dir, procs, shards)
    if clean:
        # 최종 CSV 가 둘 다 나온 뒤에만 지운다. 재개 안전망이 먼저다.
        if os.path.exists(frames_csv) and os.path.exists(clips_csv):
            shutil.rmtree(os.path.join(out_dir, SHARD_DIR), ignore_errors=True)
            print("샤드 삭제 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
