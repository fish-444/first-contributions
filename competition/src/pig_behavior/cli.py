# -*- coding: utf-8 -*-
"""명령줄 인터페이스.

    python -m pig_behavior.cli --onnx assets/onnx/end2end.onnx frame.jpg
    python -m pig_behavior.cli --onnx ... --out result.jsonl 폴더/
    python -m pig_behavior.cli --onnx ... --bench frame.jpg
    python -m pig_behavior.cli --checkpoint pig_polygon_epoch12.pth frame.jpg   # pytorch 백엔드
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .predictor import PigBehaviorPredictor

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def collect(paths):
    out = []
    for p in paths:
        p = pathlib.Path(p)
        if p.is_dir():
            out += sorted(f for f in p.rglob('*') if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            out.append(p)
        else:
            print(f'건너뜀(이미지 아님): {p}', file=sys.stderr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='모돈 행동 15종 인스턴스 분할')
    ap.add_argument('inputs', nargs='+', help='이미지 파일 또는 폴더')
    ap.add_argument('--checkpoint', help='pig_polygon_epoch12.pth')
    ap.add_argument('--onnx', help='end2end.onnx (주면 onnx 백엔드)')
    ap.add_argument('--device', default='cpu', help='cpu 또는 cuda:0')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--out', help='JSONL 출력 경로 (생략하면 표준출력)')
    ap.add_argument('--reliable-only', action='store_true',
                    help='검증 AP가 근거로 쓸 만한 4종만 남긴다')
    ap.add_argument('--bench', action='store_true', help='첫 장으로 속도만 측정')
    args = ap.parse_args(argv)

    if not args.checkpoint and not args.onnx:
        ap.error('--checkpoint 또는 --onnx 중 하나는 필요하다')

    files = collect(args.inputs)
    if not files:
        ap.error('처리할 이미지가 없다')

    p = PigBehaviorPredictor(checkpoint=args.checkpoint, onnx=args.onnx,
                             device=args.device, score_thr=args.score_thr)
    print(f'백엔드 {p.backend} · device {p.device} · 이미지 {len(files)}장',
          file=sys.stderr)

    if args.bench:
        sec = p.benchmark(files[0])
        print(f'장당 {sec:.3f}s ({1 / sec:.2f} fps)', file=sys.stderr)
        return 0

    sink = open(args.out, 'w', encoding='utf-8') if args.out else sys.stdout
    try:
        for f in files:
            dets = p.predict(f, with_mask=False)
            if args.reliable_only:
                dets = [d for d in dets if d.reliable]
            rec = {'image': str(f),
                   'detections': [d.to_dict() for d in dets]}
            sink.write(json.dumps(rec, ensure_ascii=False) + '\n')
            sink.flush()
    finally:
        if args.out:
            sink.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
