# -*- coding: utf-8 -*-
"""영상 한 편 → 요약 레코드. 프로그램에 붙이는 단일 진입점.

main.py 에서 이렇게 쓴다::

    from pig_behavior.analyze import VideoAnalyzer

    an = VideoAnalyzer(onnx='pig_behavior/assets/onnx/end2end.onnx')

    rec = an.analyze('room3_0300.mp4',
                     hut_type='a',                 # 스톨(a)·방목(b)·기타(c)
                     situation='발정')              # 사용자가 기록하는 값
    print(rec['file_name'])                        # 260822a_Resting
    print(rec['pig']['pig_count'], rec['pig']['pig_lable_pose'])
    print(rec['resp']['bpm'], rec['resp']['usable'])

폴더 통째로::

    records = an.analyze_folder('cctv/room3', hut_type='a')

명령줄::

    python -m pig_behavior.analyze cctv/room3 --onnx ...--out room3.json --hut a

무엇을 내는가
-------------
``영상정리`` 규약(vd / pig / 파일명)을 그대로 채우고, 근거를 함께 붙인다.

- ``vd``   : 날짜·길이
- ``pig``  : 사육장 종류·마릿수·최다 행동·상황
- ``resp`` : 호흡수 (:mod:`pig_behavior.respiration`)
- ``_model``: 어떤 근거로 그 값이 나왔는지 — 프레임 수, 라벨 분포, 기각 사유

무엇을 스스로 정하지 않는가
---------------------------
- ``pig_hut_type`` 은 영상만으로 못 정한다. 호출자가 준다.
- ``pig_lable_situtation`` 은 규약상 사용자가 직접 기록한다. 호출자가 준다.
- 호흡의 정상/빈호흡 판정은 하지 않는다. bpm 만 낸다
  (:func:`pig_behavior.respiration.zscore` 로 자기 기준선 대비를 볼 것).

행동 라벨을 믿는 정도
---------------------
검증 200장 기준 bbox mAP 0.205 다. ``RELIABLE_CLASSES``
(Resting·Eating·Walking·Searching) 밖은 근거가 없다. 그래서
``pig_lable_pose`` 는 그 4종 안에서의 최다값을 쓰고, 규약 문구 그대로의
15종 전체 최다값은 ``pig_lable_pose_raw`` 에 따로 담는다.
"""

from __future__ import annotations

import collections
import json
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .predictor import CLASSES, PigBehaviorPredictor, RELIABLE_CLASSES
from .respiration import RespirationMeter

VIDEO_EXT = ('.mp4', '.avi', '.mkv', '.mov', '.m4v')

#: 사육장 종류 코드 — 규약 그대로.
HUT_TYPES = {'a': '스톨', 'b': '방목', 'c': '기타'}


def find_play_roi(video: str, n_probe: int = 24, pad: int = 8):
    """화면 녹화에서 실제 재생 영역만 골라낸다.

    브라우저·플레이어 UI 는 시간이 지나도 안 변하고 영상 영역만 변한다.
    프레임 간 표준편차가 큰 화소의 최대 연결성분을 재생 영역으로 본다.

    CCTV 원본처럼 화면 전체가 영상이면 쓸 필요 없다 — 그때는 ``None`` 을
    넘기면 된다.

    Returns
    -------
    (x, y, w, h)
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    small = []
    for i in np.linspace(0, max(n - 2, 0), n_probe).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            small.append(cv2.cvtColor(cv2.resize(fr, (0, 0), fx=0.25, fy=0.25),
                                      cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(small) < 3:
        return (0, 0, W, H)

    sd = np.std(np.stack(small), axis=0)
    m = (sd > max(3.0, np.percentile(sd, 75))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if nlab <= 1:
        return (0, 0, W, H)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[k, :4] * 4            # 0.25 배로 줄여 봤으므로 되돌린다
    x, y = max(0, int(x) - pad), max(0, int(y) - pad)
    return (x, y, min(W - x, int(w) + 2 * pad), min(H - y, int(h) + 2 * pad))


def _date_parts(name: str):
    """파일명에서 날짜를 읽는다. ``YYYY-MM-DD`` 또는 ``YYYYMMDD``."""
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', name) or \
        re.search(r'(20\d{2})(\d{2})(\d{2})', name)
    if not m:
        return None, None
    y, mo, d = m.group(1), m.group(2), m.group(3)
    return '%s년도 %s월 %s일' % (y, mo, d), y[2:] + mo + d


class VideoAnalyzer:
    """영상 → 레코드. 모델을 한 번만 올리고 여러 편에 재사용한다.

    Parameters
    ----------
    onnx:
        ``end2end.onnx`` 경로. 권장 경로다 (의존성이 onnxruntime 뿐).
    checkpoint:
        ``.pth`` 경로. ``onnx`` 대신 pytorch 백엔드를 쓸 때.
    n_frames:
        행동 집계에 쓸 표본 프레임 수. CPU 에서 장당 몇 초 걸리니 과하게
        올리지 말 것.
    score_thr:
        이 값 미만 검출은 버린다.
    measure_resp:
        호흡수까지 잴지. 끄면 그만큼 빨라진다.
    resp_max_sec:
        호흡 측정에 볼 최대 길이. 길수록 낮은 호흡수까지 잴 수 있다
        (창 안에 4 주기는 들어와야 한다 — 30 초면 8 bpm 까지).
    """

    def __init__(self, onnx: Optional[str] = None,
                 checkpoint: Optional[str] = None,
                 device: str = 'cpu',
                 n_frames: int = 8,
                 score_thr: float = 0.30,
                 measure_resp: bool = True,
                 resp_max_sec: float = 30.0) -> None:
        self.predictor = PigBehaviorPredictor(
            onnx=onnx, checkpoint=checkpoint, device=device, score_thr=score_thr)
        self.meter = RespirationMeter(self.predictor) if measure_resp else None
        self.n_frames = n_frames
        self.score_thr = score_thr
        self.resp_max_sec = resp_max_sec

    # ------------------------------------------------------------------ 한 편
    def analyze(self, video: str,
                hut_type: str = 'c',
                situation: Optional[str] = None,
                roi: Optional[Sequence[int]] = None,
                auto_roi: bool = False,
                date: Optional[str] = None) -> Dict:
        """영상 한 편을 레코드 하나로 만든다.

        Parameters
        ----------
        hut_type:
            ``'a'`` 스톨 · ``'b'`` 방목 · ``'c'`` 기타. 영상만으로는 못 정한다.
        situation:
            분만·임신·발정·질병 등. 규약상 사용자가 직접 기록하는 값이다.
        roi:
            ``(x, y, w, h)``. 화면 녹화라 UI 를 잘라내야 할 때.
        auto_roi:
            True 면 :func:`find_play_roi` 로 재생 영역을 자동으로 찾는다.
            CCTV 원본에는 쓰지 말 것.
        date:
            ``YYYY-MM-DD``. 생략하면 파일명에서 읽는다.
        """
        import cv2

        if hut_type not in HUT_TYPES:
            raise ValueError('hut_type 은 %s 중 하나여야 한다: %r'
                             % (list(HUT_TYPES), hut_type))

        name = os.path.basename(video)
        if roi is None and auto_roi:
            roi = find_play_roi(video)

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, max(total - 2, 0), self.n_frames).astype(int)

        per_frame, seed_frame, seed_area = [], 0, 0.0
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            if roi is not None:
                x, y, w, h = roi
                fr = fr[y:y + h, x:x + w]
            dets = self.predictor.predict(fr[:, :, ::-1], with_mask=False)
            per_frame.append({'frame': int(i), 'sec': round(i / fps, 2),
                              'n': len(dets),
                              'dets': [d.to_dict() for d in dets]})
            for d in dets:                       # 호흡 측정 시드 = 가장 큰 개체
                x1, y1, x2, y2 = d.bbox
                a = (x2 - x1) * (y2 - y1)
                if a > seed_area:
                    seed_area, seed_frame = a, int(i)
        cap.release()

        labels = [d['label'] for f in per_frame for d in f['dets']]
        counts = collections.Counter(labels)
        counts_rel = collections.Counter(l for l in labels if l in RELIABLE_CLASSES)
        n_per_frame = [f['n'] for f in per_frame] or [0]
        pose = (counts_rel.most_common(1)[0][0] if counts_rel else
                (counts.most_common(1)[0][0] if counts else None))

        resp = None
        if self.meter is not None:
            r = self.meter.measure(video, roi=roi, seed_frame=seed_frame,
                                   max_sec=self.resp_max_sec)
            resp = r.to_dict()

        date_ko, yymmdd = _date_parts(date or name)
        sec = round(total / fps, 1) if fps else 0.0

        return {
            'vd': {
                'vd_source_file': name,
                'vd_date': date_ko,
                'vd_length': '%02d:%05.2f' % (int(sec // 60), sec % 60),
                'vd_length_sec': sec,
            },
            'pig': {
                'pig_hut_type': hut_type,
                'pig_hut_type_name': HUT_TYPES[hut_type],
                'pig_count': int(np.median(n_per_frame)),
                'pig_count_max': int(np.max(n_per_frame)),
                'pig_lable_pose': pose,
                'pig_lable_pose_raw': counts.most_common(1)[0][0] if counts else None,
                'pig_lable_situtation': situation,
            },
            'resp': resp,
            'file_name': '%s%s_%s' % (yymmdd or '000000', hut_type, pose or 'NA'),
            '_model': {
                'backend': self.predictor.backend,
                'score_thr': self.score_thr,
                'frames_scored': len(per_frame),
                'roi_xywh': list(roi) if roi is not None else None,
                'label_counts': dict(counts),
                'label_counts_reliable': dict(counts_rel),
                'resp_seed_frame': seed_frame,
                'caveat': 'bbox mAP 0.205 모델. Resting·Eating·Walking·Searching 만 '
                          '근거가 있다. pig_count 는 표본 프레임 검출 수의 중앙값이지 '
                          '돈방 전체 두수가 아니다.',
            },
            '_per_frame': per_frame,
        }

    # ------------------------------------------------------------------ 폴더
    def analyze_folder(self, folder: str, **kw) -> List[Dict]:
        """폴더 안 영상 전부. 같은 이름이 겹치면 ``_2``, ``_3`` 을 붙인다.

        규약의 파일명(날짜6자리+hut_type+pose)은 같은 날 같은 사육장에서
        여러 편을 찍으면 반드시 겹친다. 조용히 덮어쓰지 않게 여기서 번호를 붙인다.
        """
        paths = sorted(p for p in
                       (os.path.join(folder, f) for f in os.listdir(folder))
                       if p.lower().endswith(VIDEO_EXT))
        out, used = [], collections.Counter()
        for p in paths:
            rec = self.analyze(p, **kw)
            base = rec['file_name']
            used[base] += 1
            if used[base] > 1:
                rec['file_name'] = '%s_%d' % (base, used[base])
                rec['_model']['file_name_collision'] = base
            out.append(rec)
        return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog='python -m pig_behavior.analyze',
        description='영상 → 영상정리 규약 레코드')
    ap.add_argument('inputs', nargs='+', help='영상 파일 또는 폴더')
    ap.add_argument('--onnx', help='end2end.onnx (권장)')
    ap.add_argument('--checkpoint', help='pig_polygon_epoch12.pth')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--hut', default='c', choices=sorted(HUT_TYPES),
                    help='사육장 종류: a 스톨 · b 방목 · c 기타')
    ap.add_argument('--situation', help='분만·임신·발정·질병 등 (사용자 기록)')
    ap.add_argument('--frames', type=int, default=8, help='행동 집계 표본 프레임 수')
    ap.add_argument('--score-thr', type=float, default=0.30)
    ap.add_argument('--no-resp', action='store_true', help='호흡 측정 건너뛰기')
    ap.add_argument('--resp-sec', type=float, default=30.0)
    ap.add_argument('--auto-roi', action='store_true',
                    help='화면 녹화에서 재생 영역만 자동으로 잘라낸다')
    ap.add_argument('--out', help='JSON 출력 경로 (생략하면 표준출력)')
    ap.add_argument('--per-frame', action='store_true',
                    help='프레임별 검출까지 출력에 담는다')
    args = ap.parse_args(argv)

    if not args.onnx and not args.checkpoint:
        ap.error('--onnx 또는 --checkpoint 중 하나는 필요하다')

    an = VideoAnalyzer(onnx=args.onnx, checkpoint=args.checkpoint,
                       device=args.device, n_frames=args.frames,
                       score_thr=args.score_thr, measure_resp=not args.no_resp,
                       resp_max_sec=args.resp_sec)

    kw = dict(hut_type=args.hut, situation=args.situation, auto_roi=args.auto_roi)
    records = []
    for src in args.inputs:
        if os.path.isdir(src):
            records.extend(an.analyze_folder(src, **kw))
        else:
            records.append(an.analyze(src, **kw))

    for r in records:
        if not args.per_frame:
            r.pop('_per_frame', None)
        resp = r.get('resp') or {}
        print('%-40s %s  %2d마리  %-9s %s' % (
            r['vd']['vd_source_file'][:40], r['pig']['pig_hut_type'],
            r['pig']['pig_count'], r['pig']['pig_lable_pose'] or '-',
            ('호흡 %.1f bpm' % resp['bpm']) if resp.get('usable')
            else ('호흡 못 잼: ' + (resp.get('reason') or '측정 안 함'))))

    text = json.dumps(records, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
        print('\n%d 건 → %s' % (len(records), args.out))
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
