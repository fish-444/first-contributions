# -*- coding: utf-8 -*-
"""호흡수 측정 — 돼지 자신을 기준틀로 삼는다.

왜 따로 있나
------------
복식호흡·개구호흡·빈호흡은 ``CLASSES`` 15종에 **없다**. 행동 분류 헤드로는
원리적으로 못 잡는다. 대신 호흡은 들숨-날숨이 되풀이되는 주기 운동이라
그 빈도로 잰다.

무엇을 신호로 보나
------------------
배경은 안 쓴다. 처음에는 돼지 ROI 와 배경 ROI 의 스펙트럼을 견줘 카메라
흔들림을 걸렀는데, 손각대 영상 6건 중 4건이 그 관문에서 떨어졌다 — 배경이
흔들리면 돼지도 같이 흔들려서 두 스펙트럼이 붙어버린다. 배경은 기준이 못 된다.

밝기 변화도 안 쓴다. 합성 관통 검증에서 카메라를 고정하면 60.0 bpm 을 CV 0.00
으로 완벽히 되찾았지만, 흔들림을 6 화소만 얹으면 전부 무너졌다. 정합하고 남는
잔여 어긋남이 밝기를 크게 흔들기 때문이다.

그래서 **호흡을 있는 그대로** 잰다.

1. 검출된 돼지 박스를 위상상관으로 따라가며 그 돼지에 정합한다.
2. 몸통을 짧은 축(등-배 방향)을 가로질러 두 쪽으로 가른다. 호흡 팽창이
   그 방향으로 일어난다.
3. 두 쪽의 변위를 기준 프레임 대비로 각각 재고 **뺀다**. 들숨이면 등쪽과
   배쪽이 서로 멀어지고 날숨이면 가까워진다. 카메라 흔들림은 두 쪽에 똑같이
   실려 있으므로 이 차분에서 상쇄된다.
4. 그 벌어짐 신호에서 들숨-날숨 주기를 직접 센다.

무엇으로 참을 가리나
--------------------
**주기 간격의 변동계수(CV)** 다. 합성 검증에서 진짜 호흡은 0.00~0.02 로
일정한데, 호흡 없는 대조군은 **환경에 따라 움직인다** — 케글 조합
(cv2 4.x·numpy 2.0)에선 0.49~0.53, cv2 5.0·numpy 2.4 에선 0.31~0.33 까지
내려왔다. 그래서 컷(기본 0.2)은 두 환경 모두에서 진짜(≤0.02)와
잡음(≥0.31) 사이에 놓았다. 처음 컷 0.35 는 케글 잡음만 보고 정한 값이라
새 환경에서 잡음을 통과시켰다 — 합성 검증이 잡아냈다.

스펙트럼 돌출도와 등-배 결맞음도 같이 기록하지만 판별자로는 못 쓴다 —
돌출도는 진짜 1.8~3.3 대 잡음 1.3~2.7 로 겹치고, 결맞음은 진짜 호흡도
0.61 까지 내려간 적이 있다.

판정에 대하여
-------------
이 모듈은 **bpm 을 재기만 한다. 정상/빈호흡을 스스로 판정하지 않는다.**
품종·일령·기온·측정 자세에 따라 정상 범위가 달라서 고정 임계값은 근거가 없다.
프로젝트의 다른 헤드와 같은 방식 — 같은 방 같은 카메라의 자기 기준선을 모으고
그 편차(z)로 보는 것 — 을 쓸 것. :func:`zscore` 가 그 계산이다.

사용법
------
    from pig_behavior import PigBehaviorPredictor
    from pig_behavior.respiration import RespirationMeter

    p = PigBehaviorPredictor(onnx='assets/onnx/end2end.onnx')
    m = RespirationMeter(p)
    r = m.measure('room3_0300.mp4')
    print(r.bpm if r.usable else ('못 잼: ' + r.reason))
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

#: 탐색 대역 (breaths per minute). 0.2 ~ 2.5 Hz.
#: 실제 하한은 영상 길이에 따라 :data:`MIN_CYCLES` 로 더 올라간다.
BAND = (12.0, 150.0)

#: 관측창 안에 최소 몇 주기가 들어와야 "주기"로 인정할지.
#: 8 초 영상이면 하한이 30 bpm 으로 올라간다 — 그보다 느린 호흡은 못 잰다.
#: 이 관문이 없으면 느린 드리프트가 전부 대역 하한에 봉우리로 찍힌다.
MIN_CYCLES = 4.0


@dataclass
class RespirationResult:
    """측정 한 건."""

    bpm: Optional[float] = None            #: 대표값 — 들숨-날숨 주기 계수 결과
    spectrum_bpm: Optional[float] = None   #: 교차 확인용 — 스펙트럼 봉우리
    snr: Optional[float] = None            #: 돌출도 (진단용, 판별자 아님)
    cycle_cv: Optional[float] = None       #: 주기 간격 변동계수 — 실질 판별자
    n_cycles: int = 0
    coherence: Optional[float] = None      #: 등-배 결맞음 (진단용)
    n_segments: int = 0
    halves_bpm: Tuple[Optional[float], Optional[float]] = (None, None)
    halves_gap: Optional[float] = None
    usable: bool = False
    reason: str = ''
    frames: int = 0
    fps: float = 0.0
    seconds: float = 0.0
    box: Optional[tuple] = None
    label: Optional[str] = None
    score: Optional[float] = None
    track_px: Optional[float] = None
    spectrum: Optional[np.ndarray] = field(default=None, repr=False)
    freqs: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        def r2(v, n=1):
            return None if v is None else round(float(v), n)

        return {
            'bpm': r2(self.bpm),
            'spectrum_bpm': r2(self.spectrum_bpm),
            'snr': r2(self.snr, 2),
            'cycle_cv': r2(self.cycle_cv, 3),
            'n_cycles': self.n_cycles,
            'coherence': r2(self.coherence, 3),
            'n_segments': self.n_segments,
            'halves_bpm': [r2(v) for v in self.halves_bpm],
            'halves_gap': r2(self.halves_gap),
            'usable': self.usable,
            'reason': self.reason,
            'frames': self.frames,
            'fps': r2(self.fps, 2),
            'seconds': r2(self.seconds),
            'box': None if self.box is None else [r2(v) for v in self.box],
            'label': self.label,
            'score': r2(self.score, 3),
            'track_px': r2(self.track_px),
        }


def _detrend(x) -> np.ndarray:
    """평균과 1차 추세를 뺀다. 조명 변화·서서히 도는 카메라를 없앤다."""
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x), dtype=np.float64)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def _continuum(p: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """스펙트럼의 완만한 바닥(연속선)을 추정한다.

    호흡 신호는 1/f 드리프트 위에 얹혀 있다. 전역 중앙값으로 재면 드리프트가
    몰린 저주파가 항상 이긴다. 국소 바닥 대비로 봐야 좁고 뾰족한 봉우리를
    고를 수 있다.
    """
    k = max(5, int(len(p) * frac) | 1)
    q = np.pad(p, k // 2, mode='edge')
    return np.convolve(q, np.ones(k) / k, mode='valid')[:len(p)]


def _spectrum(x, fps: float, band=BAND):
    """(봉우리 bpm, 돌출도, 주파수축, 정규화 스펙트럼). 모자라면 전부 None."""
    if x is None or len(x) < 48:
        return None, None, None, None
    s = _detrend(x)
    if not np.any(s):
        return None, None, None, None
    s = s * np.hanning(len(s))
    n = max(2048, 8 * len(s))
    spec = np.abs(np.fft.rfft(s, n=n))
    bpm = np.fft.rfftfreq(n, 1.0 / fps) * 60.0
    m = (bpm >= band[0]) & (bpm <= band[1])
    if m.sum() < 16:
        return None, None, None, None
    f, p = bpm[m], spec[m]
    norm = p / (_continuum(p) + 1e-12)
    k = int(np.argmax(norm))
    return float(f[k]), float(norm[k]), f, norm


def _coherence(a, b, fps: float, nperseg: int):
    """두 신호의 크기제곱결맞음(MSC)을 Welch 방식으로 낸다.

    Returns
    -------
    (freq_bpm, msc, n_segments). 구간이 4 개 미만이면 (None, None, n).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    starts = list(range(0, len(a) - nperseg + 1, max(1, nperseg // 2)))
    if len(starts) < 4:
        return None, None, len(starts)
    win = np.hanning(nperseg)
    pxx = pyy = pxy = 0.0
    for s in starts:
        A = np.fft.rfft(_detrend(a[s:s + nperseg]) * win)
        B = np.fft.rfft(_detrend(b[s:s + nperseg]) * win)
        pxx = pxx + np.abs(A) ** 2
        pyy = pyy + np.abs(B) ** 2
        pxy = pxy + A * np.conj(B)
    msc = (np.abs(pxy) ** 2) / (pxx * pyy + 1e-20)
    return np.fft.rfftfreq(nperseg, 1.0 / fps) * 60.0, msc, len(starts)


def _bandpass(x, fps: float, band) -> np.ndarray:
    """대역 밖을 잘라낸다. FFT 로 통과대역만 남기고 되돌린다."""
    s = _detrend(x)
    n = len(s)
    F = np.fft.rfft(s)
    bpm = np.fft.rfftfreq(n, 1.0 / fps) * 60.0
    F[(bpm < band[0]) | (bpm > band[1])] = 0
    return np.fft.irfft(F, n=n)


def count_cycles(x, fps: float, band=BAND):
    """들숨-날숨 주기를 시간영역에서 직접 센다.

    주파수 분석은 관측창이 길어야 한다 — 8 초 클립에서는 Welch 구간을 4 개도
    못 만든다. 반면 주기를 직접 세는 것은 서너 번만 숨쉬면 된다. 신호를
    대역통과시킨 뒤 **위로 지나는 영점교차**를 들숨 시작으로 보고 그 간격에서
    bpm 을 낸다.

    규칙적인 호흡이면 간격이 고르고 잡음이면 들쭉날쭉하다. 그래서 간격의
    변동계수(CV)를 같이 낸다 — 이 모듈의 실질적인 판별자다.

    Returns
    -------
    (bpm, cv, n_cycles). 주기가 3 개 미만이면 (None, None, n).
    """
    s = _bandpass(x, fps, band)
    if not np.any(s):
        return None, None, 0
    s = s / (np.std(s) + 1e-12)

    armed = True
    idx = []
    for i in range(1, len(s)):
        if armed and s[i - 1] <= 0.0 < s[i]:
            # 선형보간으로 소수 위치까지
            idx.append(i - 1 + (0.0 - s[i - 1]) / (s[i] - s[i - 1] + 1e-12))
            armed = False
        elif not armed and s[i] < -0.25:      # 잔떨림으로 여러 번 세지 않게
            armed = True
    if len(idx) < 4:                          # 교차 4 개 = 주기 3 개
        return None, None, max(0, len(idx) - 1)

    iv = np.diff(np.asarray(idx)) / fps       # 주기 길이(초)
    iv = iv[iv > 0]
    if len(iv) < 3:
        return None, None, len(iv)
    bpm = 60.0 / float(np.median(iv))
    cv = float(np.std(iv, ddof=1) / (np.mean(iv) + 1e-12))
    return bpm, cv, len(iv)


class RespirationMeter:
    """돼지 기준 호흡수 측정기.

    Parameters
    ----------
    predictor:
        ``PigBehaviorPredictor``. 대상 개체를 찾고 마스크를 얻는 데 쓴다.
        ``box`` 를 직접 넘기면 없어도 된다.
    max_cycle_cv:
        들숨-날숨 주기 간격의 변동계수 상한. **이 모듈의 실질적인 판별자다.**
        합성 검증에서 진짜 호흡 0.00~0.01, 잡음 0.49~0.53 으로 갈렸다.
    min_cycles:
        최소 몇 주기를 세야 인정할지.
    min_coherence:
        등-배 결맞음의 참고 문턱. **기록만 하고 기각하지 않는다** — 진짜
        호흡도 0.61 까지 내려간 적이 있고, cv2 5.0·numpy 2.4 에서는 8초
        진짜 클립 0.06 vs 무호흡 잡음 0.89 로 역전까지 됐다.
    min_snr:
        돌출도 하한. 판별력이 약해 **기본값 0.0 = 꺼짐**이고 기록만 한다
        (진짜 1.8~3.3 대 잡음 1.3~2.7 로 겹친다).
    max_bpm_gap:
        주기 계수와 스펙트럼 봉우리가 이보다 벌어지면 불일치로 기록한다.
        기각하지는 않는다 — 짧은 클립은 스펙트럼 분해능이 모자라기 때문이다.
    patch_w:
        정합·신호추출에 쓸 패치 가로 화소. 작을수록 빠르다.
    """

    def __init__(self, predictor=None, max_cycle_cv: float = 0.2,
                 min_cycles: int = 3, min_coherence: float = 0.40,
                 min_snr: float = 0.0, max_bpm_gap: float = 8.0,
                 patch_w: int = 160, band: Tuple[float, float] = BAND) -> None:
        self.predictor = predictor
        self.max_cycle_cv = max_cycle_cv
        self.min_cycles = min_cycles
        self.min_coherence = min_coherence
        self.min_snr = min_snr
        self.max_bpm_gap = max_bpm_gap
        self.patch_w = patch_w
        self.band = band

    # ------------------------------------------------------------------ 대상 선정
    def pick_target(self, frame_rgb: np.ndarray,
                    prefer: Sequence[str] = ('Resting', 'Lying')):
        """호흡을 재기 좋은 개체 하나를 고른다.

        누워서 제자리에 오래 있는 큰 개체가 좋다. ``prefer`` 라벨을 우선하고
        없으면 가장 큰 검출을 쓴다.
        """
        if self.predictor is None:
            raise ValueError('predictor 없이 pick_target 을 쓸 수 없다. box 를 직접 넘길 것')
        dets = self.predictor.predict(frame_rgb, with_mask=True)
        if not dets:
            return None

        def area(d):
            x1, y1, x2, y2 = d.bbox
            return (x2 - x1) * (y2 - y1)

        return max([d for d in dets if d.label in prefer] or dets, key=area)

    # -------------------------------------------------------------------- 측정
    def measure(self, video: Union[str, os.PathLike, None] = None,
                roi: Optional[Sequence[int]] = None,
                box: Optional[Sequence[float]] = None,
                mask: Optional[np.ndarray] = None,
                seed_frame: int = 0,
                max_sec: float = 30.0,
                frames: Optional[Sequence[np.ndarray]] = None,
                fps_override: Optional[float] = None) -> RespirationResult:
        """영상 하나에서 호흡수를 잰다.

        Parameters
        ----------
        roi:
            ``(x, y, w, h)``. 화면 녹화처럼 영상 밖 UI 가 붙어 있을 때 잘라낼 영역.
        box, mask:
            대상 개체를 직접 지정할 때. 생략하면 ``seed_frame`` 에서 골라낸다.
        max_sec:
            앞에서부터 몇 초까지 볼지.
        frames, fps_override:
            이미 디코드된 BGR 프레임 목록. 주면 ``video`` 대신 이걸 쓴다.
            코덱 손실 없이 시험하거나 이미 읽어 둔 스트림을 넘길 때.
        """
        import cv2

        def crop(fr):
            if roi is None:
                return fr
            x, y, w, h = roi
            return fr[y:y + h, x:x + w]

        mem = frames is not None
        if mem:
            cap = None
            fps = float(fps_override or 30.0)
            n = min(len(frames), int(max_sec * fps))
        else:
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(max_sec * fps))

        def release():
            if cap is not None:
                cap.release()

        label = score = None
        if box is None:
            if mem:
                ok = seed_frame < len(frames)
                seed = frames[seed_frame] if ok else None
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(seed_frame))
                ok, seed = cap.read()
            if not ok:
                release()
                return RespirationResult(reason='시드 프레임을 못 읽었다')
            det = self.pick_target(crop(seed)[:, :, ::-1])
            if det is None:
                release()
                return RespirationResult(reason='개체 검출 0')
            box, mask, label, score = det.bbox, det.mask, det.label, det.score

        x1, y1, x2, y2 = [int(v) for v in box]
        # 정합용 패치는 박스를 조금 넓혀 잡는다 (돼지가 화면 안에서 조금 움직인다)
        pad_x, pad_y = int(0.12 * (x2 - x1)), int(0.12 * (y2 - y1))
        px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        bw, bh = (x2 + pad_x) - px1, (y2 + pad_y) - py1
        bx = tuple(float(v) for v in box)
        if bw < 24 or bh < 24:
            release()
            return RespirationResult(reason='박스가 너무 작다', box=bx)

        sc = self.patch_w / float(bw)
        pw, ph = self.patch_w, max(8, int(bh * sc))

        # 마스크를 패치 좌표로 옮긴다. 마스크가 없으면 박스 안쪽 60% 를 몸통으로 본다.
        if mask is not None:
            mroi = mask[py1:py1 + bh, px1:px1 + bw].astype(np.uint8)
            body = cv2.resize(mroi, (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            body = np.zeros((ph, pw), bool)
            body[int(.2 * ph):int(.8 * ph), int(.2 * pw):int(.8 * pw)] = True
        if body.sum() < 64:
            release()
            return RespirationResult(reason='마스크 화소가 너무 적다', box=bx)

        # ---- 돼지에 정합하며 패치를 모은다 ---------------------------------
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        win = cv2.createHanningWindow((pw, ph), cv2.CV_32F)
        ox = oy = 0.0
        prev = None
        stack, shifts = [], []
        H = W = None
        for i in range(n):
            if mem:
                ok, fr = True, frames[i]
            else:
                ok, fr = cap.read()
            if not ok:
                break
            fr = crop(fr)
            if H is None:
                H, W = fr.shape[:2]
            # 누적 이동량만큼 따라간 자리에서 패치를 뜬다 = 돼지를 기준으로 본다.
            # 정수 자르기만 하면 ±0.5 화소가 남으므로 소수부는 warpAffine 으로 되돌린다.
            ax = float(np.clip(px1 + ox, 0, max(0, W - bw)))
            ay = float(np.clip(py1 + oy, 0, max(0, H - bh)))
            ix, iy = int(np.floor(ax)), int(np.floor(ay))
            patch = fr[iy:iy + bh, ix:ix + bw]
            if patch.shape[0] < 8 or patch.shape[1] < 8:
                break
            sub = np.float32([[1, 0, -(ax - ix)], [0, 1, -(ay - iy)]])
            patch = cv2.warpAffine(patch, sub, (patch.shape[1], patch.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            g = cv2.cvtColor(cv2.resize(patch, (pw, ph)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(prev * win, g * win)
                ox += dx / sc
                oy += dy / sc
                shifts.append((dx / sc, dy / sc))
            prev = g
            stack.append(g)
        release()

        if len(stack) < 48:
            return RespirationResult(reason='프레임이 모자란다 (48 미만)',
                                     frames=len(stack), fps=fps, box=bx,
                                     label=label, score=score)

        M = np.stack(stack)                                  # (T, ph, pw)
        track_px = float(np.abs(np.asarray(shifts)).sum()) if shifts else 0.0
        seconds = M.shape[0] / fps

        # 관측창이 짧으면 낮은 주파수는 못 잰다. 창 안에 최소 MIN_CYCLES 번은
        # 돌아야 주기라고 부를 수 있어서 대역 하한을 영상 길이로 정한다.
        lo = max(self.band[0], MIN_CYCLES * 60.0 / seconds)
        band = (lo, self.band[1])
        common = dict(frames=M.shape[0], fps=fps, seconds=seconds, box=bx,
                      label=label, score=score, track_px=track_px)
        if lo >= self.band[1]:
            return RespirationResult(
                reason='영상이 %.1f 초뿐이라 잴 수 있는 대역이 없다' % seconds, **common)

        # ---- 신호: 등쪽/배쪽 절반의 상대 변위 ------------------------------
        ys, xs = np.nonzero(body)
        long_x = (xs.max() - xs.min()) >= (ys.max() - ys.min())
        # 짧은 축(등-배 방향)을 가로질러 자른다. 호흡 팽창이 그 방향이다.
        if long_x:
            mid = (ys.min() + ys.max()) // 2
            sl_a = (slice(ys.min(), mid), slice(xs.min(), xs.max() + 1))
            sl_b = (slice(mid, ys.max() + 1), slice(xs.min(), xs.max() + 1))
            axis = 1                                   # 변위의 y 성분
        else:
            mid = (xs.min() + xs.max()) // 2
            sl_a = (slice(ys.min(), ys.max() + 1), slice(xs.min(), mid))
            sl_b = (slice(ys.min(), ys.max() + 1), slice(mid, xs.max() + 1))
            axis = 0                                   # 변위의 x 성분

        def half_disp(sl):
            """기준 프레임 대비 그 절반의 변위 시계열. 누적하지 않는다."""
            ref = np.ascontiguousarray(M[0][sl])
            if ref.shape[0] < 8 or ref.shape[1] < 8:
                return None
            wn = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
            ref = ref * wn
            out = np.empty(M.shape[0], dtype=np.float64)
            for t in range(M.shape[0]):
                d, _ = cv2.phaseCorrelate(ref, np.ascontiguousarray(M[t][sl]) * wn)
                out[t] = d[axis]
            return out

        da, db = half_disp(sl_a), half_disp(sl_b)
        if da is None or db is None:
            return RespirationResult(reason='몸통을 두 구획으로 가르지 못했다', **common)

        # 들숨이면 벌어지고 날숨이면 좁아진다. 카메라 흔들림은 두 쪽에 똑같이
        # 실려 있으므로 이 차분에서 상쇄된다.
        sep = da - db
        spec_bpm, snr, freqs, spec = _spectrum(sep, fps, band)
        a_bpm = _spectrum(da, fps, band)[0]
        b_bpm = _spectrum(db, fps, band)[0]
        gap = None if (a_bpm is None or b_bpm is None) else abs(a_bpm - b_bpm)

        coh, nseg = None, 0
        if spec_bpm is not None:
            cf, msc, nseg = _coherence(da, db, fps, int(np.clip(M.shape[0] // 4, 32, 512)))
            if cf is not None:
                inband = (cf >= band[0]) & (cf <= band[1])
                if inband.any():
                    j = int(np.argmin(np.abs(cf[inband] - spec_bpm)))
                    coh = float(msc[inband][j])

        cyc_bpm, cyc_cv, n_cyc = count_cycles(sep, fps, band)

        res = dict(bpm=cyc_bpm, spectrum_bpm=spec_bpm, snr=snr,
                   cycle_cv=cyc_cv, n_cycles=n_cyc,
                   coherence=coh, n_segments=nseg,
                   halves_bpm=(a_bpm, b_bpm), halves_gap=gap,
                   spectrum=spec, freqs=freqs, **common)

        # ---- 인정할지 결정한다 (판단 재료가 전부 돼지 안에 있다) -------------
        if n_cyc < self.min_cycles or cyc_bpm is None:
            return RespirationResult(
                reason='들숨-날숨을 %d 주기밖에 못 셌다 (%d 이상 필요). 영상이 %.1f 초로 짧다'
                       % (n_cyc, self.min_cycles, seconds), **res)
        if cyc_bpm <= lo * 1.05:
            return RespirationResult(
                reason='%.1f bpm 이 대역 하한 %.1f 에 붙어 있다 — 호흡이 아니라 '
                       '드리프트일 공산이 크다. 더 긴 영상이 필요하다' % (cyc_bpm, lo), **res)
        if cyc_cv > self.max_cycle_cv:
            return RespirationResult(
                reason='주기 간격이 고르지 않다 — CV %.2f > %.2f. 규칙적인 호흡이 아니라 잡음이다'
                       % (cyc_cv, self.max_cycle_cv), **res)
        # 결맞음은 기록만 한다 — 본문 그대로 판별자로 못 쓴다. cv2 5.0·
        # numpy 2.4 에서 8초 진짜 클립이 0.06, 무호흡 잡음이 0.89 로
        # **역전**됐다. 문턱 이하는 reason 에 남기되 기각하지 않는다.
        coh_note = ('' if coh is None or coh >= self.min_coherence
                    else ' · 결맞음 %.2f < %.2f (기록만 — 판별자 아님)'
                         % (coh, self.min_coherence))
        if self.min_snr > 0 and snr is not None and snr < self.min_snr:
            return RespirationResult(
                reason='돌출도 %.1f < %.1f' % (snr, self.min_snr), **res)

        agree = spec_bpm is not None and abs(cyc_bpm - spec_bpm) <= self.max_bpm_gap
        return RespirationResult(
            usable=True,
            reason=('주기 %d 회, %.1f bpm, CV %.2f, 결맞음 %s, 스펙트럼 %s'
                    + coh_note)
                   % (n_cyc, cyc_bpm, cyc_cv,
                      '%.2f' % coh if coh is not None else '구간부족',
                      ('%.1f bpm 일치' % spec_bpm) if agree else
                      (('%.1f bpm — 불일치, 분해능 부족일 수 있다' % spec_bpm)
                       if spec_bpm is not None else '없음')),
            **res)


def zscore(bpm: float, baseline: Sequence[float]) -> Optional[float]:
    """자기 기준선 대비 편차.

    고정 임계값(예: "50 bpm 넘으면 빈호흡")은 품종·일령·기온·자세에 따라
    달라져서 근거가 없다. 같은 방 같은 카메라에서 모은 평상시 bpm 을
    ``baseline`` 으로 주고 그 편차로 볼 것. 표본이 5 미만이면 None 을 낸다.
    """
    b = np.asarray([v for v in baseline if v is not None], dtype=float)
    if len(b) < 5:
        return None
    sd = b.std(ddof=1)
    if sd < 1e-6:
        return None
    return float((bpm - b.mean()) / sd)
