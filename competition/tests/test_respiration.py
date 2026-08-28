# -*- coding: utf-8 -*-
"""호흡수 측정의 합성 관통 검증.

이 테스트가 :mod:`pig_behavior.respiration` 의 유일한 근거다. 실제 영상에는
정답이 없으므로, 정답을 아는 호흡을 만들어 넣고 되찾는지 본다. 손대기 전에
반드시 여기부터 돌릴 것 — 지표를 바꾸면 이 표가 먼저 무너진다.

합성 영상을 만들 때 지킨 것
---------------------------
- **보간을 쓰지 않는다.** ``cv2.resize`` 로 몸을 늘였다 줄이면 보간 강도가
  |변형| 에 비례해서 **2 배 주파수** 성분이 생긴다. 실제로 45 bpm 을 넣고
  90 bpm 이 나오는 걸 확인했다. 그래서 정수 화소 이동만 쓴다.
- **강체 이동이 아니라 몸 안의 변형이다.** 통째로 움직이면 정합기가 지워버린다.
  흉부는 위로, 복부는 아래로 벌어지게 해야 실제 호흡과 성질이 같다.
- **카메라 흔들림을 얹는다.** 흔들림 0 이면 어떤 방법이든 통과한다. 손각대
  영상을 흉내내야 의미가 있다.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from pig_behavior.respiration import RespirationMeter, count_cycles  # noqa: E402

FPS, W, H = 30.0, 640, 480
CX, CY = 320, 240
BOX = (170, 155, 470, 325)


def _texture():
    """돼지 자리에 넣을 잔무늬. 결정적이어야 해서 난수 씨앗을 고정한다."""
    rng = np.random.default_rng(3)
    t = rng.integers(60, 200, (150, 300, 3)).astype(np.uint8)
    return cv2.GaussianBlur(t, (0, 0), 1.5)


def _background():
    rng = np.random.default_rng(7)
    return cv2.GaussianBlur(rng.integers(40, 90, (H, W, 3)).astype(np.uint8), (0, 0), 3)


def synth(bpm, amp, shake, sec):
    """정답 bpm 으로 흉복부가 벌어졌다 좁아지는 프레임 목록."""
    tex, bg = _texture(), _background()
    top, bot = tex[:75], tex[75:]
    out, sx, sy = [], 0.0, 0.0
    rng = np.random.default_rng(11)
    for i in range(int(sec * FPS)):
        d = int(round(amp * np.sin(2 * np.pi * (bpm / 60.0) * i / FPS)))
        c = bg.copy()
        c[CY - 75 - d:CY - d, CX - 150:CX + 150] = top      # 흉부: 위로
        c[CY + d:CY + 75 + d, CX - 150:CX + 150] = bot      # 복부: 아래로
        if d > 0:                                            # 벌어진 틈은 경계행으로
            c[CY - d:CY + d, CX - 150:CX + 150] = tex[75:76]
        sx = 0.92 * sx + rng.normal(0, shake * 0.3)          # 손각대 흔들림
        sy = 0.92 * sy + rng.normal(0, shake * 0.3)
        out.append(cv2.warpAffine(c, np.float32([[1, 0, sx], [0, 1, sy]]), (W, H),
                                  borderMode=cv2.BORDER_REFLECT))
    return out


def measure(bpm, amp, shake, sec):
    m = RespirationMeter()
    return m.measure(box=BOX, frames=synth(bpm, amp, shake, sec),
                     fps_override=FPS, max_sec=sec)


# --------------------------------------------------------------- 되찾는가
@pytest.mark.parametrize('bpm,shake,sec', [
    (45.0, 6.0, 20.0),
    (36.0, 6.0, 20.0),
    (72.0, 6.0, 20.0),
    (60.0, 0.0, 20.0),
    (45.0, 6.0, 8.0),      # 짧은 클립 — 스펙트럼은 틀리고 주기 계수만 맞는다
])
def test_recovers_known_rate(bpm, shake, sec):
    r = measure(bpm, amp=4, shake=shake, sec=sec)
    assert r.usable, r.reason
    assert abs(r.bpm - bpm) < 5.0, '정답 %.0f, 측정 %.1f' % (bpm, r.bpm)
    assert r.cycle_cv < 0.1, '진짜 호흡인데 CV 가 %.2f 나 된다' % r.cycle_cv


# --------------------------------------------------------------- 안 속는가
@pytest.mark.parametrize('sec', [20.0, 8.0])
def test_rejects_when_no_breathing(sec):
    """흔들리기만 하고 호흡은 없는 영상. 반드시 기각돼야 한다."""
    r = measure(45.0, amp=0, shake=6.0, sec=sec)
    assert not r.usable
    # 잡음 CV 는 환경에 따라 0.31(cv2 5.0)~0.53(케글) — 컷(0.2) 위면 된다
    assert r.cycle_cv > 0.2, 'CV %.2f — 잡음인데 너무 규칙적으로 보인다' % r.cycle_cv


def test_cv_separates_signal_from_noise():
    """CV 가 판별자인 근거. 진짜와 잡음 사이에 여유가 있어야 한다."""
    real = measure(45.0, amp=4, shake=6.0, sec=20.0).cycle_cv
    noise = measure(45.0, amp=0, shake=6.0, sec=20.0).cycle_cv
    assert real < 0.1 < 0.2 < noise, '진짜 %.2f, 잡음 %.2f' % (real, noise)


def test_too_short_is_refused_not_guessed():
    """주기를 3 번도 못 세면 숫자를 지어내지 말고 기각해야 한다."""
    r = measure(20.0, amp=4, shake=0.0, sec=3.0)
    assert not r.usable
    assert r.reason


# --------------------------------------------------------------- 계수기 단위
def test_count_cycles_on_pure_sine():
    t = np.arange(600) / FPS
    bpm, cv, n = count_cycles(np.sin(2 * np.pi * 0.75 * t), FPS, (12.0, 150.0))
    assert abs(bpm - 45.0) < 1.0
    assert cv < 0.01
    assert n >= 10


def test_count_cycles_on_white_noise():
    x = np.random.default_rng(0).normal(size=600)
    bpm, cv, n = count_cycles(x, FPS, (12.0, 150.0))
    assert cv > 0.2, '백색잡음의 CV 가 %.2f 밖에 안 된다' % cv
