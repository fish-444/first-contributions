# -*- coding: utf-8 -*-
"""모돈 행동 15종 인스턴스 분할 추론기.

백엔드 두 가지를 지원한다.

- ``pytorch``  : mmdet + 체크포인트(.pth). 검증된 경로. 느리다(CPU 약 5.5초/장).
- ``onnx``     : onnxruntime + end2end.onnx. 변환에 성공한 경우에만.

기본은 ``auto`` — onnx 파일이 주어졌으면 onnx, 아니면 pytorch.

성능에 대한 정직한 기준선 (2026-08-21, 처음 보는 200장, AI Hub 622 ts06):

    bbox mAP 0.205 / segm mAP 0.192

    Resting 0.633 · Eating 0.631 : 쓸 만함
    Walking 0.261 · Searching 0.242 : 약함
    Lying 0.058 · Standing 0.007 · Sitting 0.015 · Running 0.0 · Scrubbing 0.0 : 못 씀
    Suckling · Parturition · Drinking · Urinating · Defecating · Coughing : 검증 불가(표본 0)

원 학습이 보고한 0.953 은 train==val 인 상태에서 잰 값이라 인용하면 안 된다.
따라서 프로그램에 붙일 때는 ``RELIABLE_CLASSES`` 밖의 출력을 그대로 신뢰하지 말 것.
"""

from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

# 체크포인트의 meta['dataset_meta']['classes'] 와 순서가 정확히 같아야 한다.
# 순서가 어긋나면 라벨이 통째로 밀린다. from_checkpoint() 가 실제 값으로 검증한다.
CLASSES = (
    'Scrubbing', 'Searching', 'Lying', 'Resting', 'Suckling',
    'Urinating', 'Defecating', 'Drinking', 'Standing', 'Parturition',
    'Walking', 'Sitting', 'Running', 'Eating', 'Coughing',
)

#: 검증 200장에서 AP 0.2 이상이 나온, 근거를 갖고 쓸 수 있는 클래스.
RELIABLE_CLASSES = frozenset({'Resting', 'Eating', 'Walking', 'Searching'})

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_SCALE = (1333, 800)  # (long, short) — 학습 config 의 Resize 와 동일


@dataclass
class Detection:
    """검출 한 건."""

    label: str
    score: float
    bbox: tuple  # (x1, y1, x2, y2) — 원본 이미지 좌표계
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # bool, 원본 크기

    @property
    def reliable(self) -> bool:
        """이 클래스의 검증 AP가 근거로 쓸 만한 수준인가."""
        return self.label in RELIABLE_CLASSES

    def to_dict(self, with_mask: bool = False) -> dict:
        d = {'label': self.label, 'score': round(float(self.score), 4),
             'bbox': [round(float(v), 1) for v in self.bbox],
             'reliable': self.reliable}
        if with_mask and self.mask is not None:
            d['mask_area'] = int(self.mask.sum())
        return d


def patch_mmdet_mmcv_ceiling() -> None:
    """mmdet 3.3.0 의 ``mmcv < 2.2.0`` 상한을 완화한다.

    Python 3.12 에는 mmcv 2.1.0 휠이 없어 2.2.0 을 쓸 수밖에 없다.
    **mmdet 을 import 하기 전에** 호출해야 한다 (import 시점에 assert 가 돈다).
    이미 완화돼 있거나 mmdet 이 없으면 조용히 넘어간다.
    """
    import importlib.util

    spec = importlib.util.find_spec('mmdet')
    if spec is None or spec.origin is None:
        return
    init = pathlib.Path(spec.origin)
    try:
        src = init.read_text(encoding='utf-8')
    except OSError:
        return
    patched = src.replace("mmcv_maximum_version = '2.2.0'",
                          "mmcv_maximum_version = '2.3.0'")
    if patched != src:
        init.write_text(patched, encoding='utf-8')


def _read_image(src: Union[str, os.PathLike, np.ndarray]) -> np.ndarray:
    """RGB uint8 배열로 통일한다."""
    if isinstance(src, np.ndarray):
        if src.ndim != 3 or src.shape[2] != 3:
            raise ValueError(f'RGB 3채널 배열이어야 한다: {src.shape}')
        return src
    import cv2

    bgr = cv2.imread(str(src))
    if bgr is None:
        raise FileNotFoundError(f'이미지를 못 읽었다: {src}')
    return bgr[:, :, ::-1]


def _preprocess(rgb: np.ndarray):
    """keep-ratio 리사이즈 + normalize. (텐서, 스케일) 반환."""
    import cv2

    h, w = rgb.shape[:2]
    scale = min(_SCALE[0] / max(h, w), _SCALE[1] / min(h, w))
    resized = cv2.resize(rgb, (int(w * scale + 0.5), int(h * scale + 0.5)))
    x = (resized.astype(np.float32) - _MEAN) / _STD
    return x.transpose(2, 0, 1)[None], scale


class PigBehaviorPredictor:
    """행동 분할 추론기.

    Parameters
    ----------
    checkpoint:
        ``pig_polygon_epoch12.pth`` 경로. pytorch 백엔드에 필수.
    onnx:
        ``end2end.onnx`` 경로. 주면 onnx 백엔드를 쓴다.
    device:
        ``'cpu'`` 또는 ``'cuda:0'``. onnx 백엔드는 cpu만 지원한다.
    score_thr:
        이 값 미만의 검출은 버린다.
    """

    def __init__(self, checkpoint: Optional[str] = None,
                 onnx: Optional[str] = None,
                 device: str = 'cpu',
                 score_thr: float = 0.3,
                 backend: str = 'auto') -> None:
        if backend == 'auto':
            backend = 'onnx' if onnx else 'pytorch'
        if backend not in ('onnx', 'pytorch'):
            raise ValueError(f'backend 는 onnx 또는 pytorch: {backend!r}')

        self.backend = backend
        self.device = device
        self.score_thr = score_thr
        self.classes: Sequence[str] = CLASSES

        if backend == 'pytorch':
            if not checkpoint:
                raise ValueError('pytorch 백엔드에는 checkpoint 가 필요하다')
            self._init_pytorch(checkpoint, device)
        else:
            if not onnx:
                raise ValueError('onnx 백엔드에는 onnx 경로가 필요하다')
            self._init_onnx(onnx)

    # ------------------------------------------------------------------ 초기화
    def _init_pytorch(self, checkpoint: str, device: str) -> None:
        patch_mmdet_mmcv_ceiling()
        import torch
        from mmdet.apis import init_detector
        from mmengine.config import Config

        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        classes = tuple(ck['meta']['dataset_meta']['classes'])
        if classes != CLASSES:
            raise RuntimeError(
                '체크포인트의 클래스 순서가 CLASSES 와 다르다. 라벨이 밀린다.\n'
                f'  체크포인트: {classes}\n  CLASSES  : {CLASSES}')
        self.classes = classes

        # 체크포인트 안에 학습 config 전문이 들어 있다. 원본 파일이 없어도 된다.
        cfg_dir = pathlib.Path(checkpoint).parent
        cfg_path = cfg_dir / '_pig_behavior_cfg.py'
        cfg_path.write_text(ck['meta']['cfg'], encoding='utf-8')
        cfg = Config.fromfile(str(cfg_path))
        # 순수 추론에는 GT 가 필요 없다
        cfg.test_pipeline = [t for t in cfg.test_pipeline if t['type'] != 'LoadAnnotations']
        cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline
        del ck

        self._model = init_detector(cfg, checkpoint, device=device)
        self._torch = torch

    def _init_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        try:  # mmdeploy 커스텀 op(NMS·RoIAlign)이 필요한 그래프일 수 있다
            from mmdeploy.backend.onnxruntime import get_ops_path

            lib = get_ops_path()
            if lib and os.path.exists(lib):
                so.register_custom_ops_library(lib)
        except Exception:  # mmdeploy 없이도 표준 op만으로 도는 그래프면 문제없다
            pass
        self._sess = ort.InferenceSession(onnx_path, so,
                                          providers=['CPUExecutionProvider'])
        self._input_name = self._sess.get_inputs()[0].name

    # -------------------------------------------------------------------- 추론
    def predict(self, image: Union[str, os.PathLike, np.ndarray],
                with_mask: bool = True) -> List[Detection]:
        """이미지 한 장에서 검출 목록을 낸다."""
        rgb = _read_image(image)
        if self.backend == 'pytorch':
            return self._predict_pytorch(rgb, with_mask)
        return self._predict_onnx(rgb, with_mask)

    def _predict_pytorch(self, rgb: np.ndarray, with_mask: bool) -> List[Detection]:
        from mmdet.apis import inference_detector

        res = inference_detector(self._model, rgb[:, :, ::-1])  # mmdet 은 BGR
        inst = res.pred_instances
        keep = inst.scores > self.score_thr
        out = []
        scores = inst.scores[keep].cpu().numpy()
        labels = inst.labels[keep].cpu().numpy()
        bboxes = inst.bboxes[keep].cpu().numpy()
        masks = (inst.masks[keep].cpu().numpy()
                 if with_mask and 'masks' in inst else [None] * len(scores))
        for s, l, b, m in zip(scores, labels, bboxes, masks):
            out.append(Detection(self.classes[int(l)], float(s), tuple(b), m))
        return out

    def _predict_onnx(self, rgb: np.ndarray, with_mask: bool) -> List[Detection]:
        """ONNX 그래프의 출력을 원본 좌표계로 되돌린다.

        내보내기 설정이 ``export_postprocess_mask: false`` 라서
        ``masks`` 는 전체 이미지가 아니라 **박스별 28x28 RoI 확률맵**이다
        (assets/onnx/detail.json 로 확인). 전체 크기로 그냥 늘리면 안 되고,
        각 박스 영역에 붙여 넣어야 한다.
        ``dets`` 의 좌표는 리사이즈된 입력 텐서 기준이므로 scale 로 되돌린다.
        """
        import cv2

        h, w = rgb.shape[:2]
        x, scale = _preprocess(rgb)
        outputs = self._sess.run(None, {self._input_name: x})
        dets, labels = outputs[0], outputs[1]
        masks = outputs[2] if len(outputs) > 2 and with_mask else None

        out = []
        for i in range(dets.shape[1]):
            score = float(dets[0, i, 4])
            if score < self.score_thr:
                continue
            x1, y1, x2, y2 = (float(v) / scale for v in dets[0, i, :4])
            bbox = (x1, y1, x2, y2)

            mask = None
            if masks is not None:
                mask = self._paste_mask(masks[0, i], bbox, h, w)
            out.append(Detection(self.classes[int(labels[0, i])], score, bbox, mask))
        return out

    @staticmethod
    def _paste_mask(roi_mask: np.ndarray, bbox, h: int, w: int) -> Optional[np.ndarray]:
        """28x28 RoI 확률맵을 원본 크기 bool 마스크의 박스 자리에 붙인다."""
        import cv2

        x1, y1, x2, y2 = bbox
        cx1, cy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        cx2, cy2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        patch = cv2.resize(roi_mask.astype(np.float32), (cx2 - cx1, cy2 - cy1),
                           interpolation=cv2.INTER_LINEAR)
        full = np.zeros((h, w), dtype=bool)
        full[cy1:cy2, cx1:cx2] = patch > 0.5
        return full

    def predict_batch(self, images: Iterable) -> List[List[Detection]]:
        """여러 장. 내부적으로는 한 장씩 돈다(배치 이득이 크지 않다)."""
        return [self.predict(im) for im in images]

    def benchmark(self, image, n: int = 10, warmup: int = 3) -> float:
        """워밍업 후 장당 평균 초. 전처리 포함."""
        rgb = _read_image(image)
        for _ in range(warmup):
            self.predict(rgb, with_mask=False)
        t0 = time.perf_counter()
        for _ in range(n):
            self.predict(rgb, with_mask=False)
        return (time.perf_counter() - t0) / n
