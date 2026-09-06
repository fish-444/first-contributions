"""이미지 분석 전용 CLI — 챗봇/벡터DB 없이 이미지 하나만 분석한다.

  - YOLO-World: 이미지 안의 객체를 "구분"(라벨 + 위치 박스)
  - SAM-2:      이미지 안의 객체 영역을 "분리"(마스크 이미지)

사용법:
    export REPLICATE_API_TOKEN="r8_본인토큰"
    python analyze.py 사진.jpg
    python analyze.py 사진.jpg --classes "person, dog, car"
    python analyze.py 사진.jpg --outdir masks   # 마스크 이미지를 masks/ 폴더에 저장

REPLICATE_API_TOKEN 이 필요하다 (https://replicate.com/account/api-tokens).
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import image_analysis


def download(url: str, dest: Path) -> bool:
    """마스크 이미지 URL을 로컬 파일로 내려받는다."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  (다운로드 실패: {dest.name} — {e})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="이미지 분석 (YOLO-World 구분 + SAM-2 분리)"
    )
    parser.add_argument("image", help="분석할 이미지 파일 경로")
    parser.add_argument(
        "--classes",
        default="",
        help="탐지할 클래스(쉼표 구분). 비우면 기본 클래스 사용",
    )
    parser.add_argument(
        "--outdir",
        default="",
        help="마스크 이미지를 저장할 폴더(지정 시에만 저장)",
    )
    args = parser.parse_args()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print(
            "오류: REPLICATE_API_TOKEN 환경변수가 설정되어 있지 않습니다.\n"
            "  https://replicate.com/account/api-tokens 에서 토큰을 발급받아\n"
            '  export REPLICATE_API_TOKEN="r8_..." 로 등록하세요.',
            file=sys.stderr,
        )
        return 1

    path = Path(args.image)
    if not path.is_file():
        print(f"오류: 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        return 1

    image_bytes = path.read_bytes()
    print(f"분석 시작: {path.name} (Replicate 클라우드 실행, 수십 초 소요)\n")

    result = image_analysis.analyze_image(
        image_bytes, path.name, class_names=args.classes.strip() or None
    )

    # --- 요약 ---
    print("=" * 60)
    print(result["description"])
    print("=" * 60)

    # --- YOLO-World: 객체 구분 ---
    detections = result["detections"]
    print(f"\n[YOLO-World] 구분된 객체 {len(detections)}개")
    if detections:
        print(f"{'클래스':<16}{'신뢰도':>8}   박스 [x0, y0, x1, y1]")
        print("-" * 60)
        for d in detections:
            box = ", ".join(str(v) for v in d["box"])
            print(f"{d['cls']:<16}{d['score']:>8.3f}   [{box}]")
    else:
        print("  (지정한 클래스의 객체를 찾지 못했습니다)")

    # --- SAM-2: 영역 분리 ---
    seg = result["segmentation"]
    print(f"\n[SAM-2] 분리된 영역(마스크) {seg['mask_count']}개")
    print(f"  전체 마스크: {seg['combined_mask'] or '(없음)'}")
    for i, m in enumerate(seg["individual_masks"]):
        print(f"  마스크 {i}: {m}")

    # --- 마스크 저장(옵션) ---
    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n마스크 이미지를 '{outdir}/' 에 저장합니다…")
        if seg["combined_mask"]:
            download(seg["combined_mask"], outdir / "combined_mask.png")
        for i, m in enumerate(seg["individual_masks"]):
            download(m, outdir / f"mask_{i:02d}.png")
        print("저장 완료.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
