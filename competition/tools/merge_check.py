"""합류 점검 — 다른 자리에서 온 브랜치를 **합치기 전에** 본다.

로컬 CLI 자리와 이 원격 자리가 각자 앞서 나갔다. 합칠 때 위험한 것은
충돌 자체가 아니라 **조용히 통과하는 것** 셋이다:

1. **유출** — 저장소가 공개다. 저쪽 작업 폴더에는 가중치(`*.pth`,
   `end2end.onnx`)·영상·바이트코드(`*.pyc`)가 같이 있다. 한 번 올라가면
   git 이력에서 회수가 안 된다. `.gitignore` 는 **우리 쪽 규칙**이라
   저쪽에서 이미 커밋해 온 파일은 막지 못한다 — 그래서 여기서 다시 본다.
2. **정본 갈림** — 같은 계산이 양쪽에 하나씩 생기면 수치가 갈린다.
   양쪽이 모두 고친 파일이 그 후보다.
3. **수치 표류** — 합친 뒤 `check_docs` 가 0 이 아니면 문서가 낡은 것이다.

    python competition/tools/merge_check.py origin/<브랜치>     # 합치기 전
    python competition/tools/merge_check.py --after            # 합친 뒤 검증

비밀값 패턴은 `check_docs` 의 것을 그대로 쓴다 — 규칙을 두 벌 두지 않는다.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(HERE)
ROOT = os.path.dirname(COMP)
sys.path.insert(0, HERE)

from check_docs import KEYLIKE  # noqa: E402  비밀값 패턴의 정본

# 공개 저장소에 들어오면 안 되는 것 — 확장자로 막는다(파일명은 새 자료가
# 올 때마다 샌다). `.gitignore` 와 같은 생각이되, 저쪽이 이미 커밋한
# 파일에는 gitignore 가 안 듣기 때문에 여기서 한 번 더 본다.
BLOCKED_EXT = {
    ".pyc": "바이트코드(소스가 정본이다)",
    ".pth": "모델 가중치", ".pt": "모델 가중치", ".onnx": "모델 가중치",
    ".xlsx": "원자료 스프레드시트(농장 식별자)",
    ".mp4": "영상(저작권·식별자)", ".avi": "영상", ".mkv": "영상",
    ".mov": "영상", ".m4v": "영상", ".webm": "영상",
    ".db": "SQLite DB(농장 이름·성적)", ".sqlite": "SQLite DB",
    ".key": "키 파일", ".env": "환경 비밀값",
    ".zip": "압축 원자료", ".tar": "압축 원자료", ".npz": "캐시 배열",
}
BLOCKED_PATH = {
    "data/cctv/": "CCTV 실증 자료(영상·라벨·검출)",
    "__pycache__/": "바이트코드 캐시",
    "data/aihub/": "AI Hub 원자료",
    "data/edinburgh/": "케글 원자료(CC BY-NC)",
}
TEXT_EXT = (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".sh", ".csv")
# 합칠 때 반드시 사람이 보는 자리 — 양쪽이 다 손대는 것이 정해져 있다
HOTSPOTS = {
    "competition/tests/smoke_test.py": "테스트 목록 — 양쪽 추가분을 **둘 다** 남긴다",
    "competition/docs/STATUS.md": "수치 정본 — 합친 뒤 check_docs 가 정한다",
    "competition/docs/CAPABILITIES.md": "기능 목록 — 양쪽 표를 합친다",
    "competition/docs/HANDOFF.md": "인계 — 양쪽 경고를 둘 다 남긴다",
    "competition/README.md": "규모 수치 — check_docs 가 정한다",
}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                          text=True, check=False).stdout.strip()


def divergence(ref: str, ours: str = "HEAD") -> dict:
    """merge-base 기준 세 갈래. 양쪽 수정이 충돌·정본 갈림 후보다."""
    base = _git("merge-base", ours, ref)
    if not base:
        raise SystemExit(f"merge-base 를 못 찾는다 — '{ref}' 를 fetch 했는가")
    ours_f = set(filter(None, _git("diff", "--name-only", base, ours).split("\n")))
    theirs_f = set(filter(None, _git("diff", "--name-only", base, ref).split("\n")))
    return {"base": base, "ours_only": sorted(ours_f - theirs_f),
            "theirs_only": sorted(theirs_f - ours_f),
            "both": sorted(ours_f & theirs_f)}


def leak_scan(ref: str, files: list) -> list:
    """들어오는 파일에 유출물이 섞였는가. **경로와 내용을 둘 다** 본다."""
    hits = []
    for f in files:
        low = f.lower()
        ext = os.path.splitext(low)[1]
        if ext in BLOCKED_EXT:
            hits.append({"file": f, "why": BLOCKED_EXT[ext], "how": "확장자"})
            continue
        for pat, why in BLOCKED_PATH.items():
            if pat in low:
                hits.append({"file": f, "why": why, "how": "경로"})
                break
        else:
            if not low.endswith(TEXT_EXT):
                continue
            blob = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{f}"],
                                  capture_output=True, text=True, check=False)
            if blob.returncode:
                continue                      # 저쪽에서 지운 파일
            for pat, what in KEYLIKE:
                for m in re.finditer(pat, blob.stdout):
                    ctx = blob.stdout[max(0, m.start() - 60):m.start()].lower()
                    if any(w in ctx for w in ("예시", "example", "발급받은",
                                              "your_", "<", "sha256")):
                        continue
                    line = blob.stdout[:m.start()].count("\n") + 1
                    hits.append({"file": f"{f}:{line}", "how": "내용",
                                 "why": f"{what}: {m.group(0)[:8]}…"})
    return hits


def after_merge() -> int:
    """합친 뒤 검증 — 통과해야 합류가 끝난 것이다."""
    ok = True
    for name, cmd in (("테스트", [sys.executable,
                                os.path.join(COMP, "tests", "smoke_test.py")]),
                      ("문서 대조", [sys.executable,
                                 os.path.join(HERE, "check_docs.py")])):
        r = subprocess.run(cmd, capture_output=True, text=True, check=False,
                           cwd=ROOT)
        tail = (r.stdout or r.stderr).strip().splitlines()
        print(f"  {name}: {'✅' if r.returncode == 0 else '❌'} "
              f"{tail[-1] if tail else ''}")
        ok = ok and r.returncode == 0
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print("=" * 72)
    if "--after" in argv:
        print("  합류 후 검증")
        print("=" * 72)
        return after_merge()
    if not argv:
        print("  사용법: merge_check.py origin/<브랜치>   또는   --after")
        return 2
    ref = argv[0]
    d = divergence(ref)
    print(f"  합류 점검 — {ref} (갈린 지점 {d['base'][:7]})")
    print("=" * 72)
    print(f"  우리만 {len(d['ours_only'])} · 저쪽만 {len(d['theirs_only'])} · "
          f"양쪽 {len(d['both'])}")

    leaks = leak_scan(ref, d["theirs_only"] + d["both"])
    if leaks:
        print(f"\n  🔴 유출 위험 {len(leaks)}건 — **합치기 전에 저쪽에서 빼야 한다**")
        print("     (한 번 올라가면 공개 저장소 이력에서 회수가 안 된다)")
        for h in leaks[:20]:
            print(f"     {h['file']}  — {h['why']} [{h['how']}]")
        if len(leaks) > 20:
            print(f"     … 외 {len(leaks) - 20}건")
    else:
        print("\n  ✅ 유출 위험 없음 (확장자·경로·내용)")

    hot = [f for f in d["both"] if f in HOTSPOTS]
    if hot:
        print(f"\n  ⚠ 손으로 볼 자리 {len(hot)}건 — 양쪽이 모두 고쳤다")
        for f in hot:
            print(f"     {f}\n        → {HOTSPOTS[f]}")
    other = [f for f in d["both"] if f not in HOTSPOTS]
    if other:
        print(f"\n  ⚠ 양쪽 수정 {len(other)}건 — **정본 갈림 후보**")
        for f in other[:15]:
            print(f"     {f}")
        if len(other) > 15:
            print(f"     … 외 {len(other) - 15}건")

    new_src = [f for f in d["theirs_only"] if f.endswith(".py")]
    if new_src:
        print(f"\n  ＋ 저쪽에서 오는 새 소스 {len(new_src)}건")
        for f in new_src[:15]:
            print(f"     {f}")
        if len(new_src) > 15:
            print(f"     … 외 {len(new_src) - 15}건")

    print("\n  다음: 유출 0 을 확인하고 합친 뒤 `merge_check.py --after`")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
