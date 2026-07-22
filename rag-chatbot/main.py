"""문서 기반 RAG 챗봇 MVP 백엔드.

구조:
  1. 문서 업로드(POST /upload) → 텍스트 추출 → 청크 분할 → 임베딩 → Chroma 벡터DB 저장
  2. 질문(POST /chat) → 관련 청크 검색 → Claude API로 답변 생성

실행 방법은 같은 폴더의 README.md 참고.
"""

import os
import uuid
from pathlib import Path

import anthropic
import chromadb
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

import image_analysis

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"          # 벡터DB가 저장되는 폴더 (자동 생성됨)
COLLECTION_NAME = "documents"

CHUNK_SIZE = 1000      # 청크 하나의 최대 글자 수
CHUNK_OVERLAP = 200    # 인접 청크끼리 겹치는 글자 수 (문맥이 끊기지 않게)
TOP_K = 5              # 질문마다 검색해 오는 청크 개수

CLAUDE_MODEL = "claude-opus-4-8"

# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------

app = FastAPI(title="문서 RAG 챗봇 MVP")

# Chroma: 로컬 디스크에 저장되는 벡터DB.
# 기본 임베딩 모델(all-MiniLM-L6-v2)을 사용하며, 첫 문서 업로드 때
# 모델 파일(~80MB)을 한 번 자동 다운로드한다.
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Anthropic 클라이언트: ANTHROPIC_API_KEY 환경변수를 자동으로 읽는다.
claude = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# 문서 처리
# ---------------------------------------------------------------------------


def extract_text(filename: str, data: bytes) -> str:
    """업로드된 파일에서 순수 텍스트를 추출한다. (.txt, .md, .pdf 지원)"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    raise HTTPException(
        status_code=400,
        detail=f"지원하지 않는 파일 형식입니다: {suffix} (지원: .txt, .md, .pdf)",
    )


def split_into_chunks(text: str) -> list[str]:
    """긴 텍스트를 CHUNK_SIZE 글자 단위로, CHUNK_OVERLAP만큼 겹치게 나눈다."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_document(file: UploadFile):
    """문서를 받아 청크로 나누고 벡터DB에 저장한다."""
    data = await file.read()
    text = extract_text(file.filename, data)
    chunks = split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="파일에서 텍스트를 찾지 못했습니다.")

    doc_id = str(uuid.uuid4())[:8]
    collection.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": file.filename, "doc_id": doc_id, "chunk_index": i}
            for i in range(len(chunks))
        ],
    )
    return {"filename": file.filename, "doc_id": doc_id, "chunks": len(chunks)}


@app.post("/upload-image")
async def upload_image(file: UploadFile, class_names: str = Form(default="")):
    """이미지를 받아 YOLO-World(탐지) + SAM-2(분할)로 분석한다.

    분석 결과 설명은 벡터DB에도 저장되어, 이후 챗봇 질문에서 문서와 함께 검색된다.
    class_names: 탐지할 클래스를 쉼표로 구분(비우면 기본 클래스 사용).
    """
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise HTTPException(
            status_code=500,
            detail="REPLICATE_API_TOKEN 환경변수가 설정되어 있지 않습니다. "
            "https://replicate.com/account/api-tokens 에서 토큰을 발급받아 등록하세요.",
        )

    data = await file.read()
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 이미지 형식입니다: {suffix} (지원: jpg, png, webp, gif, bmp)",
        )

    try:
        result = image_analysis.analyze_image(
            data, file.filename, class_names=class_names.strip() or None
        )
    except Exception as e:  # Replicate 호출/네트워크 오류 등
        raise HTTPException(status_code=502, detail=f"이미지 분석 실패: {e}")

    # 분석 설명을 벡터DB에 저장 → 챗봇이 이미지 내용도 검색해 답변할 수 있게 함
    img_id = str(uuid.uuid4())[:8]
    collection.add(
        ids=[f"img-{img_id}"],
        documents=[result["description"]],
        metadatas=[
            {"source": file.filename, "doc_id": img_id, "type": "image"}
        ],
    )

    return {
        "filename": result["filename"],
        "doc_id": img_id,
        "description": result["description"],
        "detections": result["detections"],
        "segmentation": result["segmentation"],
    }


@app.get("/documents")
def list_documents():
    """지금까지 업로드된 문서 목록을 반환한다."""
    result = collection.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in result["metadatas"]:
        entry = docs.setdefault(
            meta["doc_id"], {"filename": meta["source"], "chunks": 0}
        )
        entry["chunks"] += 1
    return {"documents": [{"doc_id": k, **v} for k, v in docs.items()]}


@app.delete("/documents")
def delete_all_documents():
    """벡터DB를 초기화한다 (업로드한 문서 전부 삭제)."""
    global collection
    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    return {"status": "ok", "message": "모든 문서를 삭제했습니다."}


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    """질문과 관련된 청크를 검색한 뒤 Claude API로 답변을 생성한다."""
    if collection.count() == 0:
        raise HTTPException(
            status_code=400, detail="먼저 문서를 업로드해 주세요."
        )

    results = collection.query(
        query_texts=[req.question],
        n_results=min(TOP_K, collection.count()),
    )
    retrieved_chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]

    context = "\n\n---\n\n".join(
        f"[출처: {src}]\n{chunk}" for src, chunk in zip(sources, retrieved_chunks)
    )

    try:
        response = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=(
                "당신은 사용자가 업로드한 문서를 근거로 답변하는 어시스턴트입니다. "
                "아래에 제공되는 검색 결과(문서 발췌)만을 근거로 답변하세요. "
                "검색 결과에 답이 없으면 솔직하게 문서에서 찾을 수 없다고 말하세요. "
                "답변할 때 어떤 출처 문서를 참고했는지 함께 알려주세요. "
                "사용자의 질문 언어와 같은 언어로 답변하세요."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<검색결과>\n{context}\n</검색결과>\n\n"
                        f"질문: {req.question}"
                    ),
                }
            ],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(
            status_code=500,
            detail="Claude API 키가 올바르지 않습니다. ANTHROPIC_API_KEY 환경변수를 확인하세요.",
        )
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(
            status_code=502, detail=f"Claude API 오류: {e.message}"
        )
    except anthropic.APIConnectionError:
        raise HTTPException(
            status_code=502,
            detail="Claude API에 연결하지 못했습니다. 인터넷 연결을 확인하세요.",
        )

    answer = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return {"answer": answer, "sources": sorted(set(sources))}


# ---------------------------------------------------------------------------
# 웹 UI (static/index.html)
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
