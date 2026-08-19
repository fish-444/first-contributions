"""농장 등록 CRUD — SQLite.

농장 이름과 성적은 **식별자**다. 심사용 데모라 인증이 없으므로 서버를 공개
주소에 띄우지 말 것. DB 파일은 커밋 대상이 아니다.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from .. import db
from ..schemas import FarmIn, FarmOut

router = APIRouter(prefix="/api/farms", tags=["farms"])

_con = None


def get_con():
    global _con
    if _con is None:
        _con = db.connect()
    return _con


@router.get("", response_model=list[FarmOut], summary="등록한 농장 목록")
def list_farms(con=Depends(get_con)) -> list:
    return db.list_farms(con)


@router.post("", response_model=FarmOut, status_code=201, summary="농장 등록")
def create(body: FarmIn, con=Depends(get_con)) -> dict:
    return db.create_farm(con, body.name, body.setup.model_dump())


@router.get("/{farm_id}", response_model=FarmOut, summary="농장 하나")
def get_one(farm_id: int, con=Depends(get_con)) -> dict:
    f = db.get_farm(con, farm_id)
    if not f:
        raise HTTPException(404, f"농장 {farm_id} 없음")
    return f


@router.put("/{farm_id}", response_model=FarmOut, summary="농장 수정")
def update(farm_id: int, body: FarmIn, con=Depends(get_con)) -> dict:
    f = db.update_farm(con, farm_id, body.name, body.setup.model_dump())
    if not f:
        raise HTTPException(404, f"농장 {farm_id} 없음")
    return f


@router.delete("/{farm_id}", status_code=204, summary="농장 삭제")
def remove(farm_id: int, con=Depends(get_con)) -> None:
    if not db.delete_farm(con, farm_id):
        raise HTTPException(404, f"농장 {farm_id} 없음")


@router.get("/{farm_id}/capacity", summary="등록한 농장의 용량 · 상한 · 처방")
def capacity_of(farm_id: int, con=Depends(get_con)) -> dict:
    from ..schemas import FarmSetup
    from .capacity import compute

    f = db.get_farm(con, farm_id)
    if not f:
        raise HTTPException(404, f"농장 {farm_id} 없음")
    setup = FarmSetup(**f["setup"])
    if not setup.barns:
        raise HTTPException(422, "이 농장에 등록된 돈사가 없다")
    return {"farm": {"id": f["id"], "name": f["name"]}, **compute(setup)}
