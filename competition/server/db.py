"""SQLite — 등록한 농장을 저장한다.

## 왜 파일 하나인가

심사용 데모다. 서버를 띄우는 것 자체가 이미 진입장벽인데 DB 서버까지
요구하면 아무도 안 돌려 본다. SQLite 는 파이썬 표준 라이브러리라 설치가
없고, 파일 하나를 지우면 초기화된다.

## 농장 이름은 식별자다

이 프로젝트는 원자료 스프레드시트를 커밋하지 않는다 — 농장 식별자가 있기
때문이다. 같은 이유로 **DB 파일도 커밋하지 않는다**(`.gitignore`). 심사용
데모라 인증이 없으므로, 서버를 공개 주소에 띄우지 말 것.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.dirname(HERE)
DB_PATH = os.environ.get("YANGDON_DB", os.path.join(COMP, "data", "farms.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS farm (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  setup_json  TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | None = None) -> sqlite3.Connection:
    p = path or DB_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["setup"] = json.loads(d.pop("setup_json"))
    return d


def list_farms(con: sqlite3.Connection) -> list:
    cur = con.execute("SELECT * FROM farm ORDER BY updated_at DESC")
    return [_row(r) for r in cur.fetchall()]


def get_farm(con: sqlite3.Connection, farm_id: int) -> dict | None:
    cur = con.execute("SELECT * FROM farm WHERE id = ?", (farm_id,))
    r = cur.fetchone()
    return _row(r) if r else None


def create_farm(con: sqlite3.Connection, name: str, setup: dict) -> dict:
    now = _now()
    cur = con.execute(
        "INSERT INTO farm (name, setup_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (name, json.dumps(setup, ensure_ascii=False), now, now))
    con.commit()
    return get_farm(con, cur.lastrowid)


def update_farm(con: sqlite3.Connection, farm_id: int,
                name: str, setup: dict) -> dict | None:
    if not get_farm(con, farm_id):
        return None
    con.execute(
        "UPDATE farm SET name = ?, setup_json = ?, updated_at = ? WHERE id = ?",
        (name, json.dumps(setup, ensure_ascii=False), _now(), farm_id))
    con.commit()
    return get_farm(con, farm_id)


def delete_farm(con: sqlite3.Connection, farm_id: int) -> bool:
    cur = con.execute("DELETE FROM farm WHERE id = ?", (farm_id,))
    con.commit()
    return cur.rowcount > 0
