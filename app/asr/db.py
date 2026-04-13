from __future__ import annotations

from pathlib import Path

from peewee import DatabaseProxy, SqliteDatabase


database_proxy: DatabaseProxy = DatabaseProxy()
_db: SqliteDatabase | None = None


def initialize_database(path: Path) -> SqliteDatabase:
    global _db

    if _db is None or Path(_db.database) != path:
        if _db is not None and not _db.is_closed():
            _db.close()
        _db = SqliteDatabase(
            str(path),
            pragmas={
                "journal_mode": "wal",
                "foreign_keys": 1,
                "cache_size": -64 * 1000,
                "synchronous": 0,
            },
            check_same_thread=False,
        )
        database_proxy.initialize(_db)

    if _db.is_closed():
        _db.connect(reuse_if_open=True)

    return _db


def get_database() -> SqliteDatabase:
    if _db is None:
        raise RuntimeError("Database has not been initialized")
    return _db


def ensure_connection() -> SqliteDatabase:
    db = get_database()
    if db.is_closed():
        db.connect(reuse_if_open=True)
    return db


def close_database() -> None:
    global _db
    if _db is not None and not _db.is_closed():
        _db.close()
    _db = None
