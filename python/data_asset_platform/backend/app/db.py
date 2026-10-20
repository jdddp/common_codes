from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    f"sqlite:///{settings.db_path}",
    echo=False,
    future=True,
    connect_args={"check_same_thread": False, "timeout": 10},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _legacy_value_text(text_value, num_value, bool_value, json_value) -> str:
    """把 v1.5（按类型分列）的老值折算成字符串（v1.6 只有一列文本）。"""
    if text_value not in (None, ""):
        return str(text_value)
    if bool_value is not None:
        return "true" if int(bool_value) else "false"
    if num_value is not None:
        number = float(num_value)
        return str(int(number)) if number.is_integer() else str(number)
    if json_value not in (None, ""):
        return str(json_value)
    return ""


def _migrate_attribute_tables(target_engine=None) -> None:
    """v1.6 兼容升级：老库属性带 key/类型/单位/可选值 → 只保留名称，值全部折成文本。

    * `attribute_definitions.key/label` → `attribute_definitions.name`（优先显示名；重名自动加后缀）
    * `attribute_values` 的四列（text/num/bool/json）→ 单列 `value_text`（空值直接不落行）
    * 幂等：已经升级过的库不会再动；表里原来的 id 保持不变（模板字段的引用继续有效）
    """
    target_engine = target_engine or engine
    inspector = inspect(target_engine)
    if not inspector.has_table("attribute_definitions"):
        return
    columns = {column["name"] for column in inspector.get_columns("attribute_definitions")}
    if "name" in columns or "key" not in columns:
        return  # 已是 v1.6 结构

    from .models import AttributeDefinition, AttributeValue, now_str

    # 1) 用底层连接读旧数据，并在"临时关掉外键"的情况下把旧表挪走（老表是模板字段的外键父表）
    raw = target_engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        old_defs = cursor.execute(
            "SELECT id, key, label, created_at FROM attribute_definitions ORDER BY id"
        ).fetchall()
        old_values = cursor.execute(
            "SELECT file_id, attr_def_id, value_text, value_num, value_bool, value_json "
            "FROM attribute_values"
        ).fetchall()
        cursor.execute("DROP TABLE IF EXISTS attribute_values")
        cursor.execute("DROP TABLE IF EXISTS attribute_definitions")
        raw.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        raw.close()

    # 2) 按新结构建表并回填（id 保持原样，模板字段仍指向同一个属性）
    with target_engine.begin() as conn:
        AttributeDefinition.__table__.create(bind=conn, checkfirst=True)
        AttributeValue.__table__.create(bind=conn, checkfirst=True)

        seen_names: set[str] = set()
        for definition_id, key, label, created_at in old_defs:
            name = str(label or key or "").strip() or f"属性{definition_id}"
            while name in seen_names:
                name = f"{name}({definition_id})"
            seen_names.add(name)
            conn.execute(
                text(
                    "INSERT INTO attribute_definitions "
                    "(id, name, created_at, is_builtin) "
                    "VALUES (:id, :name, :created_at, 0)"
                ),
                {"id": definition_id, "name": name, "created_at": created_at or now_str()},
            )

        for file_id, attr_def_id, value_text, value_num, value_bool, value_json in old_values:
            text_value = _legacy_value_text(value_text, value_num, value_bool, value_json)
            if not text_value:
                continue
            conn.execute(
                text(
                    "INSERT INTO attribute_values (file_id, attr_def_id, value_text) "
                    "VALUES (:file_id, :attr_def_id, :value_text)"
                ),
                {"file_id": file_id, "attr_def_id": attr_def_id, "value_text": text_value},
            )


# 内置属性名：平台自动建立、不可删除、不可改名，取值必须是真实日期（见 services/attributes.py）
BUILTIN_DATE_NAME = "日期"


def _ensure_builtin_attributes() -> None:
    """v1.8：内置属性 `日期`（补列 + 建属性，幂等，新库老库都跑）。

    * 老库若已有用户自己建的同名属性 → **直接采纳成内置**（属性=名称，同名即同一个属性），
      已有的取值原样保留，只从此刻起开始按 `YYYY-MM-DD` 校验新写入的值。
    * 新库则直接建一条 `is_builtin=1` 的「日期」。
    """
    inspector = inspect(engine)
    if not inspector.has_table("attribute_definitions"):
        return
    columns = {column["name"] for column in inspector.get_columns("attribute_definitions")}
    if "is_builtin" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE attribute_definitions "
                    "ADD COLUMN is_builtin INTEGER NOT NULL DEFAULT 0"
                )
            )

    from sqlalchemy import select

    from .models import AttributeDefinition, AttributeValue
    from .services.attributes import normalize_date  # 局部导入：db 不在模块层依赖 service

    with SessionLocal() as db:
        definition = db.scalar(
            select(AttributeDefinition).where(AttributeDefinition.name == BUILTIN_DATE_NAME)
        )
        if definition is None:
            definition = AttributeDefinition(name=BUILTIN_DATE_NAME, is_builtin=1)
            db.add(definition)
            db.flush()
        elif not int(definition.is_builtin or 0):
            definition.is_builtin = 1

        # 老库里已有的「日期」取值**尽力**归一成 YYYY-MM-DD（能解析才改，解析不了的原样
        # 保留、不猜——归一后才有"字典序即时间序"，时间范围才筛得准）
        for row in db.scalars(
            select(AttributeValue).where(AttributeValue.attr_def_id == definition.id)
        ).all():
            try:
                normalized = normalize_date(row.value_text)
            except ValueError:
                continue
            if normalized and normalized != row.value_text:
                row.value_text = normalized

        db.commit()


def init_db() -> None:
    """建表（幂等）。老库会先做一次 v1.6 属性结构升级，再按模型建缺失的表，
    最后补上内置属性（v1.8 的 `日期`）。"""
    settings.ensure_dirs()
    from . import models  # noqa: F401  确保模型已注册

    _migrate_attribute_tables()
    Base.metadata.create_all(bind=engine)
    _ensure_builtin_attributes()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI 依赖。"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
