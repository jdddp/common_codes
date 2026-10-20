"""老库（v1.5：属性带 key/类型、值分四列）升级到 v1.6（属性只有名称、值全是文本）。"""
from __future__ import annotations

from sqlalchemy import create_engine

from app import db as db_module


def _make_v15_db(path):
    """造一个 v1.5 结构的库：属性有 key/label/value_type/unit/options，值分 text/num/bool/json 四列。"""
    engine = create_engine(f"sqlite:///{path}", future=True)
    with engine.begin() as conn:
        # 老库当然有文件表（属性值是挂在文件上的）
        conn.exec_driver_sql("CREATE TABLE files (id INTEGER PRIMARY KEY AUTOINCREMENT, original_name VARCHAR(255))")
        conn.exec_driver_sql("INSERT INTO files (id, original_name) VALUES (1, 'a.dat'), (2, 'b.dat')")
        conn.exec_driver_sql(
            "CREATE TABLE attribute_definitions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " key VARCHAR(100) NOT NULL UNIQUE,"
            " label VARCHAR(200) NOT NULL,"
            " value_type VARCHAR(16) NOT NULL,"
            " unit VARCHAR(32) NOT NULL DEFAULT '',"
            " options TEXT,"
            " created_at VARCHAR(32) NOT NULL DEFAULT '')"
        )
        conn.exec_driver_sql(
            "CREATE TABLE attribute_values ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " file_id INTEGER NOT NULL,"
            " attr_def_id INTEGER NOT NULL,"
            " value_text TEXT, value_num FLOAT, value_bool INTEGER, value_json TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO attribute_definitions (id, key, label, value_type, unit, options, created_at)"
            " VALUES (1, 'depth', '深度', 'float', 'm', NULL, '2026-01-01 00:00:00'),"
            "        (2, 'flag', '是否复核', 'bool', '', NULL, '2026-01-01 00:00:00'),"
            "        (3, 'note', '深度', 'string', '', NULL, '2026-01-01 00:00:00')"
        )
        conn.exec_driver_sql(
            "INSERT INTO attribute_values (file_id, attr_def_id, value_text, value_num, value_bool, value_json)"
            " VALUES (1, 1, NULL, 120.0, NULL, NULL),"      # 整数型浮点 → "120"（不带 .0）
            "        (1, 2, NULL, NULL, 1, NULL),"          # 布尔 → "true"
            "        (2, 3, '甲组', NULL, NULL, NULL),"     # 文本原样
            "        (2, 1, NULL, NULL, NULL, NULL)"        # 空值 → 不落行
        )
    return engine


def test_v15_attributes_migrate_to_name_only(tmp_path):
    engine = _make_v15_db(tmp_path / "old_v15.db")

    db_module._migrate_attribute_tables(engine)

    with engine.connect() as conn:
        definitions = conn.exec_driver_sql(
            "SELECT id, name FROM attribute_definitions ORDER BY id"
        ).fetchall()
        values = conn.exec_driver_sql(
            "SELECT file_id, attr_def_id, value_text FROM attribute_values ORDER BY file_id, attr_def_id"
        ).fetchall()

    # 名称优先取显示名；两个"深度"重名时自动加后缀保证唯一
    assert definitions == [(1, "深度"), (2, "是否复核"), (3, "深度(3)")]
    # 四列折算成文本；空值不落行
    assert values == [(1, 1, "120"), (1, 2, "true"), (2, 3, "甲组")]


def test_migrate_is_idempotent_and_noop_for_new_db(tmp_path):
    engine = _make_v15_db(tmp_path / "old_again.db")
    db_module._migrate_attribute_tables(engine)
    db_module._migrate_attribute_tables(engine)  # 再跑一次不应报错、不应重复

    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM attribute_definitions").scalar() == 3
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM attribute_values").scalar() == 3
        columns = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(attribute_values)").fetchall()
        }
    assert "value_num" not in columns and "value_text" in columns

    # 全新结构（已含 name）的库不再被处理
    fresh = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}", future=True)
    with fresh.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE attribute_definitions (id INTEGER PRIMARY KEY, name VARCHAR(100), created_at VARCHAR(32))"
        )
        conn.exec_driver_sql("INSERT INTO attribute_definitions (id, name, created_at) VALUES (1, '海域', '')")
    db_module._migrate_attribute_tables(fresh)
    with fresh.connect() as conn:
        assert conn.exec_driver_sql("SELECT name FROM attribute_definitions").fetchall() == [("海域",)]
