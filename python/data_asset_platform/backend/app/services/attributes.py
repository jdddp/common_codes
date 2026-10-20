"""属性（键）与属性值（字符串）的读写。

口径（v1.6 起）：
  * 属性 = **名称**（唯一，同名即同一个属性），不再有 key/类型/单位/可选值
  * 值一律是**字符串**；数字/布尔等只是文本的写法（`"120"`、`"true"`）
  * 空值 = 该文件没有这个属性（写入空值即删除该行）
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import Select, exists, func, select
from sqlalchemy.orm import Session

from ..db import BUILTIN_DATE_NAME
from ..models import AttributeDefinition, AttributeValue, FileEntry

# 筛选语法里用到的字符：属性名里不能出现（否则 `属性名=值` 无法解析）
_BAD_NAME_CHARS = "="

# 时间范围比较运算符（只对内置「日期」开放，见 apply_attr_condition）
_RANGE_OPS = (">=", "<=")

# 日期接受的两种写法：`2026-10-19`（可带时间尾巴）与紧凑的 `20261019`
_DATE_SEPARATED = re.compile(r"^(\d{4})\D(\d{1,2})\D(\d{1,2})(?:[ T].*)?$")
_DATE_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def validate_name(raw: Any) -> str:
    """校验并规范化属性名（去首尾空白；禁止 `=` 与换行制表符）。"""
    name = str(raw if raw is not None else "").strip()
    if not name:
        raise ValueError("属性名称不能为空")
    if len(name) > 100:
        raise ValueError("属性名称过长（最多 100 个字）")
    bad = sorted({char for char in name if char in _BAD_NAME_CHARS})
    if bad:
        raise ValueError(f"属性名称不能包含 {''.join(bad)}（筛选条件里用它分隔）")
    if any(char in name for char in ("\n", "\r", "\t")):
        raise ValueError("属性名称不能包含换行或制表符")
    return name


def normalize_text(value: Any) -> str:
    """把任意输入值归一成字符串：布尔 → true/false，整数浮点不加多余 `.0`，对象转 JSON。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def normalize_date(value: Any) -> str:
    """把日期输入归一成 **`YYYY-MM-DD`**；解析不出真实日期就报错（调用方转 422）。

    接受 `2026-10-19` / `2026/10/19` / `2026.10.19` / `20261019`，以及带时间的
    `2026-10-19 09:30:00`、`2026-10-19T09:30:00`（只取日期部分）。空值原样返回空串
    （空 = 该文件没有这个属性，与其它属性一致）。

    归一成定宽的 `YYYY-MM-DD` 是**时间范围能筛的前提**：文本列按字节比较，定宽 ISO
    日期的字典序恰好等于时间序，`>=` / `<=` 才是真正的时间比较。
    """
    text = normalize_text(value)
    if not text:
        return ""
    match = _DATE_COMPACT.match(text)
    if match:
        year, month, day = match.groups()
    else:
        match = _DATE_SEPARATED.match(text)
        if match is None:
            raise ValueError(f"日期必须是 YYYY-MM-DD 这样的日期，收到：{value}")
        year, month, day = match.groups()
    try:
        moment = datetime(int(year), int(month), int(day))
    except ValueError as exc:
        raise ValueError(f"不是一个真实存在的日期，收到：{value}") from exc
    return f"{moment.year:04d}-{moment.month:02d}-{moment.day:02d}"


def set_file_attribute(
    db: Session,
    file_entry: FileEntry,
    definition: AttributeDefinition,
    value: Any,
) -> None:
    """写入/覆盖/清除一个属性值（空值 ⇒ 删除该属性值）。

    内置属性（目前只有「日期」）在这里统一做日期校验与归一，于是**所有写入口**
    ——上传、服务端导入、单文件属性编辑、模板默认值——都受同一套约束。
    """
    row = db.scalar(
        select(AttributeValue).where(
            AttributeValue.file_id == file_entry.id,
            AttributeValue.attr_def_id == definition.id,
        )
    )
    text = normalize_date(value) if int(definition.is_builtin or 0) else normalize_text(value)
    if not text:
        if row is not None:
            db.delete(row)
        return
    if row is None:
        row = AttributeValue(file_id=file_entry.id, attr_def_id=definition.id)
        db.add(row)
    row.value_text = text


def set_file_attributes_by_name(
    db: Session, file_entry: FileEntry, values: dict[str, Any]
) -> list[str]:
    """按属性名写入；返回被忽略的未知属性名。"""
    if not values:
        return []
    definitions = {
        definition.name: definition
        for definition in db.scalars(select(AttributeDefinition)).all()
    }
    ignored: list[str] = []
    for name, value in values.items():
        definition = definitions.get(str(name).strip())
        if definition is None:
            ignored.append(str(name))
            continue
        set_file_attribute(db, file_entry, definition, value)
    return ignored


def values_map_for_files(db: Session, file_ids: Iterable[int]) -> dict[int, dict[str, str]]:
    ids = [int(value) for value in file_ids]
    if not ids:
        return {}
    rows = db.scalars(
        select(AttributeValue).where(AttributeValue.file_id.in_(ids))
    ).unique().all()
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        definition = row.attr_def
        if definition is None or not row.value_text:
            continue
        result.setdefault(row.file_id, {})[definition.name] = row.value_text
    return result


def distinct_values(db: Session, definition: AttributeDefinition, limit: int = 200) -> dict[str, Any]:
    """某个属性**在本库已经出现过的取值**（去重 + 计数）。

    用途：填写/筛选时"从已有的里面直接选"，而不是每次手敲。
      * 按出现次数降序、取值升序返回，最多 `limit` 条；超出时 `truncated=true`
    """
    used = func.count(AttributeValue.id)
    rows = db.execute(
        select(AttributeValue.value_text, used)
        .where(
            AttributeValue.attr_def_id == definition.id,
            AttributeValue.value_text != "",
        )
        .group_by(AttributeValue.value_text)
        .order_by(used.desc(), AttributeValue.value_text.asc())
    ).all()

    items = [{"value": text, "file_count": int(count)} for text, count in rows]
    size = max(1, int(limit))
    return {
        "name": definition.name,
        "items": items[:size],
        "total": len(items),
        "truncated": len(items) > size,
    }


def get_or_create_definitions(
    db: Session, names: Iterable[Any]
) -> tuple[list[AttributeDefinition], list[str], list[str]]:
    """按名称批量"取或建"属性：**属性库已有就复用，缺失的才新建**（新建即保留，供后续复用）。

    返回 (definitions, created_names, existing_names)；入参里重复的名称只处理一次。
    """
    existing = {
        definition.name: definition
        for definition in db.scalars(select(AttributeDefinition)).all()
    }
    definitions: list[AttributeDefinition] = []
    created: list[str] = []
    reused: list[str] = []
    seen: set[str] = set()

    for raw in names:
        name = validate_name(raw)
        if name in seen:
            continue
        seen.add(name)

        definition = existing.get(name)
        if definition is not None:
            reused.append(name)
            definitions.append(definition)
            continue

        definition = AttributeDefinition(name=name)
        db.add(definition)
        db.flush()  # 立刻拿到 id，并让同批次后续查询可见
        existing[name] = definition
        created.append(name)
        definitions.append(definition)

    return definitions, created, reused


def template_items_map(db: Session, template_id: int) -> dict[str, AttributeDefinition]:
    from ..models import AttributeTemplate, TemplateItem

    template = db.get(AttributeTemplate, template_id)
    if template is None:
        raise ValueError(f"属性模板不存在: {template_id}")
    result: dict[str, AttributeDefinition] = {}
    rows = db.scalars(
        select(TemplateItem)
        .where(TemplateItem.template_id == template_id)
        .order_by(TemplateItem.sort_order, TemplateItem.id)
    ).all()
    for item in rows:
        definition = item.attr_def
        if definition is not None:
            result[definition.name] = definition
    return result


def parse_attr_filter(raw: str) -> tuple[str, str, str]:
    """解析属性筛选条件，返回 `(属性名, 运算符, 值)`。

    写法：
      * `属性名=值` ——「等于」，**唯一对普通属性开放**的比较
      * `属性名>=值` / `属性名<=值` —— 时间范围，**只对内置「日期」开放**
        （够不够格由 `apply_attr_condition` 按属性判，解析本身不认识属性）

    属性名里禁含 `=`（建属性时已校验），所以第一个 `=` 一定是值的分隔符；
    再看它前面一个字符是不是 `>` / `<`，是就当作范围运算符。这样
    `面积>10=5` 这种含 `>` 的属性名也不会被误读成范围。
    """
    text = str(raw)
    index = text.find("=")
    if index <= 0:
        raise ValueError(
            f"无法解析属性筛选条件: {raw}"
            '（写法：属性名=值；普通属性筛选只支持「等于」这一种比较）'
        )
    head = text[:index]
    op = "="
    if head[-1] in "><":
        op = head[-1] + "="
        head = head[:-1]
    name = head.strip()
    if not name:
        raise ValueError(f"无法解析属性筛选条件: {raw}（属性名为空）")
    return name, op, text[index + 1 :].strip()


def apply_attr_condition(
    statement: Select,
    definition: AttributeDefinition,
    value: str,
    op: str = "=",
) -> Select:
    """把单个属性条件变成 EXISTS 子查询（见设计文档 §8.2）。

    * `=`：值按字符串比较（普通属性与内置属性都支持）
    * `>=` / `<=`：**只有内置「日期」支持**。它的值保证是 `YYYY-MM-DD`，文本列按
      字节比较时定宽 ISO 日期的字典序即时间序，所以普通字符串比较就是正确的时间比较，
      不需要引入时间类型列（见设计文档 §8.2 的取舍）。
    """
    if op == "=":
        criterion = AttributeValue.value_text == value
    elif op in _RANGE_OPS:
        if not int(getattr(definition, "is_builtin", 0) or 0):
            raise ValueError(
                f"属性「{definition.name}」不支持时间范围比较"
                "（只有内置「日期」可以用 >= / <=）"
            )
        bound = normalize_date(value)
        if not bound:
            raise ValueError("时间范围的一端不能为空")
        criterion = (
            AttributeValue.value_text >= bound
            if op == ">="
            else AttributeValue.value_text <= bound
        )
    else:
        raise ValueError(f"不支持的比较方式: {op}")

    subquery = select(AttributeValue.id).where(
        AttributeValue.file_id == FileEntry.id,
        AttributeValue.attr_def_id == definition.id,
        criterion,
    )
    return statement.where(exists(subquery))
