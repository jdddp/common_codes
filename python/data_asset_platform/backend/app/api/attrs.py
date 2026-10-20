"""自定义属性：属性（只有名称）+ 属性模板。

口径（v1.6）：属性名就是标识，值一律是字符串，没有类型/单位/可选值。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import (
    AttributeDefinition,
    AttributeTemplate,
    AttributeValue,
    TemplateItem,
)
from ..schemas import AttributeDefinitionIn, AttributeNamesIn, TemplateIn
from ..services import attributes as attr_service

router = APIRouter(prefix="/api", tags=["attributes"])


def definition_out(db: Session, definition: AttributeDefinition) -> dict:
    used = int(
        db.scalar(
            select(func.count(AttributeValue.id)).where(
                AttributeValue.attr_def_id == definition.id
            )
        )
        or 0
    )
    return {
        "id": definition.id,
        "name": definition.name,
        "used_count": used,
        "created_at": definition.created_at,
        "is_builtin": bool(int(definition.is_builtin or 0)),
    }


def _clean_name(raw: str) -> str:
    try:
        return attr_service.validate_name(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def template_out(db: Session, template: AttributeTemplate) -> dict:
    rows = db.scalars(
        select(TemplateItem)
        .where(TemplateItem.template_id == template.id)
        .order_by(TemplateItem.sort_order, TemplateItem.id)
    ).all()
    items = []
    for item in rows:
        definition = item.attr_def
        if definition is None:
            continue
        items.append(
            {
                "attr_def_id": definition.id,
                "name": definition.name,
                "required": bool(item.required),
                "default_value": item.default_value,
                "sort_order": item.sort_order,
            }
        )
    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "items": items,
        "created_at": template.created_at,
    }


# ---------------------------------------------------------------- 属性定义
@router.get("/attributes")
def list_definitions(db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(AttributeDefinition).order_by(AttributeDefinition.id)).all()
    return {"items": [definition_out(db, row) for row in rows]}


@router.get("/attributes/values")
def list_attribute_values(
    name: str = "",
    limit: int = 200,
    db: Session = Depends(get_db),
) -> dict:
    """**属性已有的取值**（去重 + 计数），供填写/筛选时"直接选"而不是手敲。

    * 不传 `name` → 返回属性库里所有属性的已有取值（免构建页/筛选面板一次拉齐）
    * 传 `name` → 只返回该属性（切换属性时按需刷新，不存在则 404）
    * 每个属性最多 `limit` 条（上限 1000），`truncated=true` 表示还有更多
    """
    statement = select(AttributeDefinition).order_by(AttributeDefinition.id)
    if name:
        statement = statement.where(AttributeDefinition.name == name.strip())
    definitions = db.scalars(statement).all()
    if name and not definitions:
        raise HTTPException(status_code=404, detail=f"属性不存在: {name}")

    size = max(1, min(int(limit), 1000))
    items = [attr_service.distinct_values(db, definition, size) for definition in definitions]
    return {"items": items, "limit": size}


@router.post("/attributes")
def create_definition(payload: AttributeDefinitionIn, db: Session = Depends(get_db)) -> dict:
    name = _clean_name(payload.name)
    if db.scalar(select(AttributeDefinition).where(AttributeDefinition.name == name)) is not None:
        raise HTTPException(status_code=409, detail=f"属性已存在: {name}")
    definition = AttributeDefinition(name=name)
    db.add(definition)
    db.commit()
    return definition_out(db, definition)


@router.post("/attributes/resolve")
def resolve_definitions(payload: AttributeNamesIn, db: Session = Depends(get_db)) -> dict:
    """批量"取或建"属性：**属性库已有则复用，没有才新建**；新建的属性保留在属性库中。

    上传/筛选时"从属性库添加 + 缺失才新建"的落地点：
      * 前端先用本接口把用户填的新属性一次性登记（幂等，重复调用不会产生重复属性）
      * 返回 created / existing 供界面提示"新建了哪些、复用了哪些"
    """
    names = list(payload.names)
    try:
        definitions, created, reused = attr_service.get_or_create_definitions(db, names)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IntegrityError:
        # 并发下另一个请求刚好建了同名属性：回滚后重试，这次会命中"已有"分支
        db.rollback()
        definitions, created, reused = attr_service.get_or_create_definitions(db, names)
    db.commit()
    return {
        "items": [definition_out(db, definition) for definition in definitions],
        "created": created,
        "existing": reused,
    }


@router.patch("/attributes/{definition_id}")
def update_definition(
    definition_id: int, payload: AttributeDefinitionIn, db: Session = Depends(get_db)
) -> dict:
    definition = db.get(AttributeDefinition, definition_id)
    if definition is None:
        raise HTTPException(status_code=404, detail="属性不存在")
    if int(definition.is_builtin or 0):
        raise HTTPException(status_code=422, detail="内置属性不可改名（名称就是它的标识）")
    name = _clean_name(payload.name)
    other = db.scalar(select(AttributeDefinition).where(AttributeDefinition.name == name))
    if other is not None and other.id != definition.id:
        raise HTTPException(status_code=409, detail=f"属性已存在: {name}")
    definition.name = name
    db.commit()
    return definition_out(db, definition)


@router.delete("/attributes/{definition_id}")
def delete_definition(definition_id: int, db: Session = Depends(get_db)) -> dict:
    definition = db.get(AttributeDefinition, definition_id)
    if definition is None:
        raise HTTPException(status_code=404, detail="属性不存在")
    if int(definition.is_builtin or 0):
        raise HTTPException(
            status_code=409,
            detail="内置属性不可删除（平台自动维护，用于按时间范围筛选数据）",
        )
    used = int(
        db.scalar(
            select(func.count(AttributeValue.id)).where(
                AttributeValue.attr_def_id == definition_id
            )
        )
        or 0
    )
    if used:
        raise HTTPException(
            status_code=409, detail=f"该属性已有 {used} 条取值，禁止删除（可改名）"
        )
    for item in db.scalars(
        select(TemplateItem).where(TemplateItem.attr_def_id == definition_id)
    ).all():
        db.delete(item)
    db.delete(definition)
    db.commit()
    return {"id": definition_id, "deleted": True}


# ---------------------------------------------------------------- 属性模板
@router.get("/templates")
def list_templates(db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(AttributeTemplate).order_by(AttributeTemplate.id)).all()
    return {"items": [template_out(db, row) for row in rows]}


@router.post("/templates")
def create_template(payload: TemplateIn, db: Session = Depends(get_db)) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="模板名称不能为空")
    if db.scalar(select(AttributeTemplate).where(AttributeTemplate.name == name)) is not None:
        raise HTTPException(status_code=409, detail=f"模板已存在: {name}")
    template = AttributeTemplate(name=name, description=payload.description)
    db.add(template)
    db.flush()
    _apply_items(db, template, payload)
    db.commit()
    return template_out(db, template)


@router.patch("/templates/{template_id}")
def update_template(template_id: int, payload: TemplateIn, db: Session = Depends(get_db)) -> dict:
    template = db.get(AttributeTemplate, template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="模板不存在")
    template.name = payload.name.strip() or template.name
    template.description = payload.description
    for item in db.scalars(
        select(TemplateItem).where(TemplateItem.template_id == template_id)
    ).all():
        db.delete(item)
    db.flush()
    _apply_items(db, template, payload)
    db.commit()
    return template_out(db, template)


@router.delete("/templates/{template_id}")
def delete_template(template_id: int, db: Session = Depends(get_db)) -> dict:
    template = db.get(AttributeTemplate, template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="模板不存在")
    db.delete(template)
    db.commit()
    return {"id": template_id, "deleted": True}


def _apply_items(db: Session, template: AttributeTemplate, payload: TemplateIn) -> None:
    seen: set[int] = set()
    for item in payload.items:
        if item.attr_def_id in seen:
            continue
        seen.add(item.attr_def_id)
        definition = db.get(AttributeDefinition, item.attr_def_id)
        if definition is None:
            raise HTTPException(status_code=422, detail=f"属性不存在: {item.attr_def_id}")
        db.add(
            TemplateItem(
                template_id=template.id,
                attr_def_id=definition.id,
                required=1 if item.required else 0,
                default_value=item.default_value,
                sort_order=item.sort_order,
            )
        )
    db.flush()
