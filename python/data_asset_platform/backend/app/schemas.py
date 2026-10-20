from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- 属性定义 / 模板
# v1.6：属性只有名称；值一律是字符串（不再有类型概念）
class AttributeDefinitionIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AttributeNamesIn(BaseModel):
    """批量"取或建"属性（属性库已有则复用，缺失才新建）。"""

    names: list[str] = Field(min_length=1)


class AttributeDefinitionOut(BaseModel):
    id: int
    name: str
    used_count: int = 0
    created_at: str


class TemplateItemIn(BaseModel):
    attr_def_id: int
    required: bool = False
    default_value: str | None = None
    sort_order: int = 0


class TemplateIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    items: list[TemplateItemIn] = Field(default_factory=list)


class TemplateItemOut(BaseModel):
    attr_def_id: int
    name: str
    required: bool
    default_value: str | None = None
    sort_order: int


class TemplateOut(BaseModel):
    id: int
    name: str
    description: str
    items: list[TemplateItemOut] = Field(default_factory=list)
    created_at: str


# ---------------------------------------------------------------- 上传
class UploadFileSpec(BaseModel):
    client_key: str = ""
    name: str = Field(min_length=1, max_length=255)
    rel_path: str = ""
    size: int = Field(ge=0)
    sha256: str | None = None
    chunk_size: int | None = None


class UploadInitIn(BaseModel):
    root_name: str = ""
    template_id: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    auto_create_attributes: bool = Field(
        False,
        description="true=attributes 里属性库还没有的属性名自动新建并保留在属性库（默认 false 时直接报错）",
    )
    files: list[UploadFileSpec]


class UploadInitItem(BaseModel):
    client_key: str
    status: Literal["uploading", "done", "failed"]
    session_id: str | None = None
    file_id: int | None = None
    total_chunks: int = 0
    received: list[int] = Field(default_factory=list)
    chunk_size: int = 0
    message: str = ""


class UploadInitOut(BaseModel):
    items: list[UploadInitItem]


class UploadProgressOut(BaseModel):
    session_id: str
    status: str
    file_name: str
    total_chunks: int
    chunk_size: int
    received: list[int]
    missing: list[int]
    file_id: int | None = None
    error: str | None = None


# ---------------------------------------------------------------- 文件
class FileOut(BaseModel):
    id: int
    original_name: str
    rel_path: str
    size_bytes: int
    ext: str
    format_label: str
    uploaded_at: str
    parse_status: str
    protocol: str
    frame_count: int | None = None
    image_shape: str | None = None
    parse_error: str | None = None
    sha256: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class AttributeValueIn(BaseModel):
    name: str
    value: Any = None


class FileAttributesIn(BaseModel):
    values: list[AttributeValueIn]
    replace: bool = False


class PageOut(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[Any]


# ---------------------------------------------------------------- 查看器
class AlgorithmOut(BaseModel):
    name: str
    description: str
    stateful: bool = False


class SonarInfoOut(BaseModel):
    supported: bool
    protocol: str = "unknown"
    reason: str = ""
    frame_count: int = 0
    image_shape: list[int] | None = None
    colormaps: list[str] = Field(default_factory=list)
    playback: dict[str, Any] = Field(default_factory=dict)
    algorithms: list[AlgorithmOut] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    first_frame_meta: dict[str, Any] | None = None


class FrameMetaOut(BaseModel):
    ping_num: int
    nBeam: int
    nRange: int
    range_m: float
    open_angle_deg: float
    frequency: float
    pluse_width: float
    band_width: float
    water_depth: float
    high_mode: bool
    time_stamp: float
    angle_base: float = 0.0
    angle_offset: float = 0.0


class PanelImage(BaseModel):
    mime: str
    data: str


class FrameOut(BaseModel):
    index: int
    frame_count: int
    gray: PanelImage
    color: PanelImage
    meta: FrameMetaOut


class FramesOut(BaseModel):
    start: int
    step: int
    count: int
    frame_count: int
    frames: list[FrameOut]


class PixelOut(BaseModel):
    index: int
    x: int
    y: int
    raw: float
    processed: float | None = None
    angle_deg: float | None = None
    in_sector: bool | None = None
