"""属性（名称 + 字符串值）/ 模板 / 属性检索 / 清单导出。

v1.6 口径：属性只有名称（名称即标识），值一律是字符串，没有类型概念。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from tests.test_upload import _put


def _create_attr(client, name):
    response = client.post("/api/attributes", json={"name": name})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_attribute_definition_crud(client):
    attr_id = _create_attr(client, "水深")
    # 重复名称冲突
    assert client.post("/api/attributes", json={"name": "水深"}).status_code == 409

    items = client.get("/api/attributes").json()
    assert any(item["name"] == "水深" for item in items["items"])
    assert "value_types" not in items  # 不再有类型概念

    # 改名（同名冲突也拦）
    response = client.patch(f"/api/attributes/{attr_id}", json={"name": "水深（米）"})
    assert response.status_code == 200
    assert response.json()["name"] == "水深（米）"

    # 名称里不能有筛选语法用的分隔符（只剩 = 一个）
    assert client.post("/api/attributes", json={"name": "a=b"}).status_code == 422
    assert client.post("/api/attributes", json={"name": "   "}).status_code == 422
    # `!` 不再是语法字符，允许出现在属性名里
    assert client.post("/api/attributes", json={"name": "a!b"}).status_code == 200

    assert client.delete(f"/api/attributes/{attr_id}").status_code == 200


def test_template_required_and_attribute_search(client, work_dir):
    depth_id = _create_attr(client, "深度")
    area_id = _create_attr(client, "海域")
    inspected_id = _create_attr(client, "是否已检查")
    notes_id = _create_attr(client, "备注")

    response = client.post(
        "/api/templates",
        json={
            "name": "声呐数据模板",
            "description": "声呐文件属性",
            "items": [
                {"attr_def_id": depth_id, "required": True, "sort_order": 1},
                {"attr_def_id": area_id, "required": True, "default_value": "渤海", "sort_order": 2},
                {"attr_def_id": inspected_id, "required": False, "default_value": "false", "sort_order": 3},
                {"attr_def_id": notes_id, "required": False, "sort_order": 4},
            ],
        },
    )
    assert response.status_code == 200, response.text
    template_id = response.json()["id"]
    assert len(response.json()["items"]) == 4
    assert {item["name"] for item in response.json()["items"]} == {"深度", "海域", "是否已检查", "备注"}

    path = Path(work_dir) / "attr_sample.bin"
    path.write_bytes(bytes((index * 7) % 251 for index in range(700_000)))
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()

    # 缺必填属性 → 422
    response = client.post(
        "/api/uploads/init",
        json={
            "template_id": template_id,
            "attributes": {},
            "files": [{"client_key": "a", "name": path.name, "size": len(data), "sha256": sha}],
        },
    )
    assert response.status_code == 422
    assert "缺少必填属性" in response.json()["detail"]

    # 未定义属性 → 422
    response = client.post(
        "/api/uploads/init",
        json={
            "template_id": template_id,
            "attributes": {"深度": 120, "没定义过": 1},
            "files": [{"client_key": "a", "name": path.name, "size": len(data), "sha256": sha}],
        },
    )
    assert response.status_code == 422
    assert "未定义" in response.json()["detail"]

    # 正常路径：模板默认值生效 + 显式值覆盖（值一律按字符串存）
    response = client.post(
        "/api/uploads/init",
        json={
            "template_id": template_id,
            "attributes": {"深度": 120.5, "海域": "黄海", "备注": "第一航次"},
            "files": [{"client_key": "a", "name": path.name, "size": len(data), "sha256": sha}],
        },
    )
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    session_id = item["session_id"]
    total = item["total_chunks"]
    csize = item["chunk_size"]
    for index in range(total):
        _put(client, session_id, index, data[index * csize : (index + 1) * csize])
    result = client.post(f"/api/uploads/{session_id}/complete").json()
    file_id = result["file_id"]

    detail = client.get(f"/api/files/{file_id}").json()
    assert detail["attributes"]["深度"] == "120.5"
    assert detail["attributes"]["海域"] == "黄海"
    assert detail["attributes"]["是否已检查"] == "false"  # 模板默认值
    assert detail["attributes"]["备注"] == "第一航次"

    # 属性检索：只有「等于」这一种比较（值都是字符串）
    assert client.get("/api/files", params={"attr": "海域=黄海"}).json()["total"] >= 1
    assert client.get("/api/files", params={"attr": "海域=东海"}).json()["total"] == 0
    assert client.get("/api/files", params={"attr": "是否已检查=false"}).json()["total"] >= 1
    # 数值格式的字符串仍可按等于匹配
    assert client.get("/api/files", params={"attr": "深度=120.5"}).json()["total"] >= 1
    # 数值比较这类写法已不再支持 → 422（明确报错，不静默乱比）
    assert client.get("/api/files", params={"attr": "深度>100"}).status_code == 422
    # `!=` 也不再是语法：它会被当成属性名的一部分 → 属性不存在，422
    assert client.get("/api/files", params={"attr": "海域!=东海"}).status_code == 422
    # `=like:` 里的前缀只是普通字符（没有模糊匹配了）
    assert client.get("/api/files", params={"attr": "海域=like:%海%"}).json()["total"] == 0

    # 分页与排序、扩展名筛选
    listed = client.get("/api/files", params={"ext": "bin", "sort": "size:desc"}).json()
    assert listed["total"] >= 1
    assert listed["items"][0]["size_bytes"] >= listed["items"][-1]["size_bytes"]

    # 有取值的属性不允许删除
    assert client.delete(f"/api/attributes/{area_id}").status_code == 409

    # 更新属性（增量 + 覆盖）
    response = client.put(
        f"/api/files/{file_id}/attributes",
        json={"values": [{"name": "备注", "value": "复核完成"}], "replace": False},
    )
    assert response.status_code == 200
    assert response.json()["attributes"]["备注"] == "复核完成"
    assert response.json()["attributes"]["海域"] == "黄海"

    response = client.put(
        f"/api/files/{file_id}/attributes",
        json={"values": [{"name": "深度", "value": 200}], "replace": True},
    )
    assert response.status_code == 200
    assert response.json()["attributes"] == {"深度": "200"}

    # 清空某个属性值 ⇒ 该属性不再挂在文件上（清空后该属性也重新变成可删）
    response = client.put(
        f"/api/files/{file_id}/attributes",
        json={"values": [{"name": "深度", "value": ""}], "replace": False},
    )
    assert response.status_code == 200
    assert response.json()["attributes"] == {}
    assert client.delete(f"/api/attributes/{depth_id}").status_code == 200

    # 未知属性名会被忽略并回报
    response = client.put(
        f"/api/files/{file_id}/attributes",
        json={"values": [{"name": "没这个", "value": 1}]},
    )
    assert response.json()["ignored_names"] == ["没这个"]

    # 未定义属性参与检索 → 422
    assert client.get("/api/files", params={"attr": "没这个=1"}).status_code == 422

    # 清单导出（带 BOM，表头含属性列）
    export = client.get("/api/files/export.csv")
    assert export.status_code == 200
    text = export.content.decode("utf-8")
    assert text.startswith("\ufeff")
    assert "海域" in text


def test_attribute_resolve_and_auto_create(client, work_dir):
    """上传/筛选"关联多个属性"的服务端语义：库里已有 → 复用；库里没有 → 新建并保留。"""
    # 0) 先确保"深度"已在属性库（本用例不依赖其它用例的执行顺序）
    client.post("/api/attributes/resolve", json={"names": ["深度"]})

    # 1) 批量取或建：缺失的建，已有的复用（幂等，不重复建）
    first = client.post(
        "/api/attributes/resolve",
        json={"names": ["航次", "深度", "航次"]},
    )
    assert first.status_code == 200, first.text
    assert first.json()["created"] == ["航次"]
    assert first.json()["existing"] == ["深度"]
    assert {item["name"] for item in first.json()["items"]} == {"航次", "深度"}

    # 幂等：重复调用不会重复建
    again = client.post("/api/attributes/resolve", json={"names": ["航次"]}).json()
    assert again["created"] == [] and again["existing"] == ["航次"]
    definitions = {item["name"]: item for item in client.get("/api/attributes").json()["items"]}
    assert "航次" in definitions

    # 非法名称 / 空 payload → 422
    assert (
        client.post("/api/attributes/resolve", json={"names": ["坏=名字"]}).status_code == 422
    )
    assert client.post("/api/attributes/resolve", json={"names": []}).status_code == 422

    # 2) 上传时关联多个属性：auto_create 打开 → 未定义属性名自动补建并保留
    path = Path(work_dir) / "auto_attr.bin"
    path.write_bytes(bytes((index * 11) % 251 for index in range(500_000)))
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()

    response = client.post(
        "/api/uploads/init",
        json={
            "attributes": {"航次": "C2026-03", "波束角": 45.5, "已复核": True},
            "auto_create_attributes": True,
            "files": [{"client_key": "a", "name": path.name, "size": len(data), "sha256": sha}],
        },
    )
    assert response.status_code == 200, response.text
    session_id = response.json()["items"][0]["session_id"]
    cs = response.json()["items"][0]["chunk_size"]
    total = response.json()["items"][0]["total_chunks"]
    for index in range(total):
        _put(client, session_id, index, data[index * cs : (index + 1) * cs])
    file_id = client.post(f"/api/uploads/{session_id}/complete").json()["file_id"]

    detail = client.get(f"/api/files/{file_id}").json()
    assert detail["attributes"]["航次"] == "C2026-03"
    assert detail["attributes"]["波束角"] == "45.5"   # 数字按字符串存
    assert detail["attributes"]["已复核"] == "true"    # 布尔按 true/false 存

    # 新建的属性立刻可用于检索（"上传关联 → 检索关联"闭环）
    assert client.get("/api/files", params={"attr": "航次=C2026-03"}).json()["total"] == 1
    assert client.get("/api/files", params={"attr": "波束角=45.5"}).json()["total"] == 1
    assert client.get("/api/files", params={"attr": "已复核=true"}).json()["total"] == 1
    assert client.get("/api/files", params={"attr": "航次=C2026-99"}).json()["total"] == 0

    # 已有取值可直接"选"而不是手敲：GET /api/attributes/values
    all_values = client.get("/api/attributes/values").json()["items"]
    by_name = {item["name"]: item for item in all_values}
    assert [entry["value"] for entry in by_name["航次"]["items"]] == ["C2026-03"]
    assert by_name["航次"]["items"][0]["file_count"] == 1
    assert [entry["value"] for entry in by_name["波束角"]["items"]] == ["45.5"]
    assert [entry["value"] for entry in by_name["已复核"]["items"]] == ["true"]
    # 只查一个属性 + 还没被用过的属性返回空（用专门建的探测属性，避免依赖其它用例）
    single = client.get("/api/attributes/values", params={"name": "航次"}).json()["items"]
    assert len(single) == 1 and single[0]["name"] == "航次"
    client.post("/api/attributes/resolve", json={"names": ["空属性"]})
    probe = client.get("/api/attributes/values", params={"name": "空属性"}).json()["items"]
    assert probe and probe[0]["items"] == [] and probe[0]["total"] == 0
    assert client.get("/api/attributes/values", params={"name": "不存在"}).status_code == 404

    # 开关关闭时仍然严格报错，且不会污染属性库
    strict = client.post(
        "/api/uploads/init",
        json={
            "attributes": {"从没建过": 1},
            "files": [{"client_key": "b", "name": path.name, "size": len(data), "sha256": sha}],
        },
    )
    assert strict.status_code == 422 and "未定义" in strict.json()["detail"]
    assert "从没建过" not in {
        item["name"] for item in client.get("/api/attributes").json()["items"]
    }

    # 3) 单文件属性编辑：create_missing → 新属性不丢值且写入属性库；默认仍是"忽略并回报"
    response = client.put(
        f"/api/files/{file_id}/attributes",
        json={
            "values": [
                {"name": "航次", "value": "C2026-04"},
                {"name": "作业人员", "value": "张三"},
            ],
            "create_missing": True,
        },
    )
    assert response.status_code == 200, response.text
    assert "ignored_names" not in response.json()
    assert response.json()["attributes"]["作业人员"] == "张三"
    assert "作业人员" in {
        item["name"] for item in client.get("/api/attributes").json()["items"]
    }

    ignored = client.put(
        f"/api/files/{file_id}/attributes",
        json={"values": [{"name": "幽灵属性", "value": 1}]},
    ).json()
    assert ignored["ignored_names"] == ["幽灵属性"]
    assert "幽灵属性" not in {
        item["name"] for item in client.get("/api/attributes").json()["items"]
    }
