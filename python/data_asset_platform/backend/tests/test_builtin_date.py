"""内置属性「日期」与时间范围筛选（v1.8）。

口径：
  * `日期` 是平台自动建立的**内置属性**：不可删除、不可改名、不可重复创建
  * 值归一成 `YYYY-MM-DD`（定宽 ISO 文本 ⇒ 字典序即时间序）
  * **只有**它支持时间范围 `>=` / `<=`；普通属性仍然只有「等于」
  * 传了不合法的日期一律 422，不能漏成 500
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from tests.test_upload import _put


def _upload(client, work_dir, name, payload, attributes):
    """上传一个文件并带上属性。内容寻址会去重，所以每个文件的字节必须不同。"""
    path = Path(work_dir) / name
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    response = client.post(
        "/api/uploads/init",
        json={
            "attributes": attributes,
            "files": [{"client_key": "k", "name": name, "size": len(payload), "sha256": sha}],
        },
    )
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    session_id, total, chunk = item["session_id"], item["total_chunks"], item["chunk_size"]
    for index in range(total):
        _put(client, session_id, index, payload[index * chunk : (index + 1) * chunk])
    result = client.post(f"/api/uploads/{session_id}/complete").json()
    assert "file_id" in result, result
    return int(result["file_id"])


def _count(client, *conditions):
    """按属性条件数文件个数（支持传多条，条件之间是「并且」）。"""
    params = [("attr", condition) for condition in conditions]
    return client.get("/api/files", params=params).json()["total"]


def test_builtin_date_is_present_and_protected(client):
    items = client.get("/api/attributes").json()["items"]
    date_item = next((item for item in items if item["name"] == "日期"), None)
    assert date_item is not None, "内置属性「日期」必须自动存在"
    assert date_item["is_builtin"] is True
    # 其余属性都是普通属性
    assert all(item["is_builtin"] is False for item in items if item["name"] != "日期")

    # 不可删除、不可改名、不可重复创建（名称即标识，所以也拦了重名）
    assert client.delete(f"/api/attributes/{date_item['id']}").status_code == 409
    renamed = client.patch(f"/api/attributes/{date_item['id']}", json={"name": "时间"})
    assert renamed.status_code == 422
    assert client.post("/api/attributes", json={"name": "日期"}).status_code == 409


def test_date_value_is_normalized_and_range_filterable(client, work_dir):
    """三种写法各上传一个：全部归一成 YYYY-MM-DD，然后按时间范围筛。"""
    files = [
        ("builtin_date_a.bin", "日期", "2041/03/01"),      # 斜杠写法
        ("builtin_date_b.bin", "日期", "2041-03-15 09:30:00"),  # 带时间（只取日期部分）
        ("builtin_date_c.bin", "日期", "20410420"),         # 紧凑写法
    ]
    ids = {}
    for offset, (name, _key, value) in enumerate(files):
        payload = f"builtin-date-{offset}".encode() * 40
        ids[value] = _upload(client, work_dir, name, payload, {"日期": value})

    # 落库的值都是归一后的 YYYY-MM-DD
    expected = {
        "2041/03/01": "2041-03-01",
        "2041-03-15 09:30:00": "2041-03-15",
        "20410420": "2041-04-20",
    }
    for raw, file_id in ids.items():
        detail = client.get(f"/api/files/{file_id}").json()
        assert detail["attributes"]["日期"] == expected[raw]

    # 时间范围（两端都夹在 2041 年内，避免受其它用例的日期影响）
    assert _count(client, "日期>=2041-01-01", "日期<=2041-12-31") == 3
    assert _count(client, "日期>=2041-03-15", "日期<=2041-12-31") == 2
    assert _count(client, "日期>=2041-01-01", "日期<=2041-03-15") == 2
    assert _count(client, "日期>=2041-03-02", "日期<=2041-04-19") == 1   # 只剩中间那个
    assert _count(client, "日期>=2041-03-02", "日期<=2041-03-14") == 0   # 区间内没有
    # 「等于」对日期同样成立（定宽文本按等于比即可）
    assert _count(client, "日期=2041-03-15") == 1


def test_invalid_date_is_422_not_500(client, work_dir):
    """不合法的日期必须在业务层被拦下（422），不能变成 500，更不能悄悄存进去。"""
    payload = b"builtin-date-invalid" * 20

    for bad in ("乱写的", "2026-02-30", "2026-13-01", "2026"):
        response = client.post(
            "/api/uploads/init",
            json={
                "attributes": {"日期": bad},
                "files": [
                    {"client_key": "k", "name": "bad.bin", "size": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()}
                ],
            },
        )
        assert response.status_code == 422, (bad, response.status_code, response.text)
        assert response.status_code != 500

    # 单文件属性编辑走的是另一条路径，同样要 422（原来是 ValueError → 500）
    path = Path(work_dir) / "builtin_date_edit.bin"
    path.write_bytes(payload)
    response = client.post(
        "/api/uploads/init",
        json={
            "attributes": {"日期": "2043-06-07"},
            "files": [{"client_key": "k", "name": path.name, "size": len(payload),
                       "sha256": hashlib.sha256(payload).hexdigest()}],
        },
    )
    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    session_id, total, chunk = item["session_id"], item["total_chunks"], item["chunk_size"]
    for index in range(total):
        _put(client, session_id, index, payload[index * chunk : (index + 1) * chunk])
    result = client.post(f"/api/uploads/{session_id}/complete").json()
    file_id = int(result["file_id"])

    ok = client.put(
        f"/api/files/{file_id}/attributes",
        json={"replace": False, "create_missing": False, "values": [{"name": "日期", "value": "2043/07/08"}]},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["attributes"]["日期"] == "2043-07-08"   # 编辑路径同样归一

    bad = client.put(
        f"/api/files/{file_id}/attributes",
        json={"replace": False, "create_missing": False, "values": [{"name": "日期", "value": "某年某月"}]},
    )
    assert bad.status_code == 422, bad.text
    # 值没被写坏
    assert client.get(f"/api/files/{file_id}").json()["attributes"]["日期"] == "2043-07-08"


def test_range_only_allowed_on_builtin_date(client):
    """普通属性不许用时间范围（否则会让人误以为按字符串比大小有意义）。"""
    _create = client.post("/api/attributes", json={"name": "海域"})
    assert _create.status_code in (200, 409), _create.text

    response = client.get("/api/files", params={"attr": "海域>=黄海"})
    assert response.status_code == 422
    assert "范围" in response.json()["detail"]

    assert client.get("/api/files", params={"attr": "日期<=2041-12-31"}).status_code == 200
    # 日期范围的一端为空 → 422
    assert client.get("/api/files", params={"attr": "日期>="}).status_code == 422


def test_empty_date_means_no_date(client, work_dir):
    """留空 = 该文件没有「日期」这个属性（与其它属性一致），不会落一行空值。"""
    payload = b"builtin-date-empty" * 30
    file_id = _upload(client, work_dir, "builtin_date_empty.bin", payload, {"日期": ""})
    detail = client.get(f"/api/files/{file_id}").json()
    assert "日期" not in detail["attributes"]
    # 任何时间范围都筛不到它——凡是有日期的文件都在结果里，唯独它不在
    matched = client.get(
        "/api/files",
        params=[("attr", "日期>=2000-01-01"), ("attr", "日期<=2099-12-31")],
    ).json()
    assert matched["total"] == 4            # 2041 年 3 个 + 2043 年 1 个
    assert file_id not in {row["id"] for row in matched["items"]}
