"""上传链路：分块、断点续传、秒传去重、下载与删除。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import settings

CHUNK = 300_000


def _payload_file(path: Path, size: int, seed: int = 1) -> Path:
    data = bytes((index * 31 + seed) % 251 for index in range(size))
    path.write_bytes(data)
    return path


def _init(client, path: Path, size: int, sha: str | None, chunk_size: int = CHUNK):
    response = client.post(
        "/api/uploads/init",
        json={
            "root_name": "raw",
            "files": [
                {
                    "client_key": "k1",
                    "name": path.name,
                    "rel_path": "nav/day1",
                    "size": size,
                    "sha256": sha,
                    "chunk_size": chunk_size,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["items"][0]


def _put(client, session_id: str, index: int, chunk: bytes):
    response = client.put(
        f"/api/uploads/{session_id}/chunks/{index}",
        content=chunk,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload(client, path: Path, chunk_size: int = CHUNK, with_sha: bool = True):
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest() if with_sha else None

    item = _init(client, path, len(data), sha, chunk_size)
    if item["status"] == "done":
        return item, data

    session_id = item["session_id"]
    total = item["total_chunks"]
    csize = item["chunk_size"]

    def chunk_of(index: int) -> bytes:
        return data[index * csize : (index + 1) * csize]

    todo = [index for index in range(total) if index not in set(item["received"])]
    # 先只传第一个分片，模拟"传一半断了"
    if todo:
        _put(client, session_id, todo[0], chunk_of(todo[0]))

    # 再次 init：应复用同一会话，并回报已收到的分片（断点续传）
    resumed = _init(client, path, len(data), sha, chunk_size)
    assert resumed["session_id"] == session_id
    assert todo[0] in set(resumed["received"])

    for index in range(total):
        if index in set(resumed["received"]):
            continue
        _put(client, session_id, index, chunk_of(index))

    progress = client.get(f"/api/uploads/{session_id}").json()
    assert progress["missing"] == []

    response = client.post(f"/api/uploads/{session_id}/complete")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "done"
    return result, data


def test_upload_resume_and_dedupe(client, work_dir):
    path = _payload_file(work_dir / "sample_payload.bin", 1_000_000)
    result, data = _upload(client, path)
    file_id = result["file_id"]
    sha = hashlib.sha256(data).hexdigest()

    def blobs_of(sha256: str) -> list[Path]:
        return [p for p in settings.blobs_dir.rglob(f"{sha256}*") if p.is_file()]

    detail = client.get(f"/api/files/{file_id}").json()
    assert detail["size_bytes"] == len(data)
    assert detail["original_name"] == "sample_payload.bin"
    assert detail["rel_path"] == "raw/nav/day1"
    assert len(blobs_of(sha)) == 1

    # 秒传：同内容再传一次，直接 done 且共享同一实体
    item, _ = _upload(client, path)
    assert item["status"] == "done"
    assert item["file_id"] != file_id
    assert "秒传" in item["message"]

    # 存储实体按内容寻址：内容相同只有一份
    assert len(blobs_of(sha)) == 1

    # 下载校验（字节级一致）
    response = client.get(f"/api/files/{file_id}/download")
    assert response.status_code == 200
    assert response.content == data

    # 删除其中一个引用：实体因引用计数保留
    assert client.delete(f"/api/files/{file_id}").status_code == 200
    assert len(blobs_of(sha)) == 1

    # 删除最后一个引用后实体释放
    assert client.delete(f"/api/files/{item['file_id']}").status_code == 200
    assert blobs_of(sha) == []


def test_chunk_validation_and_missing_chunks(client, work_dir):
    path = _payload_file(work_dir / "small.bin", 400_000, seed=9)
    data = path.read_bytes()
    item = _init(client, path, len(data), None)
    session_id = item["session_id"]
    assert item["total_chunks"] == 2

    # 分片大小不符 → 422
    response = client.put(
        f"/api/uploads/{session_id}/chunks/0",
        content=data[:1000],
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422

    # 序号越界 → 422
    response = client.put(
        f"/api/uploads/{session_id}/chunks/9",
        content=data[:CHUNK],
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 422

    # 缺分片时不允许合并
    _put(client, session_id, 0, data[:CHUNK])
    response = client.post(f"/api/uploads/{session_id}/complete")
    assert response.status_code == 422
    assert "未上传" in response.json()["detail"]

    # 上传剩余分片后可合并
    _put(client, session_id, 1, data[CHUNK:])
    assert client.post(f"/api/uploads/{session_id}/complete").status_code == 200
