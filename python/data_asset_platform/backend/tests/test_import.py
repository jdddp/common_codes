"""服务端按路径导入（存量数据迁移）。"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.config import settings
from tests.synth_hdy import build_hdy_file


def test_import_directory_and_parse(client, work_dir):
    source = Path(work_dir) / "legacy_cruise"
    if source.exists():
        shutil.rmtree(source)
    (source / "nav").mkdir(parents=True)

    build_hdy_file(source / "sonar_a.HDY", frames=6, n_beams=64, n_ranges=128)
    build_hdy_file(source / "nav" / "sonar_b.HDY", frames=5, n_beams=64, n_ranges=128, seed=7)
    (source / "readme.txt").write_text("legacy data", encoding="utf-8")

    response = client.post(
        "/api/import",
        json={
            "paths": [str(source)],
            "root_name": "legacy_cruise",
            "recursive": True,
        },
    )
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert len(items) == 3
    assert all(item["status"] == "done" for item in items)
    assert source.exists()  # 默认复制，原文件保留

    rel_paths = sorted(item["rel_path"] for item in items)
    # 目录层级原样保留：根目录下 2 个文件、nav 子目录 1 个文件
    assert rel_paths == ["legacy_cruise", "legacy_cruise", "legacy_cruise/nav"]

    # 原文件仍在磁盘上（复制语义）
    assert (source / "sonar_a.HDY").exists()

    # 导入后逐个解析，可在线查看
    for item in items:
        if item["name"].endswith(".HDY"):
            result = client.post(f"/api/files/{item['file_id']}/parse").json()
            assert result["parse_status"] == "ready"
            assert result["protocol"] == "hdy"

    info = client.get(
        f"/api/sonar/{[i for i in items if i['name'] == 'sonar_a.HDY'][0]['file_id']}/info"
    ).json()
    assert info["supported"] is True
    assert info["frame_count"] == 6

    # 再导入一次：内容相同 ⇒ 秒传，不重复占空间
    response = client.post(
        "/api/import",
        json={"paths": [str(source)], "root_name": "legacy_cruise"},
    )
    assert response.status_code == 200
    assert all(item["deduped"] for item in response.json()["items"])

    blobs = [p for p in settings.blobs_dir.rglob("*") if p.is_file()]
    # 目录内 3 个文件 ⇒ 3 个实体（第二轮全部命中，未新增）
    assert len(blobs) >= 3


def test_import_move_and_errors(client, work_dir):
    single = Path(work_dir) / "to_move.HDY"
    build_hdy_file(single, frames=3, n_beams=32, n_ranges=64)

    response = client.post(
        "/api/import",
        json={"paths": [str(single)], "move": True, "root_name": "moved"},
    )
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["status"] == "done"
    assert item["moved"] is True
    assert not single.exists()  # move 语义
    assert item["rel_path"] == "moved"

    # 路径不存在 → 该项失败但整体 200（批量导入不因单个失败中断）
    response = client.post(
        "/api/import",
        json={"paths": ["Z:/definitely/not/here"]},
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["status"] == "failed"

    # 空 paths → 422
    response = client.post("/api/import", json={"paths": []})
    assert response.status_code == 422
