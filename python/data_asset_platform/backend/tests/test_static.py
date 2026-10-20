"""内置免构建页面与健康检查。"""
from __future__ import annotations


def test_health(client):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert "storage_root" in data


def test_static_index_page(client):
    response = client.get("/static/index.html")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "内部数据资产管理平台" in body
    assert "/api/files" in body or "API = '/api'" in body
    # 服务端导入入口（存量数据迁移）必须在页面上
    assert "服务端导入" in body
    assert "'/import'" in body


def test_static_viewer_page(client):
    response = client.get("/static/viewer.html")
    assert response.status_code == 200
    body = response.text
    # 关键语义必须在页面里（与桌面版一致）
    assert "Math.trunc(BASE_INTERVAL / speed)" in body or "BASE_INTERVAL / speed" in body
    assert "sonar" in body


def test_viewer_shortcut_redirect(client):
    response = client.get("/viewer/12", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/static/viewer.html?file=12"


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
