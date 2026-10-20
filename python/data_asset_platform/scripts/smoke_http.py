"""HTTP 冒烟测试：对着真实运行的服务跑一遍"导入 → 解析 → 在线查看 → 下载"闭环。

用法：
    python scripts/smoke_http.py --base-url http://127.0.0.1:8000 --sample "D:\\data\\水L-500KHz-6节-1(1).dat"
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote


def call(base: str, method: str, path: str, payload=None, raw: bytes | None = None):
    url = base.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
        headers["Content-Type"] = "application/octet-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
            ctype = response.headers.get("content-type", "")
            if "json" in ctype:
                return response.status, json.loads(body.decode("utf-8")), response.headers
            return response.status, body, response.headers
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            return error.code, json.loads(body.decode("utf-8")), error.headers
        except Exception:
            return error.code, body, error.headers


def upload_file(
    base: str, name: str, payload: bytes, attributes: dict, auto_create: bool = True
) -> int | None:
    """上传一个文件并带上属性，返回 file_id；失败返回 None。"""
    sha = hashlib.sha256(payload).hexdigest()
    status, body, _ = call(
        base,
        "POST",
        "/api/uploads/init",
        {
            "attributes": attributes,
            "auto_create_attributes": auto_create,
            "files": [
                {"client_key": name, "name": name, "size": len(payload), "sha256": sha}
            ],
        },
    )
    if status != 200:
        return None
    item = body["items"][0]
    if item.get("status") == "done":
        return item.get("file_id")
    session, chunk = item["session_id"], item["chunk_size"]
    for index in range(item["total_chunks"]):
        call(
            base,
            "PUT",
            f"/api/uploads/{session}/chunks/{index}",
            raw=payload[index * chunk : (index + 1) * chunk],
        )
    status, body, _ = call(base, "POST", f"/api/uploads/{session}/complete")
    return body.get("file_id") if status == 200 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="平台 HTTP 冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sample", default=None, help="真实样例文件路径（可选）")
    args = parser.parse_args()

    base = args.base_url
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        print(f"[{'OK ' if ok else 'FAIL'}] {name} {detail}")

    # 1) 健康检查
    status, body, _ = call(base, "GET", "/api/health")
    check("GET /api/health", status == 200 and body.get("status") == "ok", str(body)[:120])

    # 2) 免构建页面
    for path, expect in (("/static/index.html", "内部数据资产管理平台"), ("/static/viewer.html", "sonar")):
        status, body, _ = call(base, "GET", path)
        text = body.decode("utf-8", "ignore") if isinstance(body, bytes) else str(body)
        check(f"GET {path}", status == 200 and expect in text, f"{len(text)} 字节")

    status, body, _ = call(base, "GET", "/")
    text = body.decode("utf-8", "ignore") if isinstance(body, bytes) else str(body)
    check("GET / (首页)", status == 200 and "<html" in text.lower(), f"{len(text)} 字节")

    # 3) 服务端导入（复制真实样例到内容寻址存储）
    if args.sample:
        sample = Path(args.sample)
        if not sample.exists():
            check("导入真实样例", False, f"文件不存在: {sample}")
        else:
            t0 = time.time()
            status, body, _ = call(
                base,
                "POST",
                "/api/import",
                {
                    "paths": [str(sample)],
                    "root_name": "raw",
                },
            )
            item = body["items"][0]
            check(
                "导入真实样例（服务端复制，不走浏览器上传）",
                status == 200 and item["status"] == "done",
                f"{item.get('size_bytes', 0) / 1024 / 1024:.1f} MB，用时 {time.time() - t0:.2f}s，去重={item.get('deduped')}",
            )
            file_id = item["file_id"]

            t0 = time.time()
            status, body, _ = call(base, "POST", f"/api/files/{file_id}/parse")
            check(
                "解析（建帧索引 + 缩略图）",
                status == 200 and body.get("parse_status") == "ready",
                f"协议={body.get('protocol')} 帧数={body.get('frame_count')} 尺寸={body.get('image_shape')} 用时 {time.time() - t0:.2f}s",
            )

            status, body, _ = call(base, "GET", f"/api/sonar/{file_id}/info")
            check(
                "查看能力探测",
                status == 200 and body.get("supported"),
                f"帧数={body.get('frame_count')} 扇形映射={body.get('capabilities', {}).get('sector_fan')}",
            )

            t0 = time.time()
            status, body, _ = call(base, "GET", f"/api/sonar/{file_id}/view?index=100")
            gray = len(base64.b64decode(body["gray"]["data"]))
            color = len(base64.b64decode(body["color"]["data"]))
            check(
                "取第 100 帧（双栏）",
                status == 200 and gray > 500 and color > 500,
                f"灰度={gray}B 伪彩={color}B ping={body['meta']['ping_num']} 用时 {time.time() - t0:.3f}s",
            )

            t0 = time.time()
            status, body, _ = call(
                base,
                "GET",
                f"/api/sonar/{file_id}/frames?start=200&count=16&step=1&algos=CLAHE&sector_fan=true&colormap=jet",
            )
            check(
                "批量取帧（播放用，含算法+扇形）",
                status == 200 and len(body["frames"]) == 16,
                f"16 帧用时 {time.time() - t0:.3f}s",
            )

            status, body, _ = call(base, "GET", f"/api/sonar/{file_id}/pixel?index=100&x=200&y=300")
            check("悬停取值", status == 200 and "raw" in body, f"raw={body.get('raw')} processed={body.get('processed')}")

            # 越界坐标必须拒绝（历史缺陷：静默钳制会让悬停显示错误像素）
            status, _body, _ = call(base, "GET", f"/api/sonar/{file_id}/pixel?index=100&x=99999&y=0")
            check("悬停取值越界拒绝", status == 422, f"HTTP {status}")

            # 扇形映射会重采样出更宽的伪彩画布，两栏取值必须按栏区分
            status, fan_body, _ = call(base, "GET", f"/api/sonar/{file_id}/view?index=100&sector_fan=true")
            gray_shape = (fan_body.get("shapes") or {}).get("gray")
            color_shape = (fan_body.get("shapes") or {}).get("color")
            check(
                "扇形映射两栏画布尺寸",
                status == 200 and bool(gray_shape) and bool(color_shape) and color_shape != gray_shape,
                f"灰度={gray_shape} 伪彩={color_shape}",
            )
            color_x = max(0, min((color_shape or [0, 1])[1] - 1, 400))
            status, body, _ = call(
                base,
                "GET",
                f"/api/sonar/{file_id}/pixel?index=100&x={color_x}&y=300&sector_fan=true&panel=color",
            )
            check(
                "伪彩栏取值（映射空间）",
                status == 200 and body.get("panel") == "color" and body.get("processed") is not None,
                f"processed={body.get('processed')} raw={body.get('raw')} shape={body.get('panel_shape')}",
            )

            status, body, _ = call(base, "GET", f"/api/sonar/{file_id}/frame/100.png?panel=color")
            check("单栏 PNG 帧图", status == 200 and body[:4] == b"\x89PNG", f"{len(body) / 1024:.1f} KB")

            status, body, headers = call(base, "GET", f"/api/files/{file_id}/download")
            check(
                "原始文件下载",
                status == 200 and len(body) > 1024,
                f"{len(body) / 1024 / 1024:.1f} MB",
            )

            status, body, _ = call(base, "GET", f"/viewer/{file_id}")
            check("直达查看页 /viewer/{id}", status == 200, f"{len(body) if isinstance(body, bytes) else 0} 字节")

    # 4) 属性（只有名称，值一律是字符串）：库里已有 → 复用；库里没有 → 新建并保留
    status, body, _ = call(
        base,
        "POST",
        "/api/attributes/resolve",
        {"names": ["smoke_cruise", "smoke_cruise"]},
    )
    check(
        "属性批量取或建（幂等：同名只建一次）",
        status == 200
        and body["created"] == ["smoke_cruise"]
        and [entry["name"] for entry in body["items"]] == ["smoke_cruise"],
        f"created={body['created']} existing={body['existing']}",
    )

    status, body, _ = call(base, "POST", "/api/attributes/resolve", {"names": ["smoke_cruise"]})
    check(
        "属性已有则复用（不覆盖原属性）",
        status == 200 and body["created"] == [] and body["existing"] == ["smoke_cruise"],
        f"created={body['created']} existing={body['existing']}",
    )

    status, _body, _ = call(base, "POST", "/api/attributes/resolve", {"names": ["坏=名字"]})
    check(
        "属性名里不能带 =（筛选语法用它分隔属性名和值）",
        status == 422,
        f"HTTP {status}",
    )

    # 用一个临时文件走一遍"上传时关联多个属性 + 库里没有的自动补键"
    with tempfile.TemporaryDirectory() as tmpdir:
        attr_file = Path(tmpdir) / "smoke_attr.bin"
        attr_file.write_bytes(bytes((index * 13) % 251 for index in range(200_000)))
        payload_bytes = attr_file.read_bytes()
        sha = hashlib.sha256(payload_bytes).hexdigest()
        status, body, _ = call(
            base,
            "POST",
            "/api/uploads/init",
            {
                "attributes": {"smoke_beam": 45.5, "smoke_flag": True},
                "auto_create_attributes": True,
                "files": [
                    {
                        "client_key": "attr1",
                        "name": attr_file.name,
                        "size": len(payload_bytes),
                        "sha256": sha,
                    }
                ],
            },
        )
        item = body["items"][0] if status == 200 else {}
        if status == 200 and item.get("status") != "done":
            session = item["session_id"]
            chunk = item["chunk_size"]
            for index in range(item["total_chunks"]):
                call(
                    base,
                    "PUT",
                    f"/api/uploads/{session}/chunks/{index}",
                    raw=payload_bytes[index * chunk : (index + 1) * chunk],
                )
            status, body, _ = call(base, "POST", f"/api/uploads/{session}/complete")
            attr_file_id = body.get("file_id")
        else:
            attr_file_id = item.get("file_id")

        status, detail, _ = call(base, "GET", f"/api/files/{attr_file_id}")
        check(
            "上传时关联多个属性（库里没有的自动建属性并保留取值，值一律按文本存）",
            status == 200
            and detail["attributes"].get("smoke_beam") == "45.5"
            and detail["attributes"].get("smoke_flag") == "true",
            f"attributes={detail['attributes']}",
        )

    status, listing, _ = call(base, "GET", "/api/attributes")
    definitions = {entry["name"]: entry for entry in listing["items"]}
    check(
        "自动补建的属性只有名称，没有类型概念",
        "smoke_beam" in definitions
        and "smoke_flag" in definitions
        and "value_type" not in definitions["smoke_beam"],
        f"smoke_beam={definitions.get('smoke_beam')}",
    )

    status, listing, _ = call(base, "GET", "/api/files?attr=smoke_beam=45.5")
    check("属性检索：等于", status == 200 and listing["total"] >= 1, f"total={listing['total']}")
    status, listing, _ = call(base, "GET", "/api/files?attr=smoke_beam=45.6")
    check("属性检索不匹配时为 0", status == 200 and listing["total"] == 0, f"total={listing['total']}")
    status, _body, _ = call(base, "GET", "/api/files?attr=smoke_beam!=99")
    check(
        "属性检索只有「等于」：`!=` 会当成属性名 → 属性不存在，422",
        status == 422,
        f"HTTP {status}",
    )
    status, _body, _ = call(base, "GET", "/api/files?attr=smoke_beam%3E40")
    check(
        "大于/小于这类比较已取消（值都是字符串，只有 等于）",
        status == 422,
        f"HTTP {status}",
    )

    # ---- 内置「日期」：真实日期值，唯一支持时间范围筛选的属性
    date_def = definitions.get("日期")
    check(
        "内置属性「日期」自动建立并标记 is_builtin",
        bool(date_def) and date_def.get("is_builtin") is True,
        f"date={date_def}",
    )
    if date_def:
        status, _body, _ = call(base, "DELETE", f"/api/attributes/{date_def['id']}")
        check("内置「日期」不可删除", status == 409, f"HTTP {status}")
        status, _body, _ = call(
            base, "PATCH", f"/api/attributes/{date_def['id']}", {"name": "时间"}
        )
        check("内置「日期」不可改名", status == 422, f"HTTP {status}")

    # 不合法的日期必须在业务层被拦下（422），不能漏成 500，更不能悄悄存进去
    status, _body, _ = call(
        base,
        "POST",
        "/api/uploads/init",
        {
            "attributes": {"日期": "2031-02-30"},
            "files": [
                {
                    "client_key": "smoke-bad-date",
                    "name": "smoke_bad_date.bin",
                    "size": 4,
                    "sha256": hashlib.sha256(b"bad!").hexdigest(),
                }
            ],
        },
    )
    check("非法日期（2031-02-30）上传时被拦下", status == 422, f"HTTP {status}")

    # 两个文件、两种写法：斜杠写法照样要被归一成 YYYY-MM-DD
    date_ids = {}
    for label, value in (("a", "2031/03/15"), ("b", "2031-03-20")):
        payload = f"smoke-date-{label}-".encode("ascii") * 200
        date_ids[label] = upload_file(base, f"smoke_date_{label}.bin", payload, {"日期": value})
    stored = {}
    for label, file_id in date_ids.items():
        if not file_id:
            continue
        _status, detail, _ = call(base, "GET", f"/api/files/{file_id}")
        stored[label] = (detail.get("attributes") or {}).get("日期")
    check(
        "日期写入时归一成 YYYY-MM-DD（斜杠写法也认）",
        stored.get("a") == "2031-03-15" and stored.get("b") == "2031-03-20",
        f"stored={stored}",
    )

    # 时间范围筛选：命中 / 区间内没有 / 单边
    both = f"attr={quote('日期>=')}2031-03-16&attr={quote('日期<=')}2031-03-31"
    status, listing, _ = call(base, "GET", f"/api/files?{both}")
    check(
        "时间范围命中（>=03-16 且 <=03-31 → 只有 b）",
        status == 200 and listing["total"] == 1,
        f"total={listing['total']}",
    )
    gap = f"attr={quote('日期>=')}2031-03-16&attr={quote('日期<=')}2031-03-19"
    status, listing, _ = call(base, "GET", f"/api/files?{gap}")
    check("时间范围区间内没有 → 0", status == 200 and listing["total"] == 0, f"total={listing['total']}")
    one = f"attr={quote('日期<=')}2031-03-15"
    status, listing, _ = call(base, "GET", f"/api/files?{one}")
    check(
        "单边时间范围（<=03-15 → 只有 a）",
        status == 200 and listing["total"] == 1,
        f"total={listing['total']}",
    )
    # 范围只对内置日期开放
    status, _body, _ = call(base, "GET", "/api/files?attr=smoke_beam%3E%3D40")
    check("普通属性不许用时间范围", status == 422, f"HTTP {status}")

    # 已有取值可直接枚举：筛选/填写时从下拉里挑，而不是必须手敲
    status, body, _ = call(base, "GET", "/api/attributes/values?name=smoke_beam")
    entry = (body.get("items") or [{}])[0] if status == 200 else {}
    values = [item["value"] for item in entry.get("items", [])]
    check(
        "已有取值可枚举（值是库里出现过的，可直接选）",
        status == 200 and "45.5" in values and entry.get("total", 0) >= 1,
        f"HTTP {status} values={values}",
    )

    # 上传区的取值候选读的是"不带 name 的全量接口"：刚上传用过的新值必须已经出现在里面，
    # 这样下一次上传就能直接复用；没出现过的值仍然允许手输（写入侧不校验候选列表）
    status, body, _ = call(base, "GET", "/api/attributes/values")
    by_name = {row["name"]: row for row in body.get("items", [])} if status == 200 else {}
    beam_values = [item["value"] for item in by_name.get("smoke_beam", {}).get("items", [])]
    check(
        "上传用过的新值已进入下一次上传的候选（上传区读的全量接口）",
        status == 200 and "45.5" in beam_values,
        f"HTTP {status} smoke_beam={beam_values}",
    )

    status, body, _ = call(
        base,
        "PUT",
        f"/api/files/{attr_file_id}/attributes",
        {"values": [{"name": "smoke_operator", "value": "张工"}], "create_missing": True},
    )
    check(
        "单文件属性：create_missing 时不丢值",
        status == 200 and "ignored_names" not in body and body["attributes"].get("smoke_operator") == "张工",
        f"attributes={body['attributes']}",
    )

    status, body, _ = call(
        base,
        "PUT",
        f"/api/files/{attr_file_id}/attributes",
        {"values": [{"name": "smoke_ghost", "value": 1}]},
    )
    check(
        "单文件属性：默认仍是忽略未知属性名并回报",
        status == 200 and body.get("ignored_names") == ["smoke_ghost"],
        f"ignored={body.get('ignored_names')}",
    )

    # 属性库管理：建一个没用过的属性可以删；已被文件用过的属性禁止删除
    status, body, _ = call(
        base,
        "POST",
        "/api/attributes/resolve",
        {"names": ["smoke_unused"]},
    )
    unused_id = body["items"][0]["id"] if status == 200 else 0
    status, listing, _ = call(base, "GET", "/api/attributes")
    unused = next(
        (entry for entry in listing["items"] if entry["name"] == "smoke_unused"), {}
    )
    status, body, _ = call(base, "GET", "/api/attributes/values?name=smoke_unused")
    entry = (body.get("items") or [{}])[0] if status == 200 else {}
    check(
        "没有任何取值的属性：候选列表为空",
        status == 200 and entry.get("items") == [] and entry.get("total") == 0,
        f"HTTP {status} items={entry.get('items')}",
    )

    status, body, _ = call(base, "DELETE", f"/api/attributes/{unused_id}")
    check(
        "属性库管理：没有任何取值时可以删除该属性",
        status == 200 and unused.get("used_count") == 0,
        f"used_count={unused.get('used_count')}",
    )

    used_id = next(
        (
            entry["id"]
            for entry in call(base, "GET", "/api/attributes")[1]["items"]
            if entry["name"] == "smoke_operator"
        ),
        0,
    )
    status, body, _ = call(base, "DELETE", f"/api/attributes/{used_id}")
    check(
        "属性库管理：已有关联取值的属性禁止删除",
        status == 409 and "禁止删除" in str(body),
        f"HTTP {status}",
    )

    # 5) 汇总
    failed = [name for name, ok, _ in checks if not ok]
    print("\n================ 汇总 ================")
    print(f"通过 {len(checks) - len(failed)} / {len(checks)}")
    if failed:
        print("失败项：" + "，".join(failed))
        raise SystemExit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
