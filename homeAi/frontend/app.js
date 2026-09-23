"use strict";

const STATE = {
  lastId: 0,
  records: 0,
  ws: null,
  reconnectDelay: 3000,
  livePaused: false,
  day: "", // "" = 全部, 否则为 "YYYY-MM-DD"
};

const $ = (sel) => document.querySelector(sel);

// ---------- 渲染记录 ----------
const TEMPLATE = document.getElementById("tpl-record").content;

function fmtTime(tsMs) {
  const d = new Date(tsMs);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
}

// ---------- 按天分组 ----------
function dayKey(ts) {
  const d = new Date(ts);
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${dd}`;
}

function dayLabel(key) {
  const [y, m, dd] = key.split("-").map(Number);
  const week = ["日", "一", "二", "三", "四", "五", "六"][new Date(y, m - 1, dd).getDay()];
  const base = `${m}月${dd}日 周${week}`;
  const now = new Date();
  if (key === dayKey(now.getTime())) return `今天 · ${base}`;
  const yKey = dayKey(new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1).getTime());
  if (key === yKey) return `昨天 · ${base}`;
  return base;
}

function sectionFor(listEl, day) {
  let sec = listEl.querySelector(`.day-group[data-day="${day}"]`);
  if (sec) return sec;
  sec = document.createElement("section");
  sec.className = "day-group";
  sec.dataset.day = day;
  const head = document.createElement("h3");
  head.className = "day-head";
  head.textContent = dayLabel(day);
  const body = document.createElement("div");
  body.className = "day-list";
  sec.append(head, body);
  let ref = null;
  for (const s of listEl.children) {
    if (s.dataset.day && s.dataset.day < day) { ref = s; break; }
  }
  listEl.insertBefore(sec, ref);
  return sec;
}

function buildRecord(ev) {
  const frag = TEMPLATE.cloneNode(true);
  const art = frag.querySelector(".record");
  art._id = ev.id;
  art.dataset.id = ev.id;
  art.dataset.kind = ev.kind;

  frag.querySelector(".kind-badge").textContent = ev.kind;
  frag.querySelector(".time").textContent = fmtTime(ev.ts);
  frag.querySelector(".rec-del").dataset.id = ev.id;

  const src = ev.source ? ` · <span class="src">${ev.source}</span>` : "";
  frag.querySelector(".summary").innerHTML = (ev.summary || "") + src;

  const thumb = frag.querySelector(".thumb");
  const link = frag.querySelector(".thumb-link");
  const video = frag.querySelector(".clip");

  if (ev.kind === "clip" && ev.meta && ev.meta.clip_path) {
    video.src = `/clips/${ev.meta.clip_path}`;
    video.style.display = "block";
    link.remove();
  } else if (ev.snapshot_path) {
    thumb.src = `/snapshots/${ev.snapshot_path}`;
    link.href = `/snapshots/${ev.snapshot_path}`;
    link.addEventListener("click", (e) => {
      e.preventDefault();
      openLightbox(link.href);
    });
    video.remove();
  } else {
    link.remove();
    video.remove();
  }
  return art;
}

function appended(listEl, events) {
  let last = STATE.lastId;
  for (const ev of events) {
    if (ev.id <= STATE.lastId) continue;
    if (ev.id > last) last = ev.id;
    STATE.records += 1;

    const body = sectionFor(listEl, dayKey(ev.ts)).querySelector(".day-list");
    let ref = null;
    for (const existing of body.children) {
      if (existing._id < ev.id) { ref = existing; break; }
    }
    const art = buildRecord(ev);
    if (ref) body.insertBefore(art, ref);
    else body.appendChild(art);
  }
  if (last > STATE.lastId) STATE.lastId = last;
  trim(listEl);
  updateSync(`${fmtTime(Date.now())} · ${STATE.records} 条`);
}

function trim(listEl) {
  const groups = [...listEl.querySelectorAll(".day-group")];
  let count = 0;
  for (const g of groups) count += g.querySelectorAll(".record").length;
  if (count <= 300) return;
  let toRemove = count - 300;
  for (let i = groups.length - 1; i >= 0 && toRemove > 0; i--) {
    const recs = [...groups[i].querySelectorAll(".record")];
    while (toRemove > 0 && recs.length) {
      recs.pop().remove();
      STATE.records -= 1;
      toRemove -= 1;
    }
    if (!groups[i].querySelector(".record")) groups[i].remove();
  }
}

// ---------- 删除记录 ----------
async function deleteRecord(id) {
  if (!confirm("确定删除? 同一事件的关联记录(告警/快照/录像)及其媒体文件将一并删除。")) return;
  try {
    const res = await fetch(`/api/events/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    refresh();
    fetchDays();
  } catch (err) {
    console.error("delete failed:", err);
    updateSync("删除失败");
  }
}

function removeRecord(id) {
  const list = $("#record-list");
  const art = list.querySelector(`.record[data-id="${id}"]`);
  if (!art) return;
  const body = art.parentElement;
  art.remove();
  if (body && !body.querySelector(".record")) body.remove();
  STATE.records -= 1;
  updateSync(`${fmtTime(Date.now())} · ${STATE.records} 条`);
}

// ---------- 按天筛选 ----------
async function fetchDays() {
  try {
    const res = await fetch("/api/events/days");
    if (!res.ok) return;
    const data = await res.json();
    const bar = $("#day-bar");
    bar.innerHTML = "";

    const all = document.createElement("button");
    all.type = "button";
    all.className = "day-chip" + (STATE.day ? "" : " active");
    all.textContent = "全部";
    all.addEventListener("click", () => selectDay(""));
    bar.appendChild(all);

    for (const d of data.days) {
      const [, m, dd] = d.day.split("-").map(Number);
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "day-chip" + (STATE.day === d.day ? " active" : "");
      chip.textContent = `${m}月${dd}日 (${d.count})`;
      chip.addEventListener("click", () => selectDay(d.day));
      bar.appendChild(chip);
    }
  } catch (err) {
    console.error("fetch days failed:", err);
  }
}

function selectDay(day) {
  if (STATE.day === day) return;
  STATE.day = day;
  refresh();
  fetchDays();
}

// 清空并按当前筛选条件重新拉取
function refresh() {
  STATE.lastId = 0;
  STATE.records = 0;
  $("#record-list").innerHTML = "";
  return sync();
}

// ---------- 数据拉取 ----------
async function fetchEvents(afterId) {
  try {
    let url = `/api/events?after_id=${afterId}&limit=300`;
    if (STATE.day) url += `&day=${STATE.day}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const list = $("#record-list");
    const emptyEl = list.querySelector(".empty");
    if (emptyEl) emptyEl.remove();

    appended(list, data.events);

    // 全量刷新且确无记录时, 显示占位提示
    if (afterId === 0 && list.children.length === 0) {
      const el = document.createElement("div");
      el.className = "empty";
      el.textContent = STATE.day ? "该日期暂无记录" : "暂无记录";
      list.appendChild(el);
    }
    return data.last_id;
  } catch (err) {
    console.error("fetch events failed:", err);
    updateSync("拉取失败");
    return afterId;
  }
}

// 初次载入 + 收到信号 + 重连补拉, 都走同一入口
async function sync() {
  await fetchEvents(STATE.lastId);
}

// ---------- 状态栏 ----------
async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const s = await res.json();

    const cam = $("#cam-status");
    cam.dataset.status = s.camera.status;
    cam.textContent = `摄像头: ${s.camera.status}`;
    $("#cam-fps").textContent = `FPS: ${s.camera.fps}`;
    updateSync(`WS:${s.ws_clients}`);

    renderPlugins(s.plugins);
  } catch (err) {
    console.error("fetch status failed:", err);
  }
}

function renderPlugins(plugins) {
  const bar = $("#plugin-bar");
  bar.innerHTML = "";
  for (const p of plugins) {
    const pill = document.createElement("span");
    pill.className = "plug-pill" + (p.enabled ? " on" : "");
    pill.innerHTML = `<span class="dot"></span>${p.name}`;
    pill.title = p.last_error || "";
    pill.addEventListener("click", async () => {
      const res = await fetch(`/api/plugins/${p.name}/toggle`, { method: "POST" });
      if (res.ok) fetchStatus();
    });
    bar.appendChild(pill);
  }
}

// ---------- WebSocket ----------
function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  STATE.ws = ws;

  ws.onopen = () => {
    $("#ws-status").textContent = "WS: 已连接";
    $("#ws-status").dataset.status = "ok";
    fetchStatus();
    sync(); // 重连成功: 补拉增量
  };

  ws.onmessage = (msg) => {
    let data;
    try { data = JSON.parse(msg.data); } catch { return; }
    if (data.type === "record_created") {
      sync();
      fetchDays();
    } else if (data.type === "record_deleted") {
      removeRecord(data.record_id);
    } else if (data.type === "ping") {
      ws.send("pong");
    }
  };

  ws.onclose = () => {
    $("#ws-status").textContent = "WS: 断开";
    $("#ws-status").dataset.status = "error";
    setTimeout(connectWs, STATE.reconnectDelay);
  };
  ws.onerror = () => ws.close();
}

// ---------- 灯箱 ----------
function openLightbox(src) {
  $("#lb-img").src = src;
  $("#lightbox").classList.remove("hidden");
}

// ---------- 实时画面暂停/继续 ----------
function toggleLive() {
  STATE.livePaused = !STATE.livePaused;
  const btn = $("#live-toggle");
  const mask = $("#live-paused");
  if (STATE.livePaused) {
    $("#live").removeAttribute("src"); // 断开 MJPEG 连接
    btn.textContent = "继续";
    mask.classList.remove("hidden");
  } else {
    $("#live").src = "/cam/stream";
    btn.textContent = "暂停";
    mask.classList.add("hidden");
  }
}

function initLightbox() {
  $("#lightbox").addEventListener("click", (e) => {
    if (e.target.id === "lightbox" || e.target.classList.contains("lb-close")) {
      $("#lightbox").classList.add("hidden");
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") $("#lightbox").classList.add("hidden");
  });
}

function updateSync(text) {
  const el = $("#last-sync");
  el.textContent = text;
  el.dataset.ts = Date.now();
}

// ---------- 启动 ----------
async function boot() {
  initLightbox();
  $("#live-toggle").addEventListener("click", toggleLive);
  document.addEventListener("keydown", (e) => {
    if (e.key === "p" || e.key === "P") toggleLive();
  });
  $("#record-list").addEventListener("click", (e) => {
    const btn = e.target.closest(".rec-del");
    if (btn) deleteRecord(Number(btn.dataset.id));
  });
  await sync();          // 初次进页: 加载最近记录
  fetchStatus();
  fetchDays();           // 按天筛选条
  connectWs();           // WS 信号 + 重连补拉
  setInterval(fetchStatus, 15000); // 状态栏低频兜底刷新
}

boot();