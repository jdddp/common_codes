"use strict";

// 回放页: 持续录像分片按摄像头分组列表, 每条可在线播放 + 下载
const P = { day: "", camera: "all" };

const $ = (sel) => document.querySelector(sel);

function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// 时间以后端落库的 meta.file_start/file_end(毫秒, 墙钟)为准, 不解析文件名;
// 旧记录(早期版本)无该字段时回退文件名解析
function parseLapse(ev) {
  if (!ev.meta || !ev.meta.lapse_path) return null;
  const start = Number(ev.meta.file_start);
  const end = Number(ev.meta.file_end);
  if (start > 0 && end > start) {
    return { ev, cam: ev.camera_id, start, end, dur: end - start };
  }
  const m = ev.meta.lapse_path.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.mp4$/);
  if (!m) return null;
  const s = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]).getTime();
  const hours = Number(ev.meta.file_hours) || 4;
  return { ev, cam: ev.camera_id, start: s, end: s + hours * 3600 * 1000, dur: hours * 3600 * 1000 };
}

function fmtHM(ms) {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function fmtDate(ms) {
  const d = new Date(ms);
  return `${d.getMonth() + 1}月${d.getDate()}日 ${fmtHM(ms)}`;
}

function fmtDur(ms) {
  const h = Math.floor(ms / 3600000), m = Math.round((ms % 3600000) / 60000);
  return h ? `${h}时${m ? m + "分" : ""}` : `${m}分`;
}

function segCoverage(first, last) {
  const len = last.end - first.start;
  if (len <= 0) return "";
  const h = Math.floor(len / 3600000), m = Math.floor((len % 3600000) / 60000);
  if (h) return `${h}小时${m ? m + "分" : ""}`;
  return `${m}分`;
}

async function fetchLapseSeqs() {
  let url = "/api/events?kind=lapse&limit=300";
  if (P.day) url += `&day=${P.day}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const seqs = [];
  for (const ev of data.events) {
    const s = parseLapse(ev);
    if (s) seqs.push(s);
  }
  seqs.sort((a, b) => (a.cam === b.cam ? a.start - b.start : a.cam < b.cam ? -1 : 1));
  return seqs;
}

function renderPlayCams(seqs) {
  const ids = [...new Set(seqs.map((s) => s.cam))].sort();
  const bar = $("#play-cams");
  bar.innerHTML = "";
  const add = (label, id) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "cam-chip" + (P.camera === id ? " active" : "");
    b.textContent = label;
    b.addEventListener("click", () => {
      P.camera = id;
      renderPlayCams(seqs);
      renderList(seqs);
    });
    bar.appendChild(b);
  };
  add(`全部 (${ids.length}路)`, "all");
  for (const id of ids) add(id, id);
}

function buildItem(s) {
  const art = document.createElement("article");
  art.className = "record";
  art.dataset.kind = "lapse";

  const media = document.createElement("div");
  media.className = "media";
  const vid = document.createElement("video");
  vid.className = "clip";
  vid.controls = true;
  vid.preload = "metadata";
  vid.src = `/lapse/${s.cam}/${s.ev.meta.lapse_path}`;
  vid.title = `${s.cam} · ${fmtDate(s.start)} ~ ${fmtHM(s.end)}`;
  const dl = document.createElement("a");
  dl.className = "rec-dl";
  dl.href = vid.src;
  dl.download = s.ev.meta.lapse_path;
  dl.textContent = "下载视频";
  media.append(vid, dl);

  const meta = document.createElement("div");
  meta.className = "meta";
  const row1 = document.createElement("div");
  row1.className = "row1";
  const badge = document.createElement("span");
  badge.className = "kind-badge";
  badge.textContent = s.cam;
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = `${fmtDate(s.start)} ~ ${fmtHM(s.end)} · ${fmtDur(s.dur)}`;
  row1.append(badge, time);
  const summary = document.createElement("div");
  summary.className = "summary";
  summary.textContent = "持续录像";
  meta.append(row1, summary);

  art.append(media, meta);
  return art;
}

function renderList(seqs) {
  const body = $("#play-list");
  body.innerHTML = "";
  const visible = P.camera === "all" ? seqs : seqs.filter((s) => s.cam === P.camera);
  if (!visible.length) {
    body.innerHTML = '<div class="empty">该日期暂无持续录像</div>';
    return;
  }
  const byCam = {};
  for (const s of visible) (byCam[s.cam] = byCam[s.cam] || []).push(s);
  for (const cam of Object.keys(byCam).sort()) {
    const segs = byCam[cam].sort((a, b) => a.start - b.start);
    const head = document.createElement("div");
    head.className = "list-cam-head";
    head.textContent = `${cam} · ${segs.length} 段 · 覆盖 ${segCoverage(segs[0], segs[segs.length - 1])}`;
    body.appendChild(head);
    for (const s of segs) body.appendChild(buildItem(s));
  }
}

function renderPlayback() {
  if (!P.day) P.day = todayStr();
  $("#play-date").value = P.day;
  $("#play-info").textContent = "加载中...";
  fetchLapseSeqs()
    .then((seqs) => {
      renderPlayCams(seqs);
      renderList(seqs);
      $("#play-info").textContent = `${P.day} · 共 ${seqs.length} 段`;
    })
    .catch((err) => {
      console.error("playback load failed:", err);
      $("#play-info").textContent = "回放数据拉取失败";
    });
}

function initPlayback() {
  $("#play-date").value = todayStr();
  $("#play-refresh").addEventListener("click", renderPlayback);
  $("#play-date").addEventListener("change", (e) => {
    P.day = e.target.value;
    renderPlayback();
  });
}

initPlayback();
renderPlayback();