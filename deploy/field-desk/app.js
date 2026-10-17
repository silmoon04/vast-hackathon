"use strict";
const $ = id => document.getElementById(id);
const state = { clip: null, analysis: null, mode: "people", region: [], editing: false, draft: [], busy: false, loadVersion: 0, catalogVersion: 0 };
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
let noticeTimer;

function notice(text) {
  clearTimeout(noticeTimer); $("notice").textContent = text; $("notice").hidden = false;
  noticeTimer = setTimeout(() => { $("notice").hidden = true; }, 9000);
}
function clock(t) { t = Math.max(0, Number(t) || 0); return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`; }
function node(tag, className, text) { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; }
async function api(path, body) {
  const response = await fetch(new URL(path, document.baseURI), body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || `Request failed (${response.status}).`);
  return result;
}
function enable(ready) {
  for (const id of ["draw-region", "lower-half", "reset-region", "download", "question", "ask-button"]) $(id).disabled = !ready;
  document.querySelectorAll("[data-question]").forEach(button => { button.disabled = !ready; });
}
function message(text, who = "assistant", result = null) {
  const box = node("div", `message ${who}`);
  if (who === "assistant") box.append(node("div", "message-title", "Video Atlas"));
  box.append(node("p", "", text));
  if (result?.evidence?.length) {
    const row = node("div", "evidence");
    for (const item of result.evidence) {
      const button = node("button", "", `${clock(item.time)} · ${item.count} ${item.label === "person" ? "people" : item.label === "vehicle" ? "vehicles" : item.label}`);
      button.type = "button"; button.addEventListener("click", () => seek(item.time)); row.append(button);
    }
    box.append(row);
    box.append(node("p", "message-meta", "Computed from the selected region. Click a time to inspect the video."));
  }
  $("messages").append(box); $("messages").scrollTop = $("messages").scrollHeight;
  return box;
}
function cameraName(name) { return name.replaceAll("_", " ").replace(/-\d+$/, "").replace(/\bcam\b/i, "camera"); }

async function loadCatalog() {
  const version = ++state.catalogVersion;
  $("list-caption").textContent = "Filtering the indexed catalogue…";
  const params = new URLSearchParams({ mode: $("mode").value, q: $("query").value.trim(), camera: $("camera").value });
  try {
    const data = await api(`api/catalog?${params}`);
    if (version !== state.catalogVersion) return;
    state.mode = $("mode").value;
    $("connection").textContent = `${data.total} indexed clips connected`; $("connection").classList.add("ready");
    const selectedCamera = $("camera").value;
    $("camera").replaceChildren(new Option("All cameras", ""), ...data.cameras.map(camera => new Option(cameraName(camera), camera)));
    $("camera").value = selectedCamera;
    $("list-caption").textContent = `${data.matches} matching clips${data.matches > data.clips.length ? `; showing ${data.clips.length}` : ""}`;
    $("clips").replaceChildren();
    for (const clip of data.clips) {
      const button = node("button", "clip-card" + (state.clip?.id === clip.id ? " selected" : ""));
      button.type = "button"; button.dataset.clip = clip.id;
      button.append(node("span", "clip-camera", cameraName(clip.camera)), node("span", "clip-file", clip.filename), node("span", "clip-caption", clip.caption || "No catalogue caption."));
      button.addEventListener("click", () => selectClip(clip)); $("clips").append(button);
    }
    if (!data.clips.length) $("clips").append(node("p", "empty", "No matching clips. Try another term or camera."));
    if (!state.clip && data.clips.length) await selectClip(data.clips[0]);
    else if (state.analysis) drawChart();
  } catch (error) { if (version === state.catalogVersion) { $("list-caption").textContent = "Catalogue unavailable"; notice(error.message); } }
}

async function selectClip(clip) {
  state.clip = clip; state.region = []; state.editing = false; state.draft = [];
  $("draw-region").classList.remove("active"); $("draw-region").textContent = "Draw a region"; $("stage").classList.remove("editing");
  $("region-tag").textContent = "Whole image";
  document.querySelectorAll(".clip-card").forEach(button => button.classList.toggle("selected", button.dataset.clip === clip.id));
  $("camera-label").textContent = clip.camera;
  $("clip-title").textContent = cameraName(clip.camera);
  $("clip-duration").textContent = clip.duration ? `${Number(clip.duration).toFixed(0)}s clip` : "";
  $("source-note").textContent = clip.filename;
  video.pause(); video.src = new URL(`video?clip=${clip.id}`, document.baseURI).href; video.load();
  $("messages").replaceChildren();
  message("I’m checking this clip’s detections. Once ready, you can ask about counts, time ranges or sustained activity.");
  await analyse(true);
}

async function analyse(initial = false) {
  if (!state.clip) return;
  const version = ++state.loadVersion;
  state.busy = true; state.analysis = null; enable(false);
  $("answer-status").textContent = "Analysing detections…";
  $("stage-message").textContent = "Reading detections and computing occupancy…"; $("stage-message").hidden = false;
  $("chart").replaceChildren(node("p", "", "Computing the occupancy timeline…"));
  try {
    const result = await api("api/analyse", { clip: state.clip.id, region: state.region, confidence: Number($("confidence").value) });
    if (version !== state.loadVersion) return;
    state.analysis = result; state.busy = false; enable(true);
    $("stage-message").hidden = true; $("answer-status").textContent = "Ready for a question";
    $("clip-duration").textContent = `${result.duration.toFixed(0)}s clip`;
    const facts = [[`${result.frame_count}`, "frames analysed"], [`${result.coverage_pct}%`, "time coverage"], [`${result.segments_loaded}/${result.segments_total}`, "segments loaded"]];
    $("analysis-facts").replaceChildren(...facts.map(([value, label]) => { const span = node("span"); span.append(node("strong", "", value), document.createTextNode(` ${label}`)); return span; }));
    $("region-tag").textContent = state.region.length ? "Selected region" : "Whole image";
    drawChart(); drawOverlay();
    $("region-help").textContent = "Counts use the bottom-centre of each box inside your selected region.";
    if (!initial) message("The selected region has been re-analysed. New questions use this region and confidence setting.");
    else {
      message(`Ready. I analysed ${result.frame_count} frames from ${result.segments_loaded} segments of this catalogue clip. These are simultaneous detections, not unique identities.`);
      await ask("When was it busiest?", false);
    }
  } catch (error) {
    if (version !== state.loadVersion) return;
    state.busy = false; $("stage-message").textContent = error.message; $("answer-status").textContent = "Choose another clip or try again"; notice(error.message);
  }
}

async function ask(question, showUser = true) {
  if (!state.analysis || state.busy) return;
  const aid = state.analysis.analysis_id;
  if (showUser) message(question, "user");
  $("ask-button").disabled = true; $("answer-status").textContent = "Checking the analysis…";
  try {
    const result = await api("api/ask", { analysis_id: aid, question, mode: state.mode });
    if (state.analysis?.analysis_id !== aid) return;
    message(result.answer, "assistant", result);
    $("answer-status").textContent = "Answer linked to detections";
    if (!showUser && result.evidence?.length) seek(result.evidence[0].time);
  } catch (error) { notice(error.message); }
  finally { $("ask-button").disabled = !state.analysis || state.busy; }
}

function seek(time) {
  if (video.readyState >= 1) video.currentTime = Math.min(time, Math.max(0, video.duration - .05));
  else video.addEventListener("loadedmetadata", () => { video.currentTime = time; }, { once: true });
  drawOverlay();
}
function nearestFrame(time) {
  const frames = state.analysis?.frames || [];
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (frames[mid].t < time) lo = mid + 1; else hi = mid; }
  const right = frames[lo], left = frames[Math.max(0, lo - 1)];
  const frame = !right || (left && Math.abs(left.t - time) < Math.abs(right.t - time)) ? left : right;
  return frame && Math.abs(frame.t - time) < .15 ? frame : null;
}
function drawOverlay() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const frame = nearestFrame(video.currentTime);
  if ($("show-boxes").checked && frame) {
    ctx.font = `${Math.max(14, w / 95)}px Arial`; ctx.lineWidth = Math.max(2, w / 650);
    for (const box of frame.boxes) {
      const [x1, y1, x2, y2] = box.box;
      const color = !box.in_region ? "#9aa9b9" : box.label === "person" ? "#74b7ff" : "#f3c465";
      ctx.strokeStyle = color; ctx.globalAlpha = box.in_region ? 1 : .35;
      ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
      const text = `${box.label} ${Math.round(box.confidence * 100)}%`, tw = ctx.measureText(text).width;
      const ty = Math.max(18, y1 * h);
      ctx.fillStyle = "#132438dd"; ctx.fillRect(x1 * w, ty - 19, tw + 9, 21);
      ctx.fillStyle = color; ctx.fillText(text, x1 * w + 4, ty - 3);
    }
    ctx.globalAlpha = 1;
  }
  const region = state.editing ? state.draft : state.region;
  if (region.length) {
    ctx.beginPath(); region.forEach(([x, y], i) => { i ? ctx.lineTo(x * w, y * h) : ctx.moveTo(x * w, y * h); });
    if (region.length >= 3) ctx.closePath();
    ctx.fillStyle = "#72a9ec22"; if (region.length >= 3) ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(2, w / 550); ctx.setLineDash(state.editing ? [10, 7] : []); ctx.stroke(); ctx.setLineDash([]);
    for (const [x, y] of region) { ctx.beginPath(); ctx.arc(x * w, y * h, Math.max(4, w / 220), 0, Math.PI * 2); ctx.fillStyle = "#2366c0"; ctx.fill(); ctx.strokeStyle = "white"; ctx.stroke(); }
  }
  const playhead = $("playhead");
  if (playhead && state.analysis) { const x = 38 + video.currentTime / state.analysis.duration * 942; playhead.setAttribute("x1", x); playhead.setAttribute("x2", x); }
}

function svgNode(tag, attrs = {}, text) { const el = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [k, value] of Object.entries(attrs)) el.setAttribute(k, value); if (text !== undefined) el.textContent = text; return el; }
function drawChart() {
  const data = state.analysis; if (!data) return;
  const label = state.mode === "vehicles" ? "vehicle" : "person";
  $("activity-title").textContent = label === "person" ? "People visible over time" : "Vehicles visible over time";
  const top = Math.max(1, ...data.series.filter(r => r.counts !== null).map(r => r.counts[label] || 0));
  const svg = svgNode("svg", { viewBox: "0 0 1000 155", role: "img", "aria-label": `${label} counts by second, maximum ${top}` });
  const x = t => 38 + t / data.duration * 942, y = n => 119 - n / top * 103;
  for (const n of [0, Math.ceil(top / 2), top]) { svg.append(svgNode("line", { x1: 38, x2: 980, y1: y(n), y2: y(n), stroke: "#e2e9f2" }), svgNode("text", { x: 26, y: y(n) + 4, "text-anchor": "end", fill: "#607084", "font-size": 12 }, n)); }
  let points = [];
  const flush = () => { if (points.length) svg.append(svgNode("polyline", { points: points.join(" "), fill: "none", stroke: "#2366c0", "stroke-width": 3, "stroke-linejoin": "round" })); points = []; };
  for (const row of data.series) {
    if (row.counts === null) { flush(); svg.append(svgNode("rect", { x: x(row.t), y: 15, width: x(row.end) - x(row.t), height: 104, fill: "#e3e7ed" })); }
    else { points.push(`${x(row.t)},${y(row.counts[label] || 0)}`, `${x(row.end)},${y(row.counts[label] || 0)}`); }
  }
  flush();
  for (let i = 0; i <= 5; i++) { const t = data.duration * i / 5; svg.append(svgNode("text", { x: x(t), y: 146, "text-anchor": i === 5 ? "end" : i === 0 ? "start" : "middle", fill: "#607084", "font-size": 12 }, clock(t))); }
  svg.append(svgNode("line", { id: "playhead", x1: x(video.currentTime), x2: x(video.currentTime), y1: 8, y2: 123, stroke: "#a74339", "stroke-width": 2 }));
  svg.addEventListener("click", event => { const rect = svg.getBoundingClientRect(); seek(Math.max(0, Math.min(data.duration, ((event.clientX - rect.left) / rect.width * 1000 - 38) / 942 * data.duration))); });
  $("chart").replaceChildren(svg);
}

$("filter-form").addEventListener("submit", event => { event.preventDefault(); loadCatalog(); });
$("mode").addEventListener("change", () => loadCatalog());
$("camera").addEventListener("change", () => loadCatalog());
$("ask-form").addEventListener("submit", event => { event.preventDefault(); const q = $("question").value.trim(); if (q) { $("question").value = ""; ask(q); } });
$("question").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("ask-form").requestSubmit(); } });
$("suggestions").addEventListener("click", event => { const button = event.target.closest("[data-question]"); if (button) ask(button.dataset.question); });
$("show-boxes").addEventListener("change", drawOverlay);
$("confidence").addEventListener("change", () => analyse());
$("lower-half").addEventListener("click", () => { state.region = [[0, .5], [1, .5], [1, 1], [0, 1]]; analyse(); });
$("reset-region").addEventListener("click", () => { state.region = []; analyse(); });
$("draw-region").addEventListener("click", () => {
  if (state.editing) {
    if (state.draft.length < 3) return notice("Click at least three points before finishing the region.");
    state.region = [...state.draft]; state.editing = false; $("stage").classList.remove("editing"); $("draw-region").classList.remove("active"); $("draw-region").textContent = "Draw a region"; analyse();
  } else {
    state.editing = true; state.draft = []; video.pause(); $("stage").classList.add("editing"); $("draw-region").classList.add("active"); $("draw-region").textContent = "Finish region";
    $("region-help").textContent = "Click 3 to 12 corners on the video, then Finish region. Escape cancels."; drawOverlay();
  }
});
canvas.addEventListener("click", event => { if (!state.editing || state.draft.length >= 12) return; const r = canvas.getBoundingClientRect(); state.draft.push([Math.max(0, Math.min(1, (event.clientX - r.left) / r.width)), Math.max(0, Math.min(1, (event.clientY - r.top) / r.height))]); drawOverlay(); });
document.addEventListener("keydown", event => { if (event.key === "Escape" && state.editing) { state.editing = false; state.draft = []; $("stage").classList.remove("editing"); $("draw-region").classList.remove("active"); $("draw-region").textContent = "Draw a region"; $("region-help").textContent = "Region drawing cancelled."; drawOverlay(); } });
video.addEventListener("loadedmetadata", () => { canvas.width = video.videoWidth || 1280; canvas.height = video.videoHeight || 720; $("stage").style.aspectRatio = `${canvas.width}/${canvas.height}`; drawOverlay(); });
video.addEventListener("timeupdate", drawOverlay); video.addEventListener("seeked", drawOverlay);
video.addEventListener("error", () => { if (state.clip) notice("The video could not load. Try opening the clip again; its detections may still be available."); });
$("download").addEventListener("click", () => { if (!state.analysis) return; const report = { ...state.analysis }; const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }); const a = document.createElement("a"); const url = URL.createObjectURL(blob); a.href = url; a.download = `atlas-${state.clip.id}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 5000); });
loadCatalog();
