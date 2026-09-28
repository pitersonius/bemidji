const $ = (id) => document.getElementById(id);
const recBtn = $("recBtn"), timerEl = $("timer"), statusEl = $("status"), canvas = $("meter");
const player = $("player"), transcriptEl = $("transcript"), summaryBox = $("summaryBox");

let mediaRecorder = null, chunks = [], stream = null, audioCtx = null, analyser = null;
let startedAt = 0, tick = null, current = null;
// Files waiting for / going through transcription + summary. Processed one at a time.
const queue = [];
let queueRunning = false, followQueue = true, nextQid = 1;

const fmt = (ms) => {
  const s = Math.floor(ms / 1000), h = Math.floor(s / 3600);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0"), ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
};
const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function setStatus(text, { spin = false, error = false } = {}) {
  statusEl.className = "status" + (error ? " error" : "");
  statusEl.innerHTML = (spin ? '<span class="spinner"></span>' : "") + esc(text);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
  return body;
}

// ------------------------------------------------------------------ recording

function pickMime() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
}

async function startRecording() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (e) {
    setStatus("Microphone access was denied: " + e.message, { error: true });
    return;
  }
  const mimeType = pickMime();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  chunks = [];
  mediaRecorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRecorder.onstop = onRecordingStopped;
  mediaRecorder.start(1000);

  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  drawMeter();

  startedAt = Date.now();
  tick = setInterval(() => (timerEl.textContent = fmt(Date.now() - startedAt)), 250);
  recBtn.classList.add("recording");
  recBtn.querySelector(".label").textContent = "Stop";
  recBtn.setAttribute("aria-label", "Stop recording");
  setStatus("Recording…");
}

function stopRecording() {
  mediaRecorder.stop();
  stream.getTracks().forEach((t) => t.stop());
  clearInterval(tick);
  audioCtx.close();
  analyser = null;
  recBtn.classList.remove("recording");
  recBtn.querySelector(".label").textContent = "Record";
  recBtn.setAttribute("aria-label", "Start recording");
}

function onRecordingStopped() {
  const type = mediaRecorder.mimeType || "audio/webm";
  const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
  const blob = new Blob(chunks, { type });
  if (blob.size < 1000) { setStatus("Recording was empty.", { error: true }); return; }
  const title = $("titleInput").value.trim();
  $("titleInput").value = "";
  timerEl.textContent = "00:00";
  setStatus("");
  enqueue([new File([blob], `recording.${ext}`, { type })], { source: "mic", title });
}

function drawMeter() {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  const color = getComputedStyle(document.documentElement).getPropertyValue("--accent");
  const data = new Uint8Array(2048);
  const frame = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!analyser) return;
    analyser.getByteTimeDomainData(data);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / data.length) * canvas.width;
      const y = (data[i] / 255) * canvas.height;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    requestAnimationFrame(frame);
  };
  frame();
}

recBtn.addEventListener("click", () => {
  mediaRecorder && mediaRecorder.state === "recording" ? stopRecording() : startRecording();
});

// ------------------------------------------------------------------ adding recordings

const dropzone = $("dropzone"), overlay = $("dropOverlay");
const MEDIA_EXT = /\.(m4a|mp3|wav|ogg|oga|opus|webm|flac|aac|amr|caf|wma|aiff?|mp4|mov|mkv|3gp)$/i;
const isMedia = (f) => /^(audio|video)\//.test(f.type) || MEDIA_EXT.test(f.name);

function addFiles(fileList) {
  const files = [...fileList];
  for (const f of files.filter((f) => !isMedia(f))) {
    queue.push({ qid: nextQid++, name: f.name, state: "error", error: "Not an audio or video file, skipped", rec: null });
  }
  const media = files.filter(isMedia);
  media.length ? enqueue(media, { source: "file" }) : renderQueue();
}

$("fileInput").addEventListener("change", (e) => {
  addFiles(e.target.files);
  e.target.value = "";
});
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileInput").click(); }
});

// Accept drops anywhere on the page, not only on the drop zone.
let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  dragDepth++;
  overlay.hidden = false;
  dropzone.classList.add("over");
});
window.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; overlay.hidden = true; dropzone.classList.remove("over"); }
});
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  overlay.hidden = true;
  dropzone.classList.remove("over");
  addFiles(e.dataTransfer.files);
});

// ------------------------------------------------------------------ queue

const fmtSize = (b) => b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`;
const STATE_LABEL = {
  waiting: "Waiting", transcribing: "Transcribing with Whisper…",
  summarizing: "Finding key moments…", done: "Done", error: "Failed",
};

function enqueue(files, { source, title = "" }) {
  if (!queueRunning) followQueue = true;
  for (const file of files) {
    queue.push({
      qid: nextQid++, file, size: file.size, source, title, state: "waiting", error: "", note: "",
      name: source === "mic" ? (title || "New recording") : file.name, rec: null,
    });
  }
  renderQueue();
  runQueue();
}

async function runQueue() {
  if (queueRunning) return;
  queueRunning = true;
  let item;
  while ((item = queue.find((q) => q.state === "waiting"))) await processItem(item);
  queueRunning = false;
  const done = queue.filter((q) => q.state === "done").length, failed = queue.filter((q) => q.state === "error").length;
  setStatus(failed ? `${done} processed, ${failed} failed.` : `All done. Transcripts are saved in the transcripts/ folder.`, { error: !!failed });
}

async function processItem(item) {
  const fd = new FormData();
  fd.append("audio", item.file);
  fd.append("source", item.source);
  if (item.title) fd.append("title", item.title);
  try {
    setItemState(item, "transcribing");
    const rec = await api("/api/transcribe", { method: "POST", body: fd });
    item.rec = rec;
    item.name = rec.title;
    if (followQueue) show(rec);
    loadHistory();
    setItemState(item, "summarizing");
    const summarized = await summarize(rec.id);
    if (summarized && summarized.llm_off) item.note = "Transcribed · key moments skipped (LLM is off)";
    else if (!summarized || summarized.summary_error) {
      item.note = "Transcribed, but key moments failed (open it to retry)";
      item.noteErr = true;
    }
    setItemState(item, "done");
  } catch (e) {
    item.error = e.message;
    setItemState(item, "error");
  }
  item.file = null; // release the audio data
}

function setItemState(item, state) {
  item.state = state;
  const pending = queue.filter((q) => q.state === "waiting").length;
  if (state === "transcribing" || state === "summarizing") {
    setStatus(`${item.name}: ${STATE_LABEL[state]}${pending ? ` (${pending} more waiting)` : ""}`, { spin: true });
  }
  renderQueue();
}

function renderQueue() {
  $("queueBox").hidden = !queue.length;
  $("queue").innerHTML = queue.map((q) => {
    const busyNow = q.state === "transcribing" || q.state === "summarizing";
    const icon = busyNow ? '<span class="spinner"></span>' : { waiting: "•", done: "✓", error: "!" }[q.state];
    const sub = q.state === "error" ? `<span class="q-sub err">${esc(q.error)}</span>`
      : q.note ? `<span class="q-sub${q.noteErr ? " err" : ""}">${esc(q.note)}</span>`
      : `<span class="q-sub">${q.size ? fmtSize(q.size) + " · " : ""}${STATE_LABEL[q.state]}</span>`;
    const action = q.rec ? `<button class="btn small" data-view="${q.rec.id}">View</button>`
      : q.state === "waiting" ? `<button class="btn small ghost" data-remove="${q.qid}" aria-label="Remove">✕</button>` : "";
    return `<li><span class="q-icon ${q.state}">${icon}</span><div><div class="q-name">${esc(q.name)}</div>${sub}</div>${action}</li>`;
  }).join("");
}

$("queue").addEventListener("click", async (e) => {
  const view = e.target.closest("[data-view]"), rm = e.target.closest("[data-remove]");
  if (view) { followQueue = false; show(await api(`/api/recordings/${view.dataset.view}`)); }
  if (rm) {
    const i = queue.findIndex((q) => q.qid === +rm.dataset.remove && q.state === "waiting");
    if (i >= 0) queue.splice(i, 1);
    renderQueue();
  }
});

$("clearQueue").addEventListener("click", () => {
  for (let i = queue.length - 1; i >= 0; i--) if (["done", "error"].includes(queue[i].state)) queue.splice(i, 1);
  renderQueue();
});

async function summarize(id) {
  if (current && current.id === id) {
    summaryBox.innerHTML = '<div class="status"><span class="spinner"></span>Asking the local LLM for key moments…</div>';
  }
  try {
    const rec = await api(`/api/recordings/${id}/summarize`, { method: "POST" });
    if (current && current.id === id) show(rec);
    return rec;
  } catch (e) {
    if (current && current.id === id) renderSummary({ id, bullets: [], summary_error: e.message });
  }
}

// ------------------------------------------------------------------ rendering

function show(rec) {
  current = rec;
  $("result").hidden = false;
  $("resTitle").textContent = rec.title;
  const when = new Date(rec.created).toLocaleString();
  const model = rec.whisper_model ? ` · ${rec.whisper_model.replace(/^ggml-|\.bin$/g, "")} model` : "";
  const language = (modelState && modelState.whisper.languages[rec.language]) || rec.language;
  $("resMeta").textContent = `${when} · ${fmt(rec.duration * 1000)} · ${language}${model}`;
  $("dlTranscript").href = `/files/${rec.id}/transcript?dl=1`;
  player.src = `/files/${rec.id}/audio`;

  transcriptEl.innerHTML = rec.segments.length
    ? rec.segments.map((s, i) =>
        `<div class="seg" data-i="${i}"><button class="ts" data-t="${s.start}">${fmt(s.start)}</button><span>${esc(s.text)}</span></div>`
      ).join("")
    : '<p class="muted">No speech detected.</p>';
  renderSummary(rec);
  document.querySelectorAll(".history li button").forEach((b) => b.classList.toggle("active", b.dataset.id === rec.id));
}

function renderSummary(rec) {
  $("dlSummary").hidden = !(rec.bullets && rec.bullets.length);
  $("dlSummary").href = `/files/${rec.id}/summary?dl=1`;
  const hasBullets = !!(rec.bullets && rec.bullets.length);
  $("regenBtn").hidden = !(hasBullets || rec.summary_error);
  $("summaryBy").textContent = hasBullets && rec.summary_model ? `Generated by ${rec.summary_model}` : "";
  if (rec.llm_off) {
    const canStart = modelState && modelState.llm.provider === "lmstudio";
    summaryBox.innerHTML = '<div class="llm-off"><p>The local LLM is off, so key moments weren\'t generated.</p><div class="actions">' +
      (canStart ? '<button class="btn small" data-llm-on>Turn on LLM &amp; summarize</button>' : "") +
      '<button class="btn small ghost" data-open-models>Models…</button></div></div>';
  } else if (rec.bullets === null || rec.bullets === undefined) {
    summaryBox.innerHTML = '<p class="muted">Not summarized yet.</p><button class="btn small" data-retry>Summarize</button>';
  } else if (rec.summary_error) {
    summaryBox.innerHTML = `<p class="status error">${esc(rec.summary_error)}</p><button class="btn small" data-retry>Retry</button>`;
  } else if (!rec.bullets.length) {
    summaryBox.innerHTML = '<p class="muted">Nothing to summarize.</p>';
  } else {
    summaryBox.innerHTML = "<ul>" + rec.bullets.map((b) => {
      const m = b.match(/^\[?(\d{1,2}:)?(\d{1,2}):(\d{1,2})\]?\s*[-–—:]?\s*(.*)$/);
      if (!m) return `<li>${esc(b)}</li>`;
      const ms = ((parseInt(m[1] || "0") * 3600) + parseInt(m[2]) * 60 + parseInt(m[3])) * 1000;
      return `<li><button class="ts" data-t="${ms}">${fmt(ms)}</button>${esc(m[4])}</li>`;
    }).join("") + "</ul>";
  }
}

document.addEventListener("click", (e) => {
  const ts = e.target.closest(".ts");
  if (ts) { player.currentTime = ts.dataset.t / 1000; player.play(); return; }
  if (e.target.closest("[data-retry]") && current) summarize(current.id);
  if (e.target.closest("[data-open-models]")) openModels();
  if (e.target.closest("[data-llm-on]") && current) {
    const id = current.id;
    summaryBox.innerHTML = '<div class="status"><span class="spinner"></span>Loading the LLM…</div>';
    turnOnLLMAndWait().then(() => summarize(id)).catch((err) => {
      if (current && current.id === id) renderSummary({ ...current, llm_off: false, bullets: [], summary_error: err.message });
    });
  }
});

player.addEventListener("timeupdate", () => {
  if (!current) return;
  const t = player.currentTime * 1000;
  transcriptEl.querySelectorAll(".seg").forEach((el) => {
    const s = current.segments[el.dataset.i];
    el.classList.toggle("playing", t >= s.start && t < s.end);
  });
});

$("regenBtn").addEventListener("click", () => current && summarize(current.id));

$("copyBtn").addEventListener("click", async () => {
  if (!current) return;
  await navigator.clipboard.writeText(current.segments.map((s) => s.text).join(" "));
  $("copyBtn").textContent = "Copied";
  setTimeout(() => ($("copyBtn").textContent = "Copy"), 1200);
});

$("delBtn").addEventListener("click", async () => {
  if (!current || !confirm(`Delete "${current.title}"?`)) return;
  await api(`/api/recordings/${current.id}`, { method: "DELETE" });
  current = null;
  $("result").hidden = true;
  player.removeAttribute("src");
  loadHistory();
});

// ------------------------------------------------------------------ history

async function loadHistory() {
  const items = await api("/api/recordings");
  const ul = $("historyList");
  if (!items.length) { ul.innerHTML = '<li class="muted">No recordings yet</li>'; return; }
  ul.innerHTML = items.map((r) =>
    `<li><button data-id="${r.id}" class="${current && current.id === r.id ? "active" : ""}">` +
    `<span class="h-title">${esc(r.title)}</span>` +
    `<span class="h-sub">${new Date(r.created).toLocaleString([], { dateStyle: "short", timeStyle: "short" })} · ${fmt(r.duration * 1000)}</span>` +
    `</button></li>`
  ).join("");
}

$("historyList").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-id]");
  if (!b) return;
  followQueue = false;
  show(await api(`/api/recordings/${b.dataset.id}`));
});

loadHistory();
