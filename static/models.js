// Models panel: top-bar status, Whisper and LLM on/off, model selection and downloads.
// Uses the helpers defined in app.js ($, api, esc).

let modelState = null;
let modelPoll = null;
const panel = $("modelsPanel"), backdrop = $("panelBackdrop");

const fmtBytes = (b) => (!b ? "" : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
const fmtCtx = (n) => (n ? `${Math.round(n / 1024)}K context` : "");
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const whisperLabel = (s, name) => (s.whisper.models.find((m) => m.name === name) || {}).label || name;

// Replace innerHTML only when it changed, so polling doesn't steal focus or restart hover states.
const lastHTML = new Map();
function setHTML(el, html) {
  if (lastHTML.get(el) === html) return;
  lastHTML.set(el, html);
  el.innerHTML = html;
}

// ------------------------------------------------------------------ data

async function refreshModels() {
  try {
    modelState = await api("/api/models");
  } catch {
    modelState = null;
  }
  renderModels();
  scheduleModelPoll();
}

function modelsBusy(s) {
  return !!s && (s.whisper.state === "starting" || !!s.llm.busy || s.whisper.models.some((m) => m.download && !m.download.error));
}

function scheduleModelPoll() {
  clearTimeout(modelPoll);
  modelPoll = setTimeout(refreshModels, !panel.hidden || modelsBusy(modelState) ? 1000 : 8000);
}

async function modelAction(path, body, method = "POST") {
  $("modelsError").hidden = true;
  try {
    modelState = await api(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    $("modelsError").textContent = e.message;
    $("modelsError").hidden = false;
  }
  renderModels();
  scheduleModelPoll();
}

const postWhisper = (body) => modelAction("/api/models/whisper", body);
const postLLM = (body) => modelAction("/api/models/llm", body);

/** Turn the LLM on and resolve once the model is loaded (used by "Turn on & summarize"). */
async function turnOnLLMAndWait() {
  await postLLM({ on: true });
  for (let i = 0; i < 900; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    modelState = await api("/api/models");
    renderModels();
    if (!modelState.llm.busy) {
      if (modelState.llm.on) return;
      throw new Error(modelState.llm.error || "The LLM did not turn on.");
    }
  }
  throw new Error("Timed out waiting for the LLM to load.");
}

// ------------------------------------------------------------------ rendering

function renderModels() {
  const s = modelState;
  if (!s) {
    $("pillWhisper").className = "pill err";
    $("pillWhisper").textContent = "Server unreachable";
    $("pillLLM").hidden = true;
    return;
  }
  $("pillLLM").hidden = false;
  renderPills(s);
  if (!panel.hidden) {
    renderWhisper(s);
    renderLLM(s);
  }
}

function renderPills(s) {
  const w = s.whisper, l = s.llm;
  const wl = whisperLabel(s, w.selected);
  const pw = $("pillWhisper");
  pw.className = "pill " + { on: "on", starting: "busy", error: "err", off: "idle" }[w.state];
  pw.textContent = `Whisper · ${wl}` + (w.state === "starting" ? " · loading…" : w.state === "off" ? " · on demand" : "");
  pw.title = w.state === "on" ? "Whisper model is loaded in memory" : w.state === "off"
    ? "Whisper loads the model only while transcribing" : w.error || "";

  const pl = $("pillLLM");
  pl.className = "pill " + (l.busy ? "busy" : l.error ? "err" : l.on ? "on" : "");
  pl.textContent = l.busy ? `LLM · ${l.busy}…` : l.on ? `LLM · ${l.active_model}` : "LLM · off";
  pl.title = l.error || (l.on ? "Key moments are generated after each transcription" : "Key moments are skipped while the LLM is off");
}

function stateBox(el, { busy, on, err, html }) {
  el.className = "mstate" + (err ? " err" : "");
  setHTML(el, (busy ? '<span class="spinner"></span>' : err ? "" : `<span class="dot${on ? " on" : ""}"></span>`) + `<div>${html}</div>`);
}

function renderWhisper(s) {
  const w = s.whisper;
  const label = whisperLabel(s, w.selected);
  const toggle = $("whisperToggle");
  toggle.checked = w.state === "on" || w.state === "starting";
  toggle.disabled = w.state === "starting" || !w.server_available;

  if (w.state === "on") {
    stateBox($("whisperState"), { on: true, html: `<strong>Loaded in memory</strong> · ${esc(label)}. Transcription starts right away.` });
  } else if (w.state === "starting") {
    stateBox($("whisperState"), { busy: true, html: `Loading <strong>${esc(label)}</strong> into memory…` });
  } else if (w.state === "error") {
    stateBox($("whisperState"), { err: true, html: esc(w.error || "Whisper failed") });
  } else {
    stateBox($("whisperState"), {
      html: `<strong>Off</strong> · nothing kept in memory. ${esc(label)} is loaded only while a file is transcribed, then freed. Turn on for faster transcription.`,
    });
  }

  renderLanguageOptions(w);
  renderLanguageHint(w);
  setHTML($("whisperModels"), w.models.filter((m) => !m.pack).map((m) => modelRow(m, w)).join(""));
  renderPacks(w);
}

function modelRow(m, w) {
  const inMem = w.loaded_model === m.name;
  const dl = m.download;
  let action = "", extra = "";
  if (m.downloaded) {
    if (!m.selected) action = `<button class="btn small ghost" data-delete="${m.name}" title="Delete ${esc(m.label)}">Delete</button>`;
  } else if (dl && !dl.error) {
    const compressing = dl.phase === "compressing";
    const pct = compressing ? 100 : dl.total ? Math.floor((dl.done / dl.total) * 100) : 0;
    extra = `<div class="progress${compressing ? " busy" : ""}"><div style="width:${pct}%"></div></div>`;
    if (!compressing) action = `<button class="btn small ghost" data-cancel="${m.name}">Cancel</button>`;
  } else {
    action = `<button class="btn small" data-download="${m.name}">${dl && dl.error ? "Retry" : "Download"}</button>`;
  }
  const size = !m.size ? ""
    : m.downloaded ? ` · ${fmtBytes(m.size)}`
    : m.compressed_size ? ` · ~${fmtBytes(m.size)} download, ~${fmtBytes(m.compressed_size)} after compression`
    : ` · ~${fmtBytes(m.size)}`;
  const meta = dl && !dl.error
    ? (dl.phase === "compressing" ? "Compressing to save space…"
      : `Downloading… ${fmtBytes(dl.done)} of ${fmtBytes(dl.total || m.size)}`)
    : dl && dl.error ? `Download failed: ${esc(dl.error)}`
    : `${esc(m.description)}${size}`;
  return `<li class="mrow${m.downloaded ? " pickable" : " unavailable"}${m.selected ? " selected" : ""}"` +
    (m.downloaded ? ` role="radio" tabindex="0" aria-checked="${m.selected}" data-pick="${m.name}"` : "") + `>` +
    `<span class="radio"></span>` +
    `<div><div class="m-name">${esc(m.label)}${inMem ? '<span class="badge on">In memory</span>' : ""}</div>` +
    `<div class="m-meta">${meta}</div>${extra}</div>` +
    `<div class="m-actions">${action}</div></li>`;
}

// ---- language picker (100 languages, with a search box)

let langOptionsKey = "";
function renderLanguageOptions(w) {
  const q = $("langFilter").value.trim().toLowerCase();
  const key = q + "|" + w.language;
  if (key === langOptionsKey) return;
  langOptionsKey = key;
  const packLangs = new Set(Object.values(w.packs).flatMap((p) => p.languages));
  const entries = Object.entries(w.languages).sort((a, b) => a[1].localeCompare(b[1]));
  const match = ([code, name]) => !q || name.toLowerCase().includes(q) || code === q;
  const opts = [["auto", "Detect automatically"], ...entries].filter((e) => e[0] === "auto" ? !q : match(e));
  const sel = $("langSelect");
  sel.innerHTML = opts.map(([code, name]) =>
    `<option value="${code}">${esc(name)}${packLangs.has(code) ? " · language pack available" : ""}</option>`).join("");
  sel.size = q ? Math.min(8, Math.max(2, opts.length)) : 1;
  sel.value = w.language || "auto";
  if (sel.value !== (w.language || "auto")) sel.selectedIndex = -1;
}

$("langFilter").addEventListener("input", () => modelState && renderLanguageOptions(modelState.whisper));
$("langFilter").addEventListener("keydown", (e) => {
  const sel = $("langSelect");
  if (e.key === "Enter" && sel.options.length) {
    e.preventDefault();
    chooseLanguage(sel.options[0].value);
  } else if (e.key === "Escape" && e.target.value) {
    e.stopPropagation();
    e.target.value = "";
    modelState && renderLanguageOptions(modelState.whisper);
  } else if (e.key === "ArrowDown" && sel.options.length) {
    e.preventDefault();
    sel.focus();
  }
});
function chooseLanguage(code) {
  $("langFilter").value = "";
  langOptionsKey = "";
  postWhisper({ language: code });
}

// ---- guidance for the chosen language + model

function languageHint(w) {
  const lang = w.language || "auto";
  const langName = w.languages[lang] || lang;
  const sel = w.models.find((m) => m.selected) || {};
  const byFile = (f) => w.models.find((m) => m.name === f);
  const packFor = Object.entries(w.packs).find(([, p]) => p.languages.includes(lang));
  const selPack = sel.pack && w.packs[sel.pack];
  const recommended = byFile("ggml-large-v3-turbo-q5_0.bin");
  const useOrGet = (m) => !m ? null : m.selected ? null : m.downloaded
    ? { label: `Use ${m.label}`, model: m.name }
    : { label: `Download ${m.label} (${fmtBytes(m.size)})`, download: m.name };

  if (sel.english_only && lang !== "en" && lang !== "auto") {
    return { kind: "err", html: `<strong>${esc(sel.label)}</strong> only understands English, so ${esc(langName)} recordings will fail. Pick a general model${packFor ? ` or the ${esc(packFor[1].name)} pack` : ""}.`,
      action: useOrGet(recommended) };
  }
  if (selPack && sel.pack !== "en" && lang !== "auto" && !selPack.languages.includes(lang)) {
    return { kind: "warn", html: `<strong>${esc(sel.label)}</strong> is tuned for ${esc(selPack.name)}. For ${esc(langName)}, a general model will be more accurate.`,
      action: useOrGet(recommended) };
  }
  if (lang === "auto") {
    if (selPack && sel.pack !== "en") {
      return { kind: "tip", html: `<strong>${esc(sel.label)}</strong> is tuned for ${esc(selPack.name)}. Set the spoken language to ${esc(selPack.name)} for the best results.`,
        action: { label: `Set to ${selPack.name}`, lang: selPack.languages[0] } };
    }
    if (sel.english_only) return { kind: "tip", html: `<strong>${esc(sel.label)}</strong> is English-only, so every recording is transcribed as English.` };
    return { kind: "", html: "Whisper detects the language of each recording. Choosing the language yourself is more reliable for short or mixed-language clips." };
  }
  if (packFor && packFor[1].recommended === false) {
    // A pack exists, but our own test found the general model more accurate: recommend that.
    const [code, p] = packFor;
    if (recommended && recommended.selected) {
      return { kind: "ok", html: `Using <strong>${esc(recommended.label)}</strong>, the most accurate choice for ${esc(langName)} in our tests.` };
    }
    const using = sel.pack === code ? `You're using the ${esc(p.name)} pack. ` : "";
    return { kind: using ? "warn" : "tip",
      html: `${using}The general <strong>Large v3 Turbo (compressed)</strong> is the most accurate choice for ${esc(langName)}. ${esc(p.tested)}`,
      action: useOrGet(recommended) };
  }
  if (packFor) {
    const [code, p] = packFor;
    if (sel.pack === code) return { kind: "ok", html: `Using the <strong>${esc(p.name)} language pack</strong> (${esc(sel.label)}).` };
    const have = w.models.find((m) => m.pack === code && m.downloaded);
    return { kind: "tip", html: `A <strong>${esc(p.name)} language pack</strong> is available from ${esc(p.by)}. ${esc(p.about)}`,
      action: have ? { label: `Use ${have.label}`, model: have.name } : { label: `Show ${p.name} pack`, pack: code } };
  }
  const large = /large/.test(sel.name || "");
  return { kind: large ? "ok" : "tip",
    html: `No ${esc(langName)}-specific pack exists for whisper.cpp yet, but the general models understand ${esc(langName)}.` +
      (large ? ` <strong>${esc(sel.label)}</strong> is a good choice.` : " Larger models are much more accurate for it; Large v3 Turbo (compressed) is the recommended one."),
    action: large ? null : useOrGet(recommended) };
}

function renderLanguageHint(w) {
  const h = languageHint(w);
  const el = $("langHint");
  el.hidden = false;
  el.className = "lang-hint" + (h.kind ? " " + h.kind : "");
  const a = h.action;
  const btn = !a ? "" : `<button class="btn small" ${a.model ? `data-pick-hint="${a.model}"` : a.download ? `data-download="${a.download}"`
    : a.lang ? `data-lang="${a.lang}"` : `data-show-pack="${a.pack}"`}>${esc(a.label)}</button>`;
  setHTML(el, `<div>${h.html}</div>${btn}`);
}

// ---- language packs, one collapsible group per language

const openPacks = new Set();
let packsInitialised = false;
function renderPacks(w) {
  if (!packsInitialised) {
    // Start with the packs that matter right now expanded.
    for (const [code, p] of Object.entries(w.packs)) {
      if (p.languages.includes(w.language) || w.models.some((m) => m.pack === code && (m.selected || m.downloaded))) openPacks.add(code);
    }
    packsInitialised = true;
  }
  setHTML($("packList"), Object.entries(w.packs).map(([code, p]) => {
    const models = w.models.filter((m) => m.pack === code);
    const installed = models.filter((m) => m.downloaded).length;
    const using = models.some((m) => m.selected);
    const badge = using ? '<span class="badge on">In use</span>' : installed ? `<span class="badge">${installed} installed</span>` : `<span class="badge">${models.length} size${models.length === 1 ? "" : "s"}</span>`;
    return `<details class="pack" data-pack="${code}"${openPacks.has(code) ? " open" : ""}>` +
      `<summary>${esc(p.name)} ${badge}</summary><div class="pack-body">` +
      `<div class="pack-about">${esc(p.about)} By ${esc(p.by)} · <a href="${esc(p.link)}" target="_blank" rel="noopener">Details</a></div>` +
      (p.tested ? `<div class="pack-tested">${esc(p.tested)}</div>` : "") +
      `<ul class="mlist">${models.map((m) => modelRow(m, w)).join("")}</ul></div></details>`;
  }).join(""));
}
$("packList").addEventListener("toggle", (e) => {
  const d = e.target.closest("details[data-pack]");
  if (d) d.open ? openPacks.add(d.dataset.pack) : openPacks.delete(d.dataset.pack);
}, true);

function renderLLM(s) {
  const l = s.llm;
  const lmstudio = l.provider === "lmstudio";
  document.querySelectorAll(".seg-ctl button").forEach((b) => b.setAttribute("aria-checked", b.dataset.provider === l.provider));
  $("lmstudioBox").hidden = !lmstudio;
  $("customBox").hidden = lmstudio;
  $("llmSwitch").hidden = !lmstudio;
  if (document.activeElement !== $("summaryLang")) $("summaryLang").value = l.summary_language || "same";

  if (lmstudio) {
    const lm = l.lmstudio;
    const turningOn = l.busy && !/stopping|unloading models/.test(l.busy);
    const toggle = $("llmToggle");
    toggle.checked = l.busy ? !!turningOn : l.on;
    toggle.disabled = !!l.busy || !lm.installed;
    const sel = lm.models.find((m) => m.key === l.selected);

    if (!lm.installed) {
      stateBox($("llmState"), { err: true, html: "LM Studio isn't installed. Install it from lmstudio.ai, or use “Other server”." });
    } else if (l.busy) {
      stateBox($("llmState"), { busy: true, html: `${esc(cap(l.busy))}…` });
    } else if (l.error || lm.error) {
      stateBox($("llmState"), { err: true, html: esc(l.error || lm.error) });
    } else if (l.on) {
      stateBox($("llmState"), {
        on: true,
        html: `<strong>On</strong> · ${esc(sel.name)} is loaded (${fmtCtx(sel.context)}). Key moments are generated after each transcription.`,
      });
    } else {
      stateBox($("llmState"), {
        html: `<strong>Off</strong> · no model in memory. Transcripts still work; key moments are skipped until you turn this on.`,
      });
    }
    if (document.activeElement !== $("ctxSelect")) $("ctxSelect").value = String(l.context_length);

    setHTML($("llmModels"), lm.models.length ? lm.models.map((m) => {
      const selected = m.key === l.selected;
      const meta = [m.params, m.quant, fmtBytes(m.size)].filter(Boolean).join(" · ");
      return `<li class="mrow pickable${selected ? " selected" : ""}" role="radio" tabindex="0" aria-checked="${selected}" data-llm="${esc(m.key)}">` +
        `<span class="radio"></span>` +
        `<div><div class="m-name">${esc(m.name)}${m.loaded ? `<span class="badge on">Loaded · ${fmtCtx(m.context)}</span>` : ""}</div>` +
        `<div class="m-meta">${esc(meta)}</div></div><div></div></li>`;
    }).join("") : '<li class="muted">No LLMs downloaded in LM Studio yet.</li>');
  } else {
    const c = l.custom;
    if (c.reachable) {
      stateBox($("llmState"), { on: true, html: `<strong>Connected</strong> · ${esc(l.custom_base_url)}` });
    } else {
      stateBox($("llmState"), {
        err: true, html: `Can't reach ${esc(l.custom_base_url)}. Start your server, then press Connect.`,
      });
    }
    if (document.activeElement !== $("baseUrl")) $("baseUrl").value = l.custom_base_url;
    setHTML($("customModels"), c.models.length ? c.models.map((id) => {
      const selected = id === l.active_model;
      return `<li class="mrow pickable${selected ? " selected" : ""}" role="radio" tabindex="0" aria-checked="${selected}" data-llm="${esc(id)}">` +
        `<span class="radio"></span><div><div class="m-name">${esc(id)}</div></div><div></div></li>`;
    }).join("") : `<li class="muted">${c.reachable ? "The server lists no chat models." : "No models, since the server is unreachable."}</li>`);
  }
}

// ------------------------------------------------------------------ events

function openModels() {
  panel.hidden = backdrop.hidden = false;
  lastHTML.clear();
  renderModels();
  refreshModels();
  $("closeModels").focus();
}
function closeModels() {
  panel.hidden = backdrop.hidden = true;
  $("openModels").focus();
  scheduleModelPoll();
}
$("openModels").addEventListener("click", openModels);
$("closeModels").addEventListener("click", closeModels);
backdrop.addEventListener("click", closeModels);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) closeModels(); });

$("whisperToggle").addEventListener("change", (e) => postWhisper({ on: e.target.checked }));
$("langSelect").addEventListener("change", (e) => chooseLanguage(e.target.value));
$("langSelect").addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.value) chooseLanguage(e.target.value); });
$("summaryLang").addEventListener("change", (e) => postLLM({ summary_language: e.target.value }));
$("llmToggle").addEventListener("change", (e) => postLLM({ on: e.target.checked }));
$("ctxSelect").addEventListener("change", (e) => postLLM({ context_length: +e.target.value }));
document.querySelector(".seg-ctl").addEventListener("click", (e) => {
  const b = e.target.closest("[data-provider]");
  if (b && b.getAttribute("aria-checked") !== "true") postLLM({ provider: b.dataset.provider });
});
$("saveUrl").addEventListener("click", () => postLLM({ base_url: $("baseUrl").value }));
$("baseUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") postLLM({ base_url: e.target.value }); });

function onWhisperClick(e) {
  const t = e.target.closest("[data-download],[data-cancel],[data-delete],[data-pick],[data-pick-hint],[data-lang],[data-show-pack]");
  if (!t) return;
  const d = t.dataset;
  if (d.download) modelAction(`/api/models/whisper/${d.download}/download`);
  else if (d.cancel) modelAction(`/api/models/whisper/${d.cancel}/download`, null, "DELETE");
  else if (d.delete) { if (confirm(`Delete ${d.delete} from disk?`)) modelAction(`/api/models/whisper/${d.delete}`, null, "DELETE"); }
  else if (d.pickHint) postWhisper({ model: d.pickHint });
  else if (d.lang) chooseLanguage(d.lang);
  else if (d.showPack) {
    openPacks.add(d.showPack);
    lastHTML.delete($("packList"));
    renderPacks(modelState.whisper);
    document.querySelector(`details[data-pack="${d.showPack}"]`).scrollIntoView({ behavior: "smooth", block: "nearest" });
  } else if (d.pick && t.getAttribute("aria-checked") !== "true") postWhisper({ model: d.pick });
}
for (const id of ["whisperModels", "packList", "langHint"]) $(id).addEventListener("click", onWhisperClick);

for (const list of [$("llmModels"), $("customModels")]) {
  list.addEventListener("click", (e) => {
    const row = e.target.closest("[data-llm]");
    if (row && row.getAttribute("aria-checked") !== "true" && !(modelState && modelState.llm.busy)) postLLM({ model: row.dataset.llm });
  });
}

// Keyboard selection for the radio-style model rows.
panel.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && e.target.matches(".mrow[role=radio]")) {
    e.preventDefault();
    e.target.click();
  }
});

refreshModels();
