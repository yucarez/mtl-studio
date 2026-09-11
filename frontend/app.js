"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const ACCEPT = /\.(png|jpe?g|webp)$/i;
const TRANSLATOR_NAMES = {
  anthropic: "Claude (Anthropic)", openai: "OpenAI-compatible", deepl: "DeepL", google: "Google Translate",
  passthrough: "No translation (layout test)",
};

const S = {
  config: null,
  settings: {},
  pending: [],             // {key, file, url}
  items: new Map(),        // id -> item
  order: [],               // display order: item ids and pending keys
  batches: new Set(),
  cards: new Map(),        // key -> <li>
  version: new Map(),      // id -> cache-buster
  viewer: { id: null, mode: "compare", edits: {}, state: null },
  polling: false,
};

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* not json */ }
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return r;
}
const json = (path, opts) => api(path, opts).then(r => r.json());
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, isError ? 7000 : 3500);
}
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
function downloadBlob(blob, name) {
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
const langName = code => (S.config.languages.find(l => l.code === code) || {}).name || code;

// ------------------------------------------------------------------ settings
const SETTINGS_KEY = "mtl.settings.v1";
function loadSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}"); } catch (_) { /* private mode */ }
  S.settings = { ...S.config.defaults, ...saved };
  const configured = S.config.translators.filter(t => t.configured).map(t => t.id);
  if (S.settings.translator !== "auto" && !configured.includes(S.settings.translator)) S.settings.translator = "auto";
}
function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(S.settings)); } catch (_) { /* ignore */ }
  updateEngineNote();
}
function fillSettingsForm() {
  const f = $("#settingsForm");
  const tr = f.elements.translator;
  tr.innerHTML = "";
  tr.append(el("option", { value: "auto" }, "Automatic (first configured)"));
  for (const t of S.config.translators) {
    tr.append(el("option", { value: t.id, disabled: !t.configured },
      TRANSLATOR_NAMES[t.id] + (t.configured ? "" : " (add API key in .env)")));
  }
  const font = f.elements.font;
  font.innerHTML = "";
  font.append(el("option", { value: "auto" }, "Automatic (best for the language)"));
  for (const n of S.config.fonts) font.append(el("option", { value: n }, n));
  for (const [k, v] of Object.entries(S.settings)) {
    const input = f.elements[k];
    if (!input) continue;
    if (input.type === "checkbox") input.checked = !!v;
    else input.value = v;
  }
  $("#tiltOut").textContent = `${S.settings.follow_tilt_max}°`;
}
function readSettingsForm() {
  const f = $("#settingsForm");
  for (const input of f.elements) {
    if (!input.name) continue;
    if (input.type === "checkbox") S.settings[input.name] = input.checked;
    else if (input.type === "number" || input.type === "range") S.settings[input.name] = Number(input.value);
    else S.settings[input.name] = input.value;
  }
  $("#tiltOut").textContent = `${S.settings.follow_tilt_max}°`;
  saveSettings();
}
function currentOptions() {
  return { ...S.settings, source_lang: $("#src").value, target_lang: $("#tgt").value };
}
function updateEngineNote() {
  const t = S.settings.translator === "auto"
    ? (S.config.translators.find(x => x.configured) || {}).id
    : S.settings.translator;
  const model = S.config.models[t] ? ` using ${S.config.models[t]}` : "";
  $("#engineNote").textContent = !t ? "" : t === "passthrough"
    ? "Layout test: text is re-lettered untranslated"
    : `Translating with ${TRANSLATOR_NAMES[t]}${model}`;
}

// ------------------------------------------------------------------ files & queue
function addFiles(list) {
  const files = [...list].filter(f => ACCEPT.test(f.name) || /^image\/(png|jpeg|webp)$/.test(f.type));
  const rejected = list.length - files.length;
  if (rejected) toast(`${rejected} file(s) skipped: only PNG, JPG and WEBP are supported.`, true);
  const max = S.config.max_upload_mb * 1024 * 1024;
  for (const file of files) {
    if (file.size > max) { toast(`${file.name} is larger than ${S.config.max_upload_mb} MB.`, true); continue; }
    const key = "p" + Math.random().toString(36).slice(2, 10);
    S.pending.push({ key, file, url: URL.createObjectURL(file) });
    S.order.push(key);
  }
  render();
}

async function start() {
  if (!S.pending.length) return;
  const batch = S.pending.splice(0);
  const fd = new FormData();
  for (const p of batch) fd.append("files", p.file, p.file.name);
  fd.append("options", JSON.stringify(currentOptions()));
  $("#startBtn").disabled = true;
  try {
    const res = await json("/api/batches", { method: "POST", body: fd });
    S.batches.add(res.id);
    // replace pending cards (same position) with server items
    const items = [...res.items];
    for (const p of batch) {
      const idx = items.findIndex(i => i.filename === p.file.name.split(/[\\/]/).pop());
      const it = idx >= 0 ? items.splice(idx, 1)[0] : null;
      const pos = S.order.indexOf(p.key);
      if (it) {
        S.items.set(it.id, it);
        if (pos >= 0) S.order[pos] = it.id; else S.order.push(it.id);
        const card = S.cards.get(p.key);
        if (card) { S.cards.delete(p.key); card.remove(); }
      } else if (pos >= 0) {
        S.order.splice(pos, 1);
      }
      URL.revokeObjectURL(p.url);
    }
    poll();
  } catch (e) {
    S.pending.unshift(...batch);
    toast(e.message, true);
  }
  render();
}

async function poll() {
  if (S.polling) return;
  S.polling = true;
  try {
    while (true) {
      const active = [...S.batches].filter(bid =>
        [...S.items.values()].some(i => i.batch_id === bid && (i.status === "queued" || i.status === "processing")));
      if (!active.length) break;
      for (const bid of active) {
        try {
          const b = await json(`/api/batches/${bid}`);
          for (const it of b.items) {
            const prev = S.items.get(it.id);
            if (prev && prev.status !== "done" && it.status === "done") S.version.set(it.id, Date.now());
            S.items.set(it.id, it);
          }
        } catch (e) {
          if (e.status === 404) {   // batch no longer on the server: stop asking, tell the user
            S.batches.delete(bid);
            for (const it of S.items.values()) {
              if (it.batch_id === bid && (it.status === "queued" || it.status === "processing")) {
                Object.assign(it, { status: "failed", stage_label: "Failed",
                  error: "This page is no longer on the server (the server restarted or the batch was deleted). Add it again." });
              }
            }
          } /* anything else is transient: try again next round */
        }
      }
      render();
      await new Promise(r => setTimeout(r, 900));
    }
  } finally {
    S.polling = false;
    render();
  }
}

function statusText(it) {
  if (it.status === "queued") return "Waiting";
  if (it.status === "processing") return `${it.stage_label} ${Math.round(it.progress * 100)}%`;
  if (it.status === "failed") return "Failed";
  const parts = [`${it.regions} region${it.regions === 1 ? "" : "s"}`];
  if (it.seconds) parts.push(`${it.seconds} s`);
  return parts.join(", ");
}

function cardFor(key) {
  let li = S.cards.get(key);
  if (li) return li;
  li = el("li", { class: "page" },
    el("button", { class: "thumb", type: "button", "aria-label": "Open page" }, el("img", { alt: "", loading: "lazy" })),
    el("div", { class: "progress", "aria-hidden": "true" }, el("i")),
    el("p", { class: "page-name" }),
    el("p", { class: "page-status" }),
    el("p", { class: "page-error", hidden: true }),
    el("div", { class: "page-actions" }));
  S.cards.set(key, li);
  return li;
}

function renderCard(key) {
  const li = cardFor(key);
  const pend = S.pending.find(p => p.key === key);
  const img = $(".thumb img", li);
  const thumbBtn = $(".thumb", li);
  const actions = $(".page-actions", li);
  actions.innerHTML = "";
  if (pend) {
    li.className = "page pending";
    if (img.getAttribute("src") !== pend.url) img.src = pend.url;
    $(".page-name", li).textContent = pend.file.name;
    $(".page-status", li).textContent = "Ready to translate";
    $(".progress > i", li).style.width = "0";
    thumbBtn.disabled = true;
    actions.append(el("button", { class: "btn ghost", type: "button", onclick: () => removePending(key) }, "Remove"));
    return li;
  }
  const it = S.items.get(key);
  li.className = `page ${it.status}` + (it.status === "processing" ? " working" : "");
  const v = S.version.get(it.id) || 0;
  const src = it.has_result ? `/api/items/${it.id}/thumb?which=result&v=${v}` : `/api/items/${it.id}/thumb`;
  if (img.getAttribute("src") !== src) img.src = src;
  img.alt = it.filename;
  $(".page-name", li).textContent = it.filename;
  $(".page-name", li).title = it.filename;
  const st = $(".page-status", li);
  st.innerHTML = "";
  st.append(el("span", { class: "status-line" }, statusText(it)));
  if (it.errors) st.append(el("span", { class: "badge err", title: "Errors that need attention" }, `${it.errors} error${it.errors > 1 ? "s" : ""}`));
  if (it.warnings) st.append(el("span", { class: "badge warn", title: "Warnings" }, `${it.warnings} to check`));
  if (it.kept_original) st.append(el("span", { class: "badge kept", title: "Regions left in the original language" }, `${it.kept_original} kept original`));
  $(".progress > i", li).style.width = `${Math.round((it.status === "done" ? 1 : it.progress) * 100)}%`;
  const err = $(".page-error", li);
  err.hidden = it.status !== "failed";
  err.textContent = it.error || "";
  thumbBtn.disabled = !it.has_result;
  thumbBtn.onclick = () => it.has_result && openViewer(it.id);
  if (it.status === "done") {
    actions.append(el("button", { class: "btn", type: "button", onclick: () => openViewer(it.id) }, "Review"));
    actions.append(el("a", { class: "btn ghost", href: `/api/items/${it.id}/result?download=true`, download: it.download_name || "" }, "Download"));
  } else if (it.status === "failed") {
    actions.append(el("button", { class: "btn", type: "button", onclick: () => retryItem(it.id) }, "Retry"));
  }
  return li;
}

function removePending(key) {
  const i = S.pending.findIndex(p => p.key === key);
  if (i >= 0) { URL.revokeObjectURL(S.pending[i].url); S.pending.splice(i, 1); }
  S.order = S.order.filter(k => k !== key);
  const c = S.cards.get(key); if (c) { c.remove(); S.cards.delete(key); }
  render();
}

function render() {
  const grid = $("#grid");
  for (const key of S.order) {
    const li = renderCard(key);
    if (li.parentNode !== grid) grid.append(li);
  }
  // keep DOM order in sync
  S.order.forEach((key, i) => { const li = S.cards.get(key); if (grid.children[i] !== li) grid.insertBefore(li, grid.children[i] || null); });
  const items = [...S.items.values()];
  const done = items.filter(i => i.status === "done").length;
  const failed = items.filter(i => i.status === "failed").length;
  const active = items.filter(i => i.status === "queued" || i.status === "processing").length;
  const total = S.pending.length + items.length;
  const bits = [];
  if (!total) bits.push("No pages yet");
  else {
    bits.push(`${total} page${total === 1 ? "" : "s"}`);
    if (active) bits.push(`${active} in progress`);
    if (done) bits.push(`${done} done`);
    if (failed) bits.push(`${failed} failed`);
  }
  $("#queueCount").textContent = bits.join(", ");
  $("#empty").hidden = total > 0;
  const sb = $("#startBtn");
  sb.disabled = !S.pending.length;
  sb.textContent = S.pending.length ? `Translate ${S.pending.length} page${S.pending.length === 1 ? "" : "s"}` : "Translate";
  $("#retryFailedBtn").hidden = !failed;
  $("#zipBtn").disabled = !done;
  $("#zipBtn").textContent = done ? `Download ZIP (${done})` : "Download ZIP";
  $("#clearBtn").hidden = !total || active > 0;
}

async function retryItem(id) {
  try {
    const it = await json(`/api/items/${id}/retry`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(currentOptions()),
    });
    S.items.set(id, it);
    S.batches.add(it.batch_id);
    render(); poll();
  } catch (e) { toast(e.message, true); }
}

async function retryFailed() {
  const bids = new Set([...S.items.values()].filter(i => i.status === "failed").map(i => i.batch_id));
  for (const bid of bids) {
    try {
      await json(`/api/batches/${bid}/retry-failed`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(currentOptions()),
      });
      for (const it of S.items.values()) if (it.batch_id === bid && it.status === "failed") it.status = "queued";
    } catch (e) { toast(e.message, true); }
  }
  render(); poll();
}

async function downloadZip() {
  const ids = S.order.filter(k => S.items.get(k)?.status === "done");
  try {
    const r = await api("/api/zip", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ item_ids: ids }),
    });
    downloadBlob(await r.blob(), `translated-${$("#tgt").value}-${ids.length}-pages.zip`);
  } catch (e) { toast(e.message, true); }
}

async function clearList() {
  if (!confirm("Remove all pages from the list and delete their files from the server?")) return;
  for (const bid of S.batches) { try { await api(`/api/batches/${bid}`, { method: "DELETE" }); } catch (_) { /* gone */ } }
  for (const p of S.pending) URL.revokeObjectURL(p.url);
  S.pending = []; S.items.clear(); S.order = []; S.batches.clear();
  for (const c of S.cards.values()) c.remove();
  S.cards.clear();
  render();
}

// ------------------------------------------------------------------ viewer
async function openViewer(id) {
  const v = $("#viewer");
  S.viewer = { id, mode: S.viewer.mode || "compare", edits: {}, state: null };
  v.hidden = false;
  document.body.style.overflow = "hidden";
  await loadViewer();
  $("#closeViewer").focus();
}

async function loadViewer() {
  const id = S.viewer.id;
  let data;
  try { data = await json(`/api/items/${id}`); } catch (e) { toast(e.message, true); return; }
  S.viewer.state = data.state;
  S.items.set(id, { ...S.items.get(id), ...data });
  $("#viewerTitle").textContent = data.filename;
  $("#dlOne").href = `/api/items/${id}/result?download=true`;
  $("#dlOne").setAttribute("download", data.download_name || "");
  setMode(S.viewer.mode);
  renderRegions();
}

function setMode(mode) {
  S.viewer.mode = mode;
  const id = S.viewer.id;
  const v = S.version.get(id) || 0;
  $$(".modes button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
  const cmp = $("#compare");
  const a = $("#imgA"), b = $("#imgB");
  const srcs = {
    compare: [`/api/items/${id}/work`, `/api/items/${id}/result?v=${v}`],
    original: [`/api/items/${id}/work`], result: [`/api/items/${id}/result?v=${v}`],
    regions: [`/api/items/${id}/overlay?v=${v}`],
  }[mode];
  a.src = srcs[0];
  if (srcs[1]) b.src = srcs[1];
  cmp.classList.toggle("single", srcs.length === 1);
}

function setSplit(pct) {
  pct = Math.max(0, Math.min(100, pct));
  $("#clip").style.clipPath = `inset(0 0 0 ${pct}%)`;
  const h = $("#handle");
  h.style.left = pct + "%";
  h.setAttribute("aria-valuenow", Math.round(pct));
}

function renderRegions() {
  const st = S.viewer.state;
  const list = $("#regionList");
  list.innerHTML = "";
  $("#pageIssues").innerHTML = "";
  $("#rerenderBtn").disabled = true;
  if (!st) { $("#pageMeta").textContent = ""; return; }
  const m = st.meta || {};
  const bits = [];
  if (m.source_lang) bits.push(`${langName(m.source_lang)} to ${langName(st.options.target_lang)}`);
  if (m.content_type) bits.push(m.content_type);
  if (m.reading_direction) bits.push(m.reading_direction === "rtl" ? "read right to left" : "read left to right");
  if (m.ocr) bits.push(`OCR ${m.ocr}`);
  if (m.translator) bits.push(TRANSLATOR_NAMES[m.translator] || m.translator);
  $("#pageMeta").textContent = bits.join(", ");
  for (const i of st.issues || []) $("#pageIssues").append(issueEl(i));
  const regs = [...st.regions].sort((a, b) => a.order - b.order);
  if (!regs.length) list.append(el("li", { class: "empty" }, "No text was found on this page."));
  for (const r of regs) {
    const cls = r.status === "kept_original" ? "kept" : r.status === "skipped" ? "skipped" : "";
    const ta = el("textarea", { "aria-label": `Translation for region ${r.order + 1}`, rows: 2 });
    ta.value = S.viewer.edits[r.id] ?? r.translation;
    ta.addEventListener("input", () => {
      const changed = ta.value !== r.translation;
      ta.classList.toggle("edited", changed);
      if (changed) S.viewer.edits[r.id] = ta.value; else delete S.viewer.edits[r.id];
      $("#rerenderBtn").disabled = !Object.keys(S.viewer.edits).length;
    });
    const tag = [r.role || r.kind, r.vertical ? "vertical" : null,
      r.status === "kept_original" ? "original kept" : r.status === "skipped" ? r.skip_reason : null]
      .filter(Boolean).join(", ");
    const li = el("li", { class: `region ${cls}` },
      el("span", { class: "region-n", title: "Reading order" }, String(r.order + 1)),
      el("div", {},
        el("p", { class: "region-src" }, r.source_text || "(no text)", el("span", { class: "region-tag" }, tag)),
        ta,
        visibleIssues(r.issues || []).map(issueEl)));
    list.append(li);
  }
}

function visibleIssues(list) {
  // hide routine auto-fixes, and fixes that were superseded by an unresolved issue of the same kind
  const open = new Set(list.filter(i => !i.fixed).map(i => i.code));
  return list.filter(i => !i.fixed || (i.severity !== "info" && !open.has(i.code)));
}

function issueEl(i) {
  return el("p", { class: `issue ${i.severity}${i.fixed ? " fixed" : ""}` },
    (i.fixed ? "Fixed: " : "") + i.message);
}

async function rerender() {
  const btn = $("#rerenderBtn");
  btn.disabled = true;
  btn.textContent = "Lettering…";
  try {
    const data = await json(`/api/items/${S.viewer.id}/rerender`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ edits: S.viewer.edits }),
    });
    S.version.set(S.viewer.id, Date.now());
    S.items.set(S.viewer.id, { ...S.items.get(S.viewer.id), ...data });
    S.viewer.edits = {};
    S.viewer.state = data.state;
    setMode(S.viewer.mode);
    renderRegions();
    render();
    toast("Re-lettered.");
  } catch (e) { toast(e.message, true); }
  btn.textContent = "Re-letter edited regions";
}

function closeViewer() {
  if (Object.keys(S.viewer.edits).length && !confirm("Discard your unsaved edits?")) return;
  $("#viewer").hidden = true;
  document.body.style.overflow = "";
  S.viewer.id = null;
}

function stepViewer(dir) {
  const done = S.order.filter(k => S.items.get(k)?.has_result);
  const i = done.indexOf(S.viewer.id);
  const next = done[i + dir];
  if (!next) return;
  if (Object.keys(S.viewer.edits).length && !confirm("Discard your unsaved edits?")) return;
  S.viewer.id = next; S.viewer.edits = {};
  loadViewer();
}

// ------------------------------------------------------------------ init
async function restoreRecent() {
  try {
    const batches = await json("/api/batches");
    for (const b of batches.slice(0, 5).reverse()) {
      const full = await json(`/api/batches/${b.id}`);
      S.batches.add(b.id);
      for (const it of full.items) { S.items.set(it.id, it); S.order.push(it.id); }
    }
  } catch (_) { /* first run */ }
}

function bind() {
  const drop = $("#drop");
  const input = $("#fileInput");
  $("#pick").addEventListener("click", e => { e.stopPropagation(); input.click(); });
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  let depth = 0;
  window.addEventListener("dragenter", e => { if (e.dataTransfer?.types?.includes("Files")) { depth++; drop.classList.add("over"); } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) drop.classList.remove("over"); });
  window.addEventListener("dragover", e => e.preventDefault());
  window.addEventListener("drop", e => {
    e.preventDefault(); depth = 0; drop.classList.remove("over");
    if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
  });
  window.addEventListener("paste", e => {
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) addFiles(files.map((f, i) => f.name && f.name !== "image.png" ? f
      : new File([f], `pasted-${Date.now()}-${i}.png`, { type: f.type })));
  });

  $("#startBtn").addEventListener("click", start);
  $("#retryFailedBtn").addEventListener("click", retryFailed);
  $("#zipBtn").addEventListener("click", downloadZip);
  $("#clearBtn").addEventListener("click", clearList);

  $("#swap").addEventListener("click", () => {
    const s = $("#src"), t = $("#tgt");
    if (s.value === "auto") { toast("Choose a source language to swap."); return; }
    [s.value, t.value] = [t.value, s.value];
    S.settings.source_lang = s.value; S.settings.target_lang = t.value; saveSettings();
  });
  $("#src").addEventListener("change", e => { S.settings.source_lang = e.target.value; saveSettings(); });
  $("#tgt").addEventListener("change", e => { S.settings.target_lang = e.target.value; saveSettings(); });

  const drawer = $("#settings"), scrim = $("#scrim");
  const openD = () => { drawer.classList.add("open"); drawer.setAttribute("aria-hidden", "false"); scrim.hidden = false; $("#openSettings").setAttribute("aria-expanded", "true"); $("#closeSettings").focus(); };
  const closeD = () => { drawer.classList.remove("open"); drawer.setAttribute("aria-hidden", "true"); scrim.hidden = true; $("#openSettings").setAttribute("aria-expanded", "false"); };
  $("#openSettings").addEventListener("click", openD);
  $("#closeSettings").addEventListener("click", closeD);
  scrim.addEventListener("click", closeD);
  $("#settingsForm").addEventListener("input", readSettingsForm);
  $("#settingsForm").addEventListener("change", readSettingsForm);

  // viewer
  $("#closeViewer").addEventListener("click", closeViewer);
  $("#prevPage").addEventListener("click", () => stepViewer(-1));
  $("#nextPage").addEventListener("click", () => stepViewer(1));
  $$(".modes button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#rerenderBtn").addEventListener("click", rerender);
  $("#retryOneBtn").addEventListener("click", () => { const id = S.viewer.id; S.viewer.edits = {}; closeViewer(); retryItem(id); });
  const stage = $("#stage");
  stage.classList.add("fit");
  $("#zoomBtn").addEventListener("click", () => {
    const fit = stage.classList.toggle("fit");
    stage.classList.toggle("actual", !fit);
    $("#zoomBtn").textContent = fit ? "Actual size" : "Fit to screen";
  });
  const cmp = $("#compare");
  let dragging = false;
  const move = e => {
    if (!dragging) return;
    const r = cmp.getBoundingClientRect();
    setSplit(((e.clientX - r.left) / r.width) * 100);
  };
  cmp.addEventListener("pointerdown", e => { if (cmp.classList.contains("single")) return; dragging = true; cmp.setPointerCapture(e.pointerId); move(e); });
  cmp.addEventListener("pointermove", move);
  cmp.addEventListener("pointerup", () => { dragging = false; });
  $("#handle").addEventListener("keydown", e => {
    const now = Number($("#handle").getAttribute("aria-valuenow"));
    if (e.key === "ArrowLeft") { setSplit(now - 5); e.preventDefault(); }
    if (e.key === "ArrowRight") { setSplit(now + 5); e.preventDefault(); }
  });
  document.addEventListener("keydown", e => {
    if ($("#viewer").hidden) { if (e.key === "Escape") closeD(); return; }
    const typing = /TEXTAREA|INPUT|SELECT/.test(document.activeElement?.tagName || "");
    if (e.key === "Escape") closeViewer();
    else if (!typing && e.key === "ArrowLeft" && document.activeElement !== $("#handle")) stepViewer(-1);
    else if (!typing && e.key === "ArrowRight" && document.activeElement !== $("#handle")) stepViewer(1);
  });
}

async function init() {
  try {
    S.config = await json("/api/config");
  } catch (e) {
    $("#notices").append(el("p", { class: "notice" }, `Cannot reach the server: ${e.message}`));
    return;
  }
  loadSettings();
  const src = $("#src"), tgt = $("#tgt");
  src.append(el("option", { value: "auto" }, "Detect language"));
  for (const l of S.config.languages) {
    src.append(el("option", { value: l.code }, l.name));
    tgt.append(el("option", { value: l.code }, l.name));
  }
  src.value = S.settings.source_lang || "auto";
  tgt.value = S.settings.target_lang || "en";
  for (const w of S.config.warnings) $("#notices").append(el("p", { class: "notice" }, w));
  fillSettingsForm();
  updateEngineNote();
  bind();
  setSplit(50);
  await restoreRecent();
  render();
  poll();
}

init();
