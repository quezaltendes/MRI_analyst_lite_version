"use strict";
const $ = id => document.getElementById(id);
let mode = "knee", task = "mci", busy = false, serverStatus = null, jobId = null, result = null;
let descriptionText = "";
let chatBusy = false, chatHistory = [];

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function clearFiles() {
  ["file", "flair", "t1ce"].forEach(id => { $(id).value = ""; });
  $("flair-name").textContent = $("t1ce-name").textContent = ".nii or .nii.gz";
  $("preprocessed").checked = false;
  $("selected-file").hidden = true;
}
function ready() {
  const key = mode === "knee" ? "knee" : task;
  const installed = serverStatus?.models[key]?.installed;
  const filesReady = mode === "brain" && task === "idh"
    ? $("flair").files.length && $("t1ce").files.length : $("file").files.length;
  $("analyze-button").disabled = busy || !installed || !filesReady || (mode === "brain" && !$("preprocessed").checked);
  $("ready-label").replaceChildren();
  const dot = document.createElement("span"); dot.className = "tiny-dot";
  $("ready-label").append(dot, document.createTextNode(busy ? "Analysis in progress" : !installed ? "Analysis unavailable" : filesReady ? "Files selected" : "Waiting for a file"));
  document.querySelectorAll("[data-mode], [data-task], #upload-form input, #remove-file").forEach(el => { el.disabled = busy; });
  $("weights-status").classList.toggle("ready", !!installed);
  $("weights-status").textContent = installed ? "Ready to analyze" : "Analysis unavailable. See README for setup instructions.";
  $("description-availability").textContent = serverStatus?.description?.available
    ? "Generated locally from selected image slices. The draft requires radiologist review."
    : serverStatus?.description?.message || "Checking local description availability…";
  chatReady();
}
function configure() {
  const brain = mode === "brain", dual = brain && task === "idh";
  document.querySelectorAll("[data-mode]").forEach(el => {
    el.classList.toggle("selected", el.dataset.mode === mode);
    el.setAttribute("aria-pressed", String(el.dataset.mode === mode));
  });
  document.querySelectorAll("[data-task]").forEach(el => {
    el.classList.toggle("selected", el.dataset.task === task);
    el.setAttribute("aria-pressed", String(el.dataset.task === task));
  });
  $("brain-task").hidden = $("preprocess-row").hidden = !brain;
  $("dropzone").hidden = dual;
  $("dual-inputs").hidden = !dual;
  $("file").accept = brain ? ".nii,.nii.gz" : ".zip";
  $("drop-title").textContent = brain ? "Drop a T1-weighted NIfTI" : "Drop a ZIP containing DICOM series";
  $("drop-description").textContent = brain ? "One preprocessed 3D volume" : "One MRI study of one knee";
  const max = serverStatus?.max_upload_mb || 512;
  $("file-format").textContent = (brain ? "NIfTI .nii / .nii.gz" : "DICOM in ZIP") + ` · up to ${max} MB`;
  $("task-description").textContent = task === "mci"
    ? "Non-contrast T1. Binary MCI classification against a control group; not a general screen for brain diseases."
    : "FLAIR + contrast-enhanced T1 in the same space. Estimates IDH-mutant / wild-type status in glioma; not a test for tumor presence.";
  $("input-note").replaceChildren();
  if (brain) $("input-note").textContent = "Brain scans must be preprocessed with N4 correction, template registration, and skull stripping.";
  else {
    $("input-note").append("Preserve spatial DICOM tags. If fat suppression is not specified in the tags, add this table to your ZIP: ");
    const code = document.createElement("code"); code.textContent = "series.csv";
    $("input-note").append(code, ".");
  }
  ready();
}
async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error();
    serverStatus = await response.json();
    configure();
  } catch {
    serverStatus = null;
    $("weights-status").textContent = "Cannot connect to the local server.";
    $("analyze-button").disabled = true;
    chatReady();
  }
}
document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
  if (busy) return;
  if (mode !== button.dataset.mode) {
    mode = button.dataset.mode; clearFiles(); showError(""); configure();
  }
  if (button.classList.contains("mode-card")) {
    $("analysis").scrollIntoView({behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start"});
  }
}));
document.querySelectorAll("[data-task]").forEach(button => button.addEventListener("click", () => {
  if (busy || task === button.dataset.task) return;
  task = button.dataset.task; clearFiles(); showError(""); configure();
}));
function fileChanged() {
  const file = $("file").files[0];
  $("selected-file").hidden = !file;
  if (file) {
    $("file-name").textContent = file.name;
    $("file-size").textContent = `${(file.size / 1024 ** 2).toFixed(1)} MB`;
  }
  showError(""); ready();
}
$("file").addEventListener("change", fileChanged);
$("remove-file").addEventListener("click", () => { $("file").value = ""; fileChanged(); });
for (const id of ["flair", "t1ce"]) $(id).addEventListener("change", () => {
  $(id + "-name").textContent = $(id).files[0]?.name || ".nii or .nii.gz";
  showError(""); ready();
});
$("preprocessed").addEventListener("change", ready);
for (const event of ["dragenter", "dragover"]) $("dropzone").addEventListener(event, e => {
  e.preventDefault(); if (!busy) $("dropzone").classList.add("dragover");
});
for (const event of ["dragleave", "drop"]) $("dropzone").addEventListener(event, e => {
  e.preventDefault(); $("dropzone").classList.remove("dragover");
});
$("dropzone").addEventListener("drop", e => {
  if (busy) return;
  if (e.dataTransfer.files.length !== 1) return showError("Drop one file.");
  const transfer = new DataTransfer(); transfer.items.add(e.dataTransfer.files[0]);
  $("file").files = transfer.files; fileChanged();
});
function upload(data) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open("POST", "/api/jobs");
    xhr.responseType = "json";
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) $("progress-title").textContent = `Uploading scans · ${Math.round(e.loaded / e.total * 100)}%`;
    };
    xhr.onerror = () => reject(new Error("Could not connect to the local server."));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error(typeof xhr.response?.detail === "string" ? xhr.response.detail : "Check the files and request parameters."));
    };
    xhr.send(data);
  });
}
async function pollJob(id) {
  for (;;) {
    const response = await fetch(`/api/jobs/${id}`);
    if (!response.ok) throw new Error("The job is unavailable. The server may have restarted.");
    const job = await response.json();
    $("progress-title").textContent = job.status === "queued" ? "Queued"
      : job.status === "running" ? "Analyzing your study"
      : job.status === "complete" ? "Complete" : "Processing stopped";
    if (job.status === "complete") return job.result;
    if (job.status === "error") throw new Error(job.error);
    await new Promise(resolve => setTimeout(resolve, 1200));
  }
}
function renderResult(data) {
  result = data;
  $("result-time").textContent = `${data.elapsed_seconds} s`;
  $("preview").src = data.preview;
  $("probability-help").textContent = data.probability_type === "multilabel"
    ? "Findings are independent: multiple abnormalities may coexist. The estimates do not add up to 100%."
    : "Two mutually exclusive classes for the selected task. The control class does not rule out other conditions.";
  $("probability-bars").replaceChildren();
  for (const item of [...data.probabilities].sort((a,b) => b.probability - a.probability)) {
    const row = document.createElement("div"); row.className = "probability-row";
    const label = document.createElement("div"); label.className = "probability-label";
    const name = document.createElement("span"); name.textContent = item.label;
    const value = document.createElement("strong"); value.textContent = (item.probability * 100).toFixed(1) + "%";
    label.append(name, value);
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("div"); fill.className = "bar-fill"; fill.style.width = `${item.probability * 100}%`;
    track.append(fill); row.append(label, track); $("probability-bars").append(row);
  }
  $("result-warnings").replaceChildren();
  for (const warning of data.warnings) {
    const p = document.createElement("p"); p.textContent = "ⓘ " + warning; $("result-warnings").append(p);
  }
  $("results").hidden = false;
  renderDescription(data.description);
  $("results").scrollIntoView({behavior: "smooth", block: "start"});
}
function renderDescription(description) {
  const show = description && description.status !== "not_requested";
  $("description-panel").hidden = !show;
  $("description-content").replaceChildren();
  $("description-coverage").textContent = "";
  $("copy-description").hidden = true;
  descriptionText = "";
  if (!show) return;
  if (description.status !== "complete") {
    $("description-status").textContent = description.message || "A written image description could not be generated. Probability estimates are shown below.";
    return;
  }
  $("description-status").textContent = "Experimental draft from selected slices; not a final radiology report.";
  const titles = {technique: "Images reviewed", observations: "Observations", impression: "Interpretation", limitations: "Limitations"};
  const copied = ["Study description — experimental draft"];
  for (const [key, title] of Object.entries(titles)) {
    const section = document.createElement("div"); section.className = "description-section";
    const heading = document.createElement("h4"); heading.textContent = title; section.append(heading);
    const value = description.sections[key];
    copied.push("\n" + title);
    if (Array.isArray(value)) {
      const list = document.createElement("ul");
      for (const line of value) { const item = document.createElement("li"); item.textContent = line; list.append(item); copied.push("- " + line); }
      section.append(list);
    } else { const text = document.createElement("p"); text.textContent = value; section.append(text); copied.push(value); }
    $("description-content").append(section);
  }
  const slices = description.coverage.reduce((n, panel) => n + panel.sampled_frames, 0);
  $("description-coverage").textContent = `Based on ${slices} sampled slice views across ${description.panel_count} image panels. The full study was not reviewed slice by slice.`;
  descriptionText = copied.join("\n");
  $("copy-description").hidden = false;
}
$("upload-form").addEventListener("submit", async e => {
  e.preventDefault(); if (busy || $("analyze-button").disabled) return;
  showError("");
  const files = mode === "brain" && task === "idh" ? [$("flair").files[0], $("t1ce").files[0]] : [$("file").files[0]];
  if (files.some(f => !f)) return showError("Select the required files.");
  if (files.reduce((sum,f) => sum + f.size,0) > (serverStatus?.max_upload_mb || 512) * 1024 ** 2) return showError("The combined file size exceeds the upload limit.");
  if (mode === "knee" && !files[0].name.toLowerCase().endsWith(".zip")) return showError("Knee mode requires a ZIP containing DICOM files.");
  if (mode === "brain" && files.some(f => !/\.nii(\.gz)?$/i.test(f.name))) return showError("Brain mode requires NIfTI files.");
  if (jobId && result) await fetch(`/api/jobs/${jobId}`, {method:"DELETE"}).catch(() => {});
  jobId = null; result = null; $("results").hidden = true;
  busy = true; $("progress").hidden = false; $("progress-title").textContent = "Preparing upload"; ready();
  const data = new FormData(); data.set("mode", mode); data.set("task", task); data.set("preprocessed", String($("preprocessed").checked));
  data.set("include_description", String($("include-description").checked));
  if (mode === "brain" && task === "idh") { data.set("flair", files[0]); data.set("t1ce", files[1]); }
  else data.set("file", files[0]);
  try {
    const job = await upload(data); jobId = job.id;
    renderResult(await pollJob(jobId));
  } catch (error) { showError(error.message); }
  finally { busy = false; $("progress").hidden = true; ready(); }
});
$("download-result").addEventListener("click", () => {
  if (!result) return;
  const {preview, model, source, ...report} = result;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type:"application/json"}));
  const a = document.createElement("a"); a.href = url; a.download = `mri-doctor-${result.task}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("copy-description").addEventListener("click", async () => {
  if (!descriptionText) return;
  try { await navigator.clipboard.writeText(descriptionText); $("copy-description").textContent = "Copied ✓";
    setTimeout(() => { $("copy-description").textContent = "Copy text ↗"; }, 1800);
  } catch { showError("Could not copy the description. You can download it in the JSON report."); }
});
$("forget-result").addEventListener("click", async () => {
  if (busy) return;
  if (jobId) await fetch(`/api/jobs/${jobId}`, {method:"DELETE"}).catch(() => {});
  jobId = null; result = null; $("preview").removeAttribute("src"); $("probability-bars").replaceChildren();
  $("results").hidden = true; clearFiles(); ready();
  renderDescription(null);
});
function chatReady() {
  const available = !!serverStatus?.chat?.available;
  $("chat-send").disabled = chatBusy || busy || !available || !$("chat-input").value.trim();
  $("clear-chat").disabled = chatBusy;
  $("chat-status").textContent = chatBusy ? "Thinking…" : busy ? "Available after study analysis" : available
    ? "Local assistant · Lite" : serverStatus?.chat?.message || "The local assistant is unavailable.";
  document.querySelectorAll("#chat-suggestions button").forEach(button => { button.disabled = chatBusy || busy || !available; });
}
function addChatMessage(role, text) {
  $("chat-welcome").hidden = true;
  $("chat-suggestions").hidden = true;
  const article = document.createElement("article"); article.className = `chat-message chat-${role}`;
  const name = document.createElement("span"); name.className = "chat-speaker"; name.textContent = role === "user" ? "You" : "Assistant";
  const body = document.createElement("p"); body.textContent = text; body.dir = "auto";
  article.append(name, body); $("chat-messages").append(article);
  $("chat-messages").scrollTop = $("chat-messages").scrollHeight;
  return article;
}
$("chat-input").addEventListener("input", chatReady);
$("chat-input").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("chat-form").requestSubmit(); }
});
document.querySelectorAll("#chat-suggestions button").forEach(button => button.addEventListener("click", () => {
  $("chat-input").value = button.textContent; chatReady(); $("chat-form").requestSubmit();
}));
$("chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const text = $("chat-input").value.trim();
  if (!text || chatBusy || busy || !serverStatus?.chat?.available) return;
  const message = {role: "user", content: text};
  // Keep at most four previous pairs; no localStorage, cookies, or server-side history.
  const messages = [...chatHistory.slice(-8), message];
  const pending = addChatMessage("user", text);
  $("chat-input").value = ""; $("chat-error").hidden = true; chatBusy = true; chatReady();
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 135000);
  try {
    const response = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({messages}), signal:controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The assistant could not reply. Please try a shorter question.");
    addChatMessage("assistant", data.reply);
    chatHistory = [...messages, {role: "assistant", content: data.reply.slice(0,1500)}];
    if (data.shortened) { const note = document.createElement("p"); note.className = "chat-note"; note.textContent = "Reply shortened for Lite mode."; $("chat-messages").append(note); }
  } catch (error) {
    pending.remove(); $("chat-input").value = text;
    if (!chatHistory.length) { $("chat-welcome").hidden = false; $("chat-suggestions").hidden = false; }
    $("chat-error").textContent = error.name === "AbortError" ? "The assistant took too long. Please try again." : error.message;
    $("chat-error").hidden = false;
  } finally { clearTimeout(timeout); chatBusy = false; chatReady(); $("chat-input").focus(); }
});
$("clear-chat").addEventListener("click", () => {
  if (chatBusy) return;
  chatHistory = []; document.querySelectorAll("#chat-messages article, #chat-messages .chat-note").forEach(node => node.remove());
  $("chat-welcome").hidden = false; $("chat-suggestions").hidden = false;
  $("chat-input").value = ""; $("chat-error").hidden = true; chatReady();
});
configure(); refreshStatus(); setInterval(() => { if (!busy) refreshStatus(); }, 15000);

// Keep every section readable without JavaScript, and honor reduced-motion preferences.
if ("IntersectionObserver" in window && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.documentElement.classList.add("motion-ready");
  const revealObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    }
  }, {threshold: 0.06});
  document.querySelectorAll(".reveal").forEach(section => revealObserver.observe(section));
}
