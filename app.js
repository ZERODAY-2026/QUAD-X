const $ = (id) => document.getElementById(id);
const ORDER = { High: 0, Medium: 1, Low: 2 };
const MAX_BYTES = 10 * 1024 * 1024;
let sid = null, timer = null, file = null, pageCount = 0, docName = "", current = null;

// textContent only: report text comes from a PDF and an AI model, never trust it as HTML.
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
const notice = (kind, msg) => $("notices").append(el("div", `notice ${kind}`, msg));

async function api(url, options) {
  let r;
  try { r = await fetch(url, options); }
  catch { throw new Error("Couldn't reach the server. Is it running?"); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || "Request failed.");
  return data;
}

function setProgress(fraction, label) {
  $("progress").hidden = false;
  $("bar-fill").style.width = `${Math.round(fraction * 100)}%`;
  $("progress-label").textContent = label;
}

// ---------- File selection (click or drag and drop) ----------

function choose(f) {
  $("notices").replaceChildren();
  if (!f) return;
  if (!f.name.toLowerCase().endsWith(".pdf")) return notice("error", "Please choose a PDF file.");
  if (f.size > MAX_BYTES) return notice("error", "This file is over 10 MB. Please choose a smaller PDF.");
  file = f;
  $("file-hint").textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
  $("analyse").disabled = false;
}

$("pick").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => choose(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));

// ---------- Analyse ----------

$("analyse").addEventListener("click", async () => {
  const prompt = $("prompt").value.trim();
  $("notices").replaceChildren();
  if (!file) return notice("error", "Please choose a PDF document first.");
  if (!prompt) return notice("error", "Please enter what should be reviewed.");

  const form = new FormData();
  form.append("file", file);
  form.append("prompt", prompt);
  $("analyse").disabled = true;
  setProgress(0, "Extracting text (scanned pages take longer)...");
  try {
    const info = await api("/api/analyse", { method: "POST", body: form });
    sid = info.id; pageCount = info.pages; docName = file.name;
    if (info.ocr_pages.length) notice("info", `Page(s) ${info.ocr_pages.join(", ")} were scanned images, so their text was read with OCR. OCR can make mistakes, so check the original.`);
    if (info.skipped.length) notice("warn", `No text found on page(s) ${info.skipped.join(", ")}. They were left out of the analysis.`);
    if (info.long) notice("info", "This is a long document, so it is analysed in sections. That can take several minutes.");
    $("long-note").hidden = !info.long;
    setProgress(0, "Analysing your document...");
    timer = setInterval(poll, 1500);
  } catch (error) { fail(error.message); }
});

function fail(message) {
  clearInterval(timer);
  $("progress").hidden = true;
  $("analyse").disabled = !file;
  notice("error", message);
}

async function poll() {
  try {
    const s = await api(`/api/status/${sid}`);
    if (s.status === "running") {
      const part = Math.min(s.done + 1, s.total);
      setProgress(s.done / s.total, s.total > 1 ? `Analysing part ${part} of ${s.total}...` : "Analysing your document...");
    } else if (s.status === "error") {
      fail(s.error);
    } else {
      clearInterval(timer);
      $("progress").hidden = true;
      $("analyse").disabled = false;
      $("empty").hidden = true;
      $("followup-area").hidden = false;
      $("st-docs").textContent = "1";
      $("st-pages").textContent = pageCount;
      show(s.report);
    }
  } catch (error) { fail(error.message); }
}

// ---------- Report ----------

function show(report) {
  current = report;
  $("st-issues").textContent = report.findings.length;
  $("meta").textContent = `${docName} · ${pageCount} pages`;
  draw();
}

function draw() {
  const root = $("report");
  const term = $("search").value.trim().toLowerCase();
  root.replaceChildren(el("h3", "", "Summary"), el("p", "", current.summary));

  const findings = current.findings
    .filter((f) => !term || [f.title, f.category, f.quotation, f.explanation].join(" ").toLowerCase().includes(term))
    .sort((a, b) => (ORDER[a.severity] ?? 3) - (ORDER[b.severity] ?? 3));

  if (!current.findings.length) return root.append(el("div", "notice", "No relevant risks were identified for this request."));
  if (!findings.length) return root.append(el("div", "notice", "No findings match your search."));
  root.append(el("h3", "", `Findings (${findings.length})`), el("p", "muted", "Severity is a preliminary assessment, not a legal determination."));
  findings.forEach((f) => root.append(card(f)));
}
$("search").addEventListener("input", () => current && draw());

function card(f) {
  const d = el("details", f.severity);
  d.open = f.severity === "High";
  const sum = el("summary");
  sum.append(el("span", "sev", f.severity), f.title);
  d.append(sum);

  const body = el("div", "finding-body");
  body.append(el("p", "muted", f.page != null ? `${f.category} · Page ${f.page}` : f.category));
  body.append(el("h4", "", "From the document"), el("blockquote", "", f.quotation));

  if (!f.quote_verified) {
    body.append(el("div", "notice warn", "Couldn't find this exact wording in the document. It may be reworded or wrong. Check the original before relying on it."));
  } else {
    const label = el("label", "src");
    const box = document.createElement("input");
    box.type = "checkbox";
    label.append(box, "Show in original document");
    const holder = el("div");
    box.addEventListener("change", () => source(box.checked, holder, f));
    body.append(label, holder);
  }
  body.append(el("h4", "", "What it means"), el("p", "", f.explanation));
  body.append(el("h4", "", "Suggested action"), el("p", "", f.suggested_action));
  d.append(body);
  return d;
}

function source(show, holder, f) {
  holder.replaceChildren();
  if (!show) return;
  const params = new URLSearchParams({ quote: f.quotation });
  if (f.page != null) params.set("page", f.page);
  const img = el("img");
  img.alt = "The highlighted passage in the original PDF";
  img.onerror = () => holder.replaceChildren(el("p", "muted", "Couldn't locate this passage in the PDF layout (it may span two pages, or be on a scanned page)."));
  img.src = `/api/source/${sid}?${params}`;
  holder.append(img);
}

// ---------- Follow-up and reset ----------

$("send-followup").addEventListener("click", async () => {
  const prompt = $("followup").value.trim();
  $("notices").replaceChildren();
  if (!prompt) return notice("warn", "Please enter a follow-up prompt.");
  const b = $("send-followup");
  b.disabled = true; b.textContent = "Updating...";
  try {
    const data = await api(`/api/follow-up/${sid}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }),
    });
    show(data.report);
    $("followup").value = "";
  } catch (error) { notice("error", error.message); }
  finally { b.disabled = false; b.textContent = "Update report"; }
});

$("reset").addEventListener("click", async () => {
  if (sid) await fetch(`/api/session/${sid}`, { method: "DELETE" }).catch(() => {});
  sid = null; file = null; current = null;
  $("report").replaceChildren();
  $("notices").replaceChildren();
  $("empty").hidden = false;
  $("followup-area").hidden = true;
  $("followup").value = "";
  $("file").value = "";
  $("search").value = "";
  $("file-hint").textContent = "PDF · Maximum file size: 10 MB";
  $("meta").textContent = "Document name · Page · Category";
  ["st-docs", "st-issues", "st-pages"].forEach((id) => ($(id).textContent = "—"));
  $("analyse").disabled = true;
});
